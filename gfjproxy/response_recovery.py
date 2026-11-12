"""Explicit request controls and recovery of the latest successful GFJ reply.

Uses GFJ's own command tokenizer so //recover and //recovery are recognized
at the start of a message, including immediately after a JanitorAI speaker
prefix. Standalone command lines and commands at the very end work too.

//notimeout is a one-message override for the DIRECT Google provider, not a
persistent setting. //stopwait (alias //notimeoutdisable) stops the currently
active wait for the same proxy user; controls are never sent to Google.
"""

import json
import os
import re

from redis.exceptions import RedisError

from .commands import _tokenize  # Same command tokenizer used by GFJ's parse_message.
from .storage import get_redis_client
from .xuiduser import XUID

_CONTROL_WORDS = frozenset(("recover", "recovery", "notimeout", "stopwait", "notimeoutdisable"))
# JanitorAI may prepend "Character: " to user reactions. A speaker label is
# allowed, but we do not treat arbitrary mentions within roleplay as commands.
_SPEAKER_LABEL = re.compile(r"[^\n:]{1,80}:[ \t]*$")
_MAX_RESPONSE_BYTES = 128 * 1024


def _enabled() -> bool:
    return os.getenv("GFJPROXY_RESPONSE_RECOVERY", "1").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _ttl() -> int | None:
    """No expiry by default; positive values set an optional TTL."""
    try:
        seconds = int(os.getenv("GFJPROXY_RESPONSE_RECOVERY_TTL", "0"))
    except ValueError:
        return None
    if seconds <= 0:
        return None
    return max(60, min(7200, seconds))


def _remove_control_commands(content: str) -> tuple[set[str], str]:
    """Extract edge/standalone GFJ control commands and strip only those tokens.

    The common tokenizer recognizes the command word exactly as the built-in
    commands do. Placement checks avoid triggering on dialogue or quoted code
    in the middle of a line. A speaker prefix is accepted for JanitorAI.
    """
    tokens = list(_tokenize(content))
    positions = []
    cursor = 0
    for token in tokens:
        positions.append(cursor)
        cursor += len(token)

    found: set[str] = set()
    spans: list[tuple[int, int]] = []
    for i in range(len(tokens) - 1):
        if tokens[i] != "//":
            continue
        name = tokens[i + 1].lower()
        if name not in _CONTROL_WORDS:
            continue

        begin = positions[i]
        end = positions[i + 1] + len(tokens[i + 1])
        line_start = content.rfind("\n", 0, begin) + 1
        line_prefix = content[line_start:begin]
        # Built-in GFJ commands can appear in a sequence on the same line.
        # Ignore already recognized controls while checking the next one.
        for earlier_begin, earlier_end in reversed(spans):
            if line_start <= earlier_begin and earlier_end <= begin:
                left = earlier_begin - line_start
                right = earlier_end - line_start
                line_prefix = line_prefix[:left] + line_prefix[right:]
        clean_prefix = line_prefix.strip(" \t\r\ufeff\u200b")
        # Accept the beginning of any line (including with JanitorAI header
        # lines above it), a speaker label, or the exact end of the message.
        at_line_start = not clean_prefix
        after_speaker = bool(_SPEAKER_LABEL.fullmatch(line_prefix))
        at_message_end = (
            not content[end:].strip()
            and begin > 0
            and content[begin - 1].isspace()
        )
        if not (at_line_start or after_speaker or at_message_end):
            continue
        # Do not eat a prefix of another command, e.g. //recoveryX.
        # The GFJ tokenizer produces the entire word as one token.
        found.add(name)
        spans.append((begin, end))

    if not spans:
        return found, content
    for begin, end in reversed(spans):
        content = content[:begin] + content[end:]
    return found, content


def _remove_recovery_commands(content: str) -> tuple[bool, str]:
    """Compatibility helper retained for older recovery tests and callers."""
    found, cleaned = _remove_control_commands(content)
    return bool(found & {"recover", "recovery"}), cleaned


def extract_control_requests_with_stop(request_data: dict) -> tuple[bool, bool, bool, dict]:
    """Return (recover, notimeout, stopwait, cleaned request).

    Only the latest user message can issue controls. Stops always take
    precedence at the route layer, without trying to generate a new reply.
    """
    messages = request_data.get("messages")
    if not isinstance(messages, list):
        return False, False, False, request_data
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return False, False, False, request_data
        markers, cleaned = _remove_control_commands(content)
        if not markers:
            return False, False, False, request_data
        new_request = dict(request_data)
        new_messages = list(messages)
        new_messages[index] = {**message, "content": cleaned}
        new_request["messages"] = new_messages
        return (
            bool(markers & {"recover", "recovery"}),
            "notimeout" in markers,
            bool(markers & {"stopwait", "notimeoutdisable"}),
            new_request,
        )
    return False, False, False, request_data


def extract_control_requests(request_data: dict) -> tuple[bool, bool, dict]:
    """Backward-compatible recovery/no-timeout extraction interface."""
    recover, no_timeout, _, cleaned = extract_control_requests_with_stop(request_data)
    return recover, no_timeout, cleaned


def recovery_request(request_data: dict) -> tuple[bool, dict]:
    """Backward-compatible entry point for existing recovery callers."""
    recover, _, cleaned = extract_control_requests(request_data)
    return recover, cleaned


def strip_saved_recovery_markers(messages: list) -> None:
    """Strip historical recovery/timeout/cancellation controls from model context."""
    for message in messages:
        if getattr(message, "role", None) == "user":
            content = getattr(message, "content", None)
            if isinstance(content, str):
                _, message.content = _remove_control_commands(content)


def _key(user: XUID) -> str:
    # Compatible with the prior recovery packages; no cache migration.
    return f":response-recovery:last:{user!r}"


def remember_response(user: XUID, _request_data: dict, reply: str) -> bool:
    """Best-effort atomic overwrite of the latest successful reply."""
    if not _enabled() or not reply:
        return False
    if len(reply.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        return False
    try:
        payload = json.dumps({"reply": reply}, ensure_ascii=False).encode("utf-8")
        client = get_redis_client()
        if client is not None:
            return bool(client.set(_key(user), payload, ex=_ttl()))
    except (RedisError, TypeError, ValueError):
        pass
    return False


def restore_response(user: XUID, _request_data: dict) -> str | None:
    """Return the most recently saved reply, regardless of prompt fingerprint."""
    if not _enabled():
        return None
    try:
        client = get_redis_client()
        if client is not None:
            value = client.get(_key(user))
            if isinstance(value, (bytes, str)):
                payload = json.loads(value)
                if isinstance(payload, dict):
                    reply = payload.get("reply")
                    if isinstance(reply, str) and reply:
                        return reply
    except (RedisError, TypeError, ValueError, UnicodeError):
        pass
    return None
