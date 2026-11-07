"""Explicit recovery of the most recent successful reply for a proxy user.

Recovery is enabled by default. ``//recover`` at the beginning or end of the
latest user message requests the cached reply regardless of prompt changes.
Ordinary messages and Regenerate always call the provider. Each new successful
reply overwrites the previous one. Entries have no fixed expiry by default,
but Redis/Valkey can still evict them under memory pressure.
"""

import json
import os
import re

from redis.exceptions import RedisError

from .storage import get_redis_client
from .xuiduser import XUID

# Require //recover to be a separate whitespace-delimited command at an edge
# of the latest user message, not an arbitrary substring inside roleplay text.
# Both a separate line and the same line (separated by spaces) are supported.
_RECOVER_PREFIX = re.compile(r"\A\s*//recover(?=\s|\Z)\s*", re.IGNORECASE)
_RECOVER_SUFFIX = re.compile(r"\s+//recover\s*\Z", re.IGNORECASE)
_MAX_RESPONSE_BYTES = 128 * 1024


def _enabled() -> bool:
    return os.getenv("GFJPROXY_RESPONSE_RECOVERY", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _ttl() -> int | None:
    """No expiry by default; positive values permit an optional TTL."""
    try:
        seconds = int(os.getenv("GFJPROXY_RESPONSE_RECOVERY_TTL", "0"))
    except ValueError:
        return None
    if seconds <= 0:
        return None
    return max(60, min(7200, seconds))


def _remove_recovery_commands(content: str) -> tuple[bool, str]:
    """Remove explicitly placed edge markers without touching interior prose."""
    original = content
    while True:
        if match := _RECOVER_PREFIX.match(content):
            content = content[match.end():]
        elif match := _RECOVER_SUFFIX.search(content):
            content = content[:match.start()]
        else:
            break
    return content != original, content


def recovery_request(request_data: dict) -> tuple[bool, dict]:
    """Identify //recover in the last user message, without mutating input."""
    messages = request_data.get("messages")
    if not isinstance(messages, list):
        return False, request_data

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return False, request_data
        found, cleaned = _remove_recovery_commands(content)
        if not found:
            return False, request_data
        new_request = dict(request_data)
        new_messages = list(messages)
        new_messages[index] = {**message, "content": cleaned}
        new_request["messages"] = new_messages
        return True, new_request

    return False, request_data


def strip_saved_recovery_markers(messages: list) -> None:
    """Keep JanitorAI's previously saved //recover tags out of model prompts."""
    for message in messages:
        if getattr(message, "role", None) == "user":
            content = getattr(message, "content", None)
            if isinstance(content, str):
                _, message.content = _remove_recovery_commands(content)


def _key(user: XUID) -> str:
    # Keep the existing key to allow recovery of replies saved before this update.
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
        pass  # Cache problems must never break a successful reply.
    return False


def restore_response(user: XUID, _request_data: dict) -> str | None:
    """Return latest saved reply; no prompt or provider fingerprint comparison."""
    if not _enabled():
        return None
    try:
        client = get_redis_client()
        if client is not None:
            value = client.get(_key(user))
            if isinstance(value, (bytes, str)):
                payload = json.loads(value)
                # Also accepts legacy records with an unused "signature" field.
                if isinstance(payload, dict):
                    reply = payload.get("reply")
                    if isinstance(reply, str) and reply:
                        return reply
    except (RedisError, TypeError, ValueError, UnicodeError):
        pass
    return None
