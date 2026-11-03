"""Explicit recovery of the latest successful chat reply, without expiration by default.

Enabled by default. An ordinary repeated prompt always generates a fresh reply;
only a final-line ``//recover`` requests the saved answer. Each successful
reply overwrites the previous saved reply for that proxy user. The full request
fingerprint must match, preventing recovery of an unrelated chat response.
"""

import hashlib
import json
import os
import re

from redis.exceptions import RedisError

from .storage import get_redis_client
from .xuiduser import XUID

_RECOVER_LINE = re.compile(r"(?:\r?\n)[ \t]*//recover[ \t]*$")
_MAX_RESPONSE_BYTES = 128 * 1024


def _enabled() -> bool:
    # Enabled by default; set the env var to 0 to disable it explicitly.
    return os.getenv("GFJPROXY_RESPONSE_RECOVERY", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _ttl() -> int | None:
    """Return Redis expiry in seconds, or None to retain until replaced/evicted.

    Zero (the default) disables expiration. Positive values remain bounded to
    60–7200 seconds, as in the previous release.
    """
    try:
        seconds = int(os.getenv("GFJPROXY_RESPONSE_RECOVERY_TTL", "0"))
    except ValueError:
        return None
    if seconds <= 0:
        return None
    return max(60, min(7200, seconds))


def recovery_request(request_data: dict) -> tuple[bool, dict]:
    """Strip explicit marker from last user message without mutating input."""
    messages = request_data.get("messages")
    if not isinstance(messages, list):
        return False, request_data

    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            return False, request_data
        match = _RECOVER_LINE.search(content)
        if match is None:
            return False, request_data
        original = dict(request_data)
        copy_messages = list(messages)
        copy_messages[index] = {**msg, "content": content[:match.start()]}
        original["messages"] = copy_messages
        return True, original

    return False, request_data


def strip_saved_recovery_markers(messages: list) -> None:
    """Keep previous //recover control markers out of future model requests."""
    for msg in messages:
        if getattr(msg, "role", None) == "user":
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                msg.content = _RECOVER_LINE.sub("", content)


def _key(user: XUID) -> str:
    # One entry per proxy-user identity. Never store API keys or
    # plaintext prompt text in Redis keys.
    return f":response-recovery:last:{user!r}"


def _signature(request_data: dict) -> str:
    # 'stream' affects delivery format only, not the content sent to Gemini.
    signature = {k: v for k, v in request_data.items() if k != "stream"}
    raw = json.dumps(
        signature, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def remember_response(user: XUID, request_data: dict, reply: str) -> bool:
    """Best-effort atomic overwrite of the last successful reply for this user."""
    if not _enabled() or not reply:
        return False
    if len(reply.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        return False  # Avoid filling Free Valkey storage.

    try:
        payload = json.dumps(
            {"signature": _signature(request_data), "reply": reply},
            ensure_ascii=False,
        ).encode("utf-8")
        client = get_redis_client()
        if client is not None:
            # SET (without NX) replaces the previous reply, including for an
            # entirely different prompt or chat. Redis keeps it until the
            # next success (or eviction) unless an explicit TTL is configured.
            return bool(client.set(_key(user), payload, ex=_ttl()))
    except (RedisError, TypeError, ValueError):
        pass  # Never break a successful generation because cache saving failed.
    return False


def restore_response(user: XUID, request_data: dict) -> str | None:
    """Return the last reply only for an exact explicit recovery request."""
    if not _enabled():
        return None
    try:
        client = get_redis_client()
        if client is not None:
            value = client.get(_key(user))
            if not isinstance(value, bytes):
                return None
            payload = json.loads(value)
            if (
                isinstance(payload, dict)
                and payload.get("signature") == _signature(request_data)
                and isinstance(payload.get("reply"), str)
            ):
                return payload["reply"]
    except (RedisError, TypeError, ValueError, UnicodeError):
        pass
    return None
