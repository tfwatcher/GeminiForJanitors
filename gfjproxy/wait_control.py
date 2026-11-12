"""Cross-worker cancellation control for a single active //notimeout request.

An authenticated //stopwait (or //notimeoutdisable) call sets a Redis
cancellation signal for the *currently active* wait of the same GFJ user.
The waiting request polls this signal and stops its own HTTP call. A unique
request token and atomic Lua compare-and-delete prevent stale commands from
contaminating later requests. No API keys or prompt content are stored.
"""

import secrets

from redis.exceptions import RedisError

from .storage import get_redis_client
from .xuiduser import XUID

WAIT_LEASE_SECONDS = 240
WAIT_CANCEL_SECONDS = 300

_STOP_WAIT_LUA = """
local token = redis.call('get', KEYS[1])
if not token then return 0 end
redis.call('set', KEYS[2], token, 'EX', ARGV[1])
return 1
"""

_RELEASE_WAIT_LUA = """
if redis.call('get', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('del', KEYS[1])
if redis.call('get', KEYS[2]) == ARGV[1] then
    redis.call('del', KEYS[2])
end
return 1
"""


class WaitControlUnavailable(Exception):
    """Redis isn't available, so safe cross-worker cancellation isn't possible."""


def _keys(user: XUID) -> tuple[str, str]:
    uid = repr(user)
    return f":notimeout:active:{uid}", f":notimeout:cancel:{uid}"


def _redis():
    client = get_redis_client()
    if client is None:
        raise WaitControlUnavailable("Redis is required for cancellable //notimeout")
    return client


def begin_wait(user: XUID) -> str | None:
    """Register one owned active wait (only after the user's lock is held).

    Returns a random per-request token or None if an active wait still exists.
    """
    client = _redis()
    token = secrets.token_urlsafe(24)
    active, _ = _keys(user)
    try:
        if not client.set(active, token, ex=WAIT_LEASE_SECONDS, nx=True):
            return None
    except RedisError as exc:
        raise WaitControlUnavailable("Cannot register active request") from exc
    return token


def stop_wait(user: XUID) -> bool:
    """Signal cancellation of ONLY the currently registered request.

    Called *before* the ordinary per-user lock, allowing it to work while
    another request is still awaiting Google. The API key defines the user.
    """
    client = _redis()
    active, cancelled = _keys(user)
    try:
        return bool(client.eval(
            _STOP_WAIT_LUA, 2, active, cancelled, WAIT_CANCEL_SECONDS
        ))
    except RedisError as exc:
        raise WaitControlUnavailable("Cannot signal cancellation") from exc


def wait_should_stop(user: XUID, token: str) -> bool:
    """Poll for cancellation or lost request ownership.

    Fail closed if Redis is unreachable: continuing to wait without a safe
    lock/cancellation channel would risk overlapping generations.
    """
    try:
        active, cancelled = _keys(user)
        active_token, cancel_token = _redis().mget(active, cancelled)
        expected = token.encode("utf-8")
        return active_token != expected or cancel_token == expected
    except (WaitControlUnavailable, RedisError):
        return True


def end_wait(user: XUID, token: str) -> None:
    """Remove only our own active/cancel state; never erase a newer request."""
    try:
        active, cancelled = _keys(user)
        _redis().eval(_RELEASE_WAIT_LUA, 2, active, cancelled, token)
    except (WaitControlUnavailable, RedisError):
        # A stale marker expires within the same window as the lock lease.
        pass
