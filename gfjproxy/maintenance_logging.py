"""Rate-limited maintenance logs shared across Gunicorn workers via Redis.

Only routine success messages are throttled; failures are always reported.
The keep-awake and bandwidth polling schedules are NOT changed here.
"""

import os
import threading
import time

import redis.exceptions

from .storage import get_redis_client

try:
    LOG_INTERVAL = max(0, int(os.getenv("GFJPROXY_MAINTENANCE_LOG_INTERVAL", "3600")))
except ValueError:
    LOG_INTERVAL = 3600

_LOCK = threading.Lock()
_LOCAL_LAST_LOG: dict[str, float] = {}
_LOCAL_LAST_BANDWIDTH_SUCCESS: float | None = None
_LOCAL_BANDWIDTH_FAILURE: float | None = None


def log_due(category: str) -> bool:
    """Only one success log per interval, even with several Gunicorn workers."""
    if LOG_INTERVAL == 0:
        return True

    client = get_redis_client()
    if client is not None:
        try:
            return bool(
                client.set(
                    f":maintenance-log:{category}",
                    "1",
                    nx=True,
                    ex=LOG_INTERVAL,
                )
            )
        except redis.exceptions.RedisError:
            pass  # Logging should never interrupt actual proxy requests.

    # Development / Redis-failure fallback: process-local throttling only.
    with _LOCK:
        now = time.monotonic()
        if now - _LOCAL_LAST_LOG.get(category, float("-inf")) < LOG_INTERVAL:
            return False
        _LOCAL_LAST_LOG[category] = now
        return True


def last_bandwidth_success_age() -> str:
    """Return human-readable elapsed time since a successful bandwidth check."""
    timestamp = _LOCAL_LAST_BANDWIDTH_SUCCESS
    client = get_redis_client()
    if client is not None:
        try:
            saved = client.get(":bandwidth-last-success-at")
            if saved is not None:
                timestamp = float(saved)
        except (redis.exceptions.RedisError, ValueError, TypeError):
            pass

    if timestamp is None:
        return "no previous success recorded"

    elapsed = max(0, int(time.time() - timestamp))
    if elapsed < 60:
        return f"{elapsed}s ago"
    if elapsed < 3600:
        return f"{elapsed // 60}m {elapsed % 60}s ago"
    return f"{elapsed // 3600}h {(elapsed % 3600) // 60}m ago"


def mark_bandwidth_failed() -> None:
    """Remember a failure so the next success can report recovery."""
    global _LOCAL_BANDWIDTH_FAILURE

    timestamp = time.time()
    with _LOCK:
        if _LOCAL_BANDWIDTH_FAILURE is None:
            _LOCAL_BANDWIDTH_FAILURE = timestamp

    client = get_redis_client()
    if client is not None:
        try:
            client.set(":bandwidth-failure-since", str(timestamp), nx=True)
        except redis.exceptions.RedisError:
            pass


def mark_bandwidth_succeeded() -> bool:
    """Record every successful query; return True if recovering from errors."""
    global _LOCAL_BANDWIDTH_FAILURE, _LOCAL_LAST_BANDWIDTH_SUCCESS

    timestamp = time.time()
    with _LOCK:
        recovered = _LOCAL_BANDWIDTH_FAILURE is not None
        _LOCAL_BANDWIDTH_FAILURE = None
        _LOCAL_LAST_BANDWIDTH_SUCCESS = timestamp

    client = get_redis_client()
    if client is not None:
        try:
            # Keep the marker until consumed; no TTL on last success so the
            # reported elapsed time survives application restarts.
            recovered = bool(client.delete(":bandwidth-failure-since"))
            client.set(":bandwidth-last-success-at", str(timestamp))
        except redis.exceptions.RedisError:
            pass

    return recovered
