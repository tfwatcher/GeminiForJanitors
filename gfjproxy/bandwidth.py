"""Bandwidth management."""

import datetime
import threading
from dataclasses import dataclass
from time import perf_counter

import httpx2
import redis
import redis.exceptions
import redis.lock

from ._globals import BANDWIDTH_WARNING, PRODUCTION, RENDER_API_KEY, RENDER_SERVICE_ID
from .http_client import http_client
from .logging import xlog
from .maintenance_logging import (
    last_bandwidth_success_age,
    log_due,
    mark_bandwidth_failed,
    mark_bandwidth_succeeded,
)
from .start_time import START_TIME
from .storage import get_redis_client, storage


@dataclass(frozen=True, kw_only=True)
class BandwidthUsage:
    total: int = -1
    "Total bandwidth usage in MiB."

    def __bool__(self):
        return self.total >= 0


def _report_query_failure(reason: str) -> None:
    mark_bandwidth_failed()
    xlog(
        None,
        f"Bandwidth: query failed ({reason}); last successful check "
        f"{last_bandwidth_success_age()}",
    )


def _query_bandwidth_usage() -> BandwidthUsage | None:
    if not RENDER_API_KEY or not RENDER_SERVICE_ID:
        total = perf_counter() - START_TIME
        if log_due("bandwidth-mock"):
            xlog(None, f"Bandwidth: using mock total {total:.2f} MiB")
        return BandwidthUsage(total=int(total))

    end_time = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)

    start_time = end_time.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    try:
        response = http_client.get(
            "https://api.render.com/v1/metrics/bandwidth",
            params={
                "resource": RENDER_SERVICE_ID,
                "endTime": end_time.isoformat() + "Z",
                "startTime": start_time.isoformat() + "Z",
            },
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {RENDER_API_KEY}",
            },
        )
    except httpx2.HTTPError as exc:
        _report_query_failure(type(exc).__name__)
        return None

    if response.status_code != 200:
        _report_query_failure(f"HTTP {response.status_code} {response.reason_phrase}")
        return None

    try:
        response_json = response.json()
    except ValueError:
        _report_query_failure("invalid JSON from Render")
        return None

    if (
        isinstance(response_json, list)
        and len(response_json) == 1
        and isinstance((usage := response_json[0]), dict)
        and usage.get("unit") == "mb"  # Render seems to always return MiB
        and isinstance((values := usage.get("values")), list)
    ):
        try:
            total = sum(float(value.get("value", 0.0)) for value in values)
        except (AttributeError, TypeError, ValueError):
            _report_query_failure("malformed bandwidth values")
            return None
        return BandwidthUsage(total=int(total))

    _report_query_failure("unexpected bandwidth response")
    return None


def _update_bandwidth_usage(client: redis.Redis, lock: redis.lock.Lock) -> None:
    try:
        if result := _query_bandwidth_usage():
            # Cache refreshes every five minutes, regardless of log frequency.
            client.set(":bandwidth-cache", result.total)
            client.set(":bandwidth-cache-fresh", "<3", ex=300)

            announcement = ""
            if 0 < BANDWIDTH_WARNING <= result.total:
                announcement = "\n".join(
                    (
                        f"Bandwidth quota is above {BANDWIDTH_WARNING / 1024:.1f} GiB.",
                        "Please consider using another URL.",
                    )
                )

            previous_announcement = storage.announcement
            storage.announcement = announcement

            recovered = mark_bandwidth_succeeded()
            if recovered:
                log_due("bandwidth-success")  # Restart the hourly summary window.
                xlog(None, f"Bandwidth: query recovered; {result.total:.2f} MiB")
            elif log_due("bandwidth-success"):
                xlog(
                    None,
                    f"Bandwidth: query succeeded: {result.total:.2f} MiB "
                    "(hourly summary)",
                )

            if bool(previous_announcement) != bool(announcement):
                state = "set" if announcement else "clear"
                xlog(None, f"Bandwidth: announcement {state}")
    finally:
        try:
            lock.release()
        except redis.exceptions.LockNotOwnedError:
            pass


def bandwidth_usage() -> BandwidthUsage:
    client = get_redis_client()

    # Redis storage is required; Render API credentials are production-only.
    if client is None or (PRODUCTION and not RENDER_API_KEY):
        return BandwidthUsage()

    if not client.get(":bandwidth-cache-fresh"):
        lock = client.lock(
            name=":bandwidth-cache-lock",
            timeout=30,
            thread_local=False,
        )
        if lock.acquire(blocking=False):
            threading.Thread(
                target=_update_bandwidth_usage,
                args=(client, lock),
                daemon=True,
            ).start()

    if isinstance((cache := client.get(":bandwidth-cache")), bytes):
        return BandwidthUsage(total=int(cache))
    return BandwidthUsage()
