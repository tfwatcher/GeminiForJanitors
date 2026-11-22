import errno
import json
import math
import os
import random
import secrets
import threading
import time
from collections import deque
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from itertools import count
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..wait_control import wait_should_stop
from ..xuiduser import XUID

################################################################################

# Scoped to the active Flask request; never log or persist the API keys.
# The first entry is the normal round-robin key, followed by the other keys.
gemini_key_candidates: ContextVar[tuple[tuple[int, str], ...]] = ContextVar(
    "gfj_gemini_key_candidates", default=()
)
# An opt-in one-message override. This does NOT change PROCESS_TIMEOUT,
# Gunicorn, Render, Google, or JanitorAI's independent timeouts.
gemini_no_timeout: ContextVar[bool] = ContextVar(
    "gfj_gemini_no_timeout", default=False
)
# The token is registered by the Flask route and checked from this request's
# cancellable HTTP wait. No other request can cancel it via a stale token.
gemini_wait_token: ContextVar[str | None] = ContextVar(
    "gfj_gemini_wait_token", default=None
)

# Request-local callback. The cancellable gevent greenlet captures the callback
# before it is spawned, so it receives the same trace without changing the
# //stopwait call interface or leaking trace state to concurrent requests.
_gemini_connection_trace: ContextVar["_ConnectionTrace | None"] = ContextVar(
    "gfj_gemini_connection_trace", default=None
)


class _ConnectionTrace:
    """Bounded, redacted HTTP transport timeline for one Google POST.

    httpx2's `extensions={"trace": callback}` exposes high-level events from
    httpcore2. It does NOT expose every HTTP/2 control frame, nor can it prove
    whether a remote peer, intermediary, or local client initiated the close.
    Store only approved event names, elapsed times and exception class names:
    never stringify raw event dictionaries (which can contain API-key headers,
    request objects, or complete response data).
    """

    def __init__(self):
        self.started = time.monotonic()
        self.events = deque(maxlen=16)
        self.tcp_connected = False
        self.tls_connected = False
        self.request_sent = False
        self.response_headers = False

    def __call__(self, event_name, info):
        # Callback failures must never interrupt actual generation.
        try:
            if not isinstance(event_name, str):
                return
            parts = event_name.split(".")
            if (
                len(parts) != 3
                or parts[0] not in ("connection", "http2", "http11")
                or parts[2] not in ("started", "complete", "failed")
                or not parts[1].replace("_", "").isalnum()
            ):
                return
            if event_name == "connection.connect_tcp.complete":
                self.tcp_connected = True
            elif event_name == "connection.start_tls.complete":
                self.tls_connected = True
            elif event_name in (
                "http2.send_request_body.complete",
                "http11.send_request_body.complete",
            ):
                self.request_sent = True
            elif event_name in (
                "http2.receive_response_headers.complete",
                "http11.receive_response_headers.complete",
            ):
                self.response_headers = True
            elapsed = max(0.0, time.monotonic() - self.started)
            failure_type = ""
            if parts[2] == "failed" and isinstance(info, dict):
                failure = info.get("exception")
                if failure is not None:
                    failure_type = "(" + type(failure).__name__ + ")"
            self.events.append(f"{elapsed:.1f}s:{event_name}{failure_type}")
        except Exception:  # ruff: ignore[BLE001]
            return

    def summary(self, exc):
        """Report where the error was observed, not an unprovable closer."""
        name = type(exc).__name__
        if name == "LocalProtocolError":
            detection = "local_http2_protocol_state"
        elif name == "RemoteProtocolError":
            detection = "remote_protocol_problem_reported_by_client"
        elif name in ("ReadError", "WriteError", "CloseError"):
            detection = "socket_read_write_failure"
        elif name in ("ConnectError", "ConnectTimeout"):
            detection = "connection_establishment_failure"
        elif name in ("ReadTimeout", "WriteTimeout", "PoolTimeout"):
            detection = "client_wait_limit_reached"
        else:
            detection = "transport_error_unspecified"

        # A LocalProtocolError means the *local library* detected an invalid
        # state transition. It does NOT establish who closed the connection.
        marker = str(exc)
        detail = (
            "recv_ping_after_closed"
            if "ConnectionInputs.RECV_PING" in marker and "ConnectionState.CLOSED" in marker
            else "none_identified"
        )
        # Surface concrete socket evidence if it is preserved by the transport.
        # ECONNRESET can indicate a peer/intermediary reset, but still cannot
        # identify whether Google itself or a network intermediary sent it.
        socket_evidence = "none_available"
        nested = exc
        visited = set()
        for _ in range(5):
            if id(nested) in visited:
                break
            visited.add(id(nested))
            number = getattr(nested, "errno", None)
            if number == errno.ECONNRESET:
                socket_evidence = "connection_reset_observed"
                break
            if number == errno.EPIPE:
                socket_evidence = "write_to_closed_socket"
                break
            if number == errno.ETIMEDOUT:
                socket_evidence = "operating_system_timeout"
                break
            nested = getattr(nested, "__cause__", None) or getattr(
                nested, "__context__", None
            )
            if nested is None:
                break

        history = ",".join(self.events) if self.events else "none_from_transport"
        return (
            f"error={name} detected_at={detection} "
            "closure_initiator=UNKNOWN "
            f"socket_evidence={socket_evidence} "
            f"http2_marker={detail} "
            f"tcp_connection={'new_observed' if self.tcp_connected else 'reused_or_unobserved'} "
            f"tls={'complete' if self.tls_connected else 'unobserved'} "
            f"request_body_sent={'yes' if self.request_sent else 'not_confirmed'} "
            f"response_headers={'received' if self.response_headers else 'not_received'} "
            f"recent_transport_events=[{history}]"
        )


def _log_connection_failure(user, trace_id, attempt_label, key_number, exc, timeline):
    """Log safe transport evidence only when existing diagnostics are on."""
    if timeline is not None:
        xlog(
            user,
            f"[Gemini conn {trace_id}] attempt {attempt_label} "
            f"Key {key_number}: {timeline.summary(exc)}",
        )


class WaitCancelled(Exception):
    """The user's authenticated //stopwait aborted this HTTP wait."""


def _cancellable_google_post(
    user: XUID,
    token: str,
    model: str,
    current_key: str,
    gemini_request: dict,
):
    """Cooperatively stop a live Google HTTP call via Gunicorn's gevent worker.

    A private HTTP client is used for cancellable requests, so killing this
    greenlet cannot close the shared connection pool used by normal requests.
    The original request holds its Redis lock until the task is actually
    terminated, preventing a duplicate in-flight generation.
    """
    from gevent import spawn

    # Capture before spawn: the child greenlet may not inherit ContextVars.
    timeline = _gemini_connection_trace.get()

    def do_request():
        with httpx2.Client(http2=True, timeout=None, max_redirects=0) as client:
            return client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": current_key},
                json=gemini_request,
                timeout=None,
                **({"extensions": {"trace": timeline}} if timeline else {}),
            )

    task = spawn(do_request)
    try:
        while True:
            # Poll across Gunicorn workers via Redis. The task runs in a
            # greenlet with gevent-monkey-patched socket I/O on Render.
            task.join(timeout=2.0)
            if wait_should_stop(user, token):
                xlog(user, "//notimeout request cancelled/lost ownership; stopping HTTP call")
                task.kill(block=True)
                raise WaitCancelled()
            if task.ready():
                return task.get()
    finally:
        # Do not release the user's request lock while Google still runs.
        if not task.ready():
            task.kill(block=True)


def gemini_retry_config() -> tuple[int, float]:
    """Return bounded retries and delay; shared by the provider and lock lease."""
    try:
        retries = min(2, max(0, int(os.getenv("GFJPROXY_GEMINI_503_RETRIES", "0"))))
    except ValueError:
        retries = 0
    try:
        delay = float(os.getenv("GFJPROXY_GEMINI_503_RETRY_DELAY", "75"))
        retry_delay = min(300.0, max(0.0, delay)) if math.isfinite(delay) else 75.0
    except (ValueError, OverflowError):
        retry_delay = 75.0
    return retries, retry_delay


def gemini_max_lock_seconds() -> int:
    """Maximum intended duration including attempts, jitter and cleanup margin.

    Key rotation on HTTP 429 happens *inside* an attempt's time window, so the
    number of Google keys does not multiply this lease. This is an upper bound
    for the normal non-streaming HTTP case; HTTP socket timeouts are inactivity
    based, not a hard interruption of an actively streaming connection.
    """
    retries, retry_delay = gemini_retry_config()
    return math.ceil(
        (retries + 1) * PROCESS_TIMEOUT
        + retries * (retry_delay + 3.0)
        + 120.0
    )


def _sleep_for_retry(user: XUID, cause: str, delay_seconds: float) -> None:
    actual = delay_seconds + random.uniform(0, 3)
    xlog(user, f"Gemini {cause}; retrying in {actual:.1f}s")
    time.sleep(actual)


# The no-timeout mode never applies an HTTP read deadline. When Google itself
# reports a quota cooldown, however, do not busy-loop on a limited project.
# Ten minutes is intentionally longer than a one-minute quota window.
_NO_TIMEOUT_MINUTE_COOLDOWN = 600.0
_NO_TIMEOUT_UNKNOWN_COOLDOWN = 600.0
_NO_TIMEOUT_POLL_SECONDS = 2.0


# All time windows are based on Dutch civil time, including DST, rather than
# the hosting server's time zone (often UTC on Render).
try:
    _NO_TIMEOUT_BACKOFF_ZONE = ZoneInfo("Europe/Amsterdam")
except ZoneInfoNotFoundError:
    # Some minimal container images omit the IANA tzdata database. The European
    # DST rule provides the same Amsterdam hours without a new dependency.
    _NO_TIMEOUT_BACKOFF_ZONE = None


def _no_timeout_local_hour() -> int:
    """Current Amsterdam hour; automatically handles winter and summer time."""
    now_utc = datetime.now(UTC)
    if _NO_TIMEOUT_BACKOFF_ZONE is not None:
        return now_utc.astimezone(_NO_TIMEOUT_BACKOFF_ZONE).hour

    # EU daylight saving time: last Sunday of March at 01:00 UTC through the
    # last Sunday of October at 01:00 UTC. Dutch time is UTC+1/UTC+2.
    year = now_utc.year
    march_last_sunday = 31 - ((datetime(year, 3, 31, tzinfo=UTC).weekday() + 1) % 7)
    october_last_sunday = 31 - ((datetime(year, 10, 31, tzinfo=UTC).weekday() + 1) % 7)
    summer_start = datetime(year, 3, march_last_sunday, 1, tzinfo=UTC)
    summer_end = datetime(year, 10, october_last_sunday, 1, tzinfo=UTC)
    offset_hours = 2 if summer_start <= now_utc < summer_end else 1
    return (now_utc + timedelta(hours=offset_hours)).hour


def _no_timeout_503_delay(
    consecutive_503s: int,
    retry_delay: float,
    *,
    after_key_switch: bool = False,
    first_503_after_switch: bool = False,
) -> float:
    """Scheduled //notimeout backoff, based on local Dutch time at each 503.

    09:00-18:00: 75s / 180s / 300s / 600s, by consecutive 503 count.
    18:00-22:00 and 06:00-09:00: same stages, capped at 300s.
    22:00-06:00: always 75s, overriding the post-key-switch floor.

    An HTTP 429 still switches keys immediately. If the next key produces a
    503, its first retry is 300s in daytime/shoulder periods, and subsequent
    failures can return to the preserved 600s stage only in daytime.
    """
    hour = _no_timeout_local_hour()
    if hour >= 22 or hour < 6:
        return 75.0

    if consecutive_503s <= 3:
        delay = retry_delay
    elif consecutive_503s <= 6:
        delay = max(retry_delay, 180.0)
    elif consecutive_503s <= 10:
        delay = max(retry_delay, 300.0)
    else:
        delay = max(retry_delay, 600.0)

    if after_key_switch:
        # Preserve the previous 5-minute key-switch protection, except that
        # nighttime is explicitly faster. The next API key is still tried
        # immediately; this only applies AFTER its own HTTP 503 response.
        if first_503_after_switch:
            delay = max(retry_delay, 300.0)
        else:
            delay = max(delay, 300.0)

    if hour < 9 or hour >= 18:
        delay = min(delay, 300.0)
    return delay


def _no_timeout_check_cancelled(user: XUID, token: str) -> None:
    """Cancel even while between Google calls or waiting for quota reset."""
    if wait_should_stop(user, token):
        raise WaitCancelled()


def _no_timeout_sleep(user: XUID, token: str, cause: str, seconds: float) -> None:
    """A cancellable, low-CPU delay; respects the existing Redis cancel token."""
    delay = max(0.0, seconds)
    xlog(user, f"Gemini //notimeout {cause}; waiting {delay:.1f}s")
    end = time.monotonic() + delay
    while True:
        _no_timeout_check_cancelled(user, token)
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, _NO_TIMEOUT_POLL_SECONDS))


def _no_timeout_quota_info(response) -> tuple[str, float]:
    """Classify Google's 429 without leaking response content or API keys.

    Daily exhaustion is terminal only if all keys report daily exhaustion.
    Unknown 429s are conservatively treated as temporary to avoid giving up
    when Google omits a QuotaFailure identifier.
    """
    ids = []
    retry_hint = 0.0
    try:
        payload = response.json()
        error = payload.get("error", payload) if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            message = error.get("message", "")
            if isinstance(message, str):
                ids.append(message)
            for detail in error.get("details", []):
                if not isinstance(detail, dict):
                    continue
                for violation in detail.get("violations", []):
                    if isinstance(violation, dict):
                        for field in ("quotaId", "description", "subject"):
                            value = violation.get(field)
                            if isinstance(value, str):
                                ids.append(value)
                if str(detail.get("@type", "")).endswith("RetryInfo"):
                    seconds = detail.get("retryDelay", 0)
                    if isinstance(seconds, str) and seconds.endswith("s"):
                        try:
                            retry_hint = max(retry_hint, float(seconds[:-1]))
                        except ValueError:
                            pass
                    elif isinstance(seconds, dict):
                        try:
                            retry_hint = max(
                                retry_hint,
                                float(seconds.get("seconds", 0))
                                + float(seconds.get("nanos", 0)) / 1_000_000_000,
                            )
                        except (ValueError, TypeError, OverflowError):
                            pass
    except (ValueError, TypeError, AttributeError):
        pass

    headers = getattr(response, "headers", {}) or {}
    try:
        retry_after = float(headers.get("retry-after", 0))
        retry_hint = max(retry_hint, retry_after)
    except (TypeError, ValueError, OverflowError, AttributeError):
        pass
    if not math.isfinite(retry_hint) or retry_hint < 0:
        retry_hint = 0.0
    retry_hint = min(retry_hint, 86400.0)
    evidence = " ".join(ids).lower().replace("_", "").replace("-", "")
    # Quota identifiers often use PerDay / PerMinute with no spaces.
    if "perday" in evidence or "per day" in evidence or "daily quota" in evidence:
        return "daily", 0.0
    if "perminute" in evidence or "per minute" in evidence or "permin" in evidence:
        return "minute", max(_NO_TIMEOUT_MINUTE_COOLDOWN, retry_hint)
    return "temporary_unknown", max(_NO_TIMEOUT_UNKNOWN_COOLDOWN, retry_hint)


def _all_daily_keys_error(count: int) -> JaiResult:
    track_stats("g.failed.client.quota")
    return JaiResult(
        429,
        f"All {count} configured Google API key(s) reported a daily quota "
        "limit. //notimeout stopped; daily quotas need to reset before retrying.",
    )


def _google_key(api_key: str) -> str:
    """Accept explicit google/ prefixed keys as well as ordinary Google keys."""
    return api_key.split("/", maxsplit=1)[1] if api_key.lower().startswith("google/") else api_key


def _quota_all_keys_error(count: int) -> JaiResult:
    track_stats("g.failed.client.quota")
    return JaiResult(
        429,
        f"All {count} configured Google API key(s) returned HTTP 429 "
        "(rate limit or quota exhausted) for this request. "
        "No generation retry was consumed for these 429 responses.",
    )


def _diag_enabled() -> bool:
    """Optional logging only; does not affect HTTP requests or retries."""
    return os.getenv("GFJPROXY_GEMINI_DIAGNOSTICS", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _diag_interval() -> float:
    """How often to log while an attempt is still awaiting a response."""
    try:
        value = float(os.getenv("GFJPROXY_GEMINI_DIAG_INTERVAL", "60"))
        if not (10 <= value <= 300):
            return 60.0
        return value
    except (ValueError, OverflowError):
        return 60.0


def _diag_waiting(
    user: XUID,
    trace_id: str,
    attempt_label: str,
    started: float,
    deadline: float | None,
    stopped: threading.Event,
    interval: float,
) -> None:
    """Report a still-pending call; never access request content or credentials."""
    while not stopped.wait(interval):
        elapsed = time.monotonic() - started
        remaining = (
            f"{max(0.0, deadline - time.monotonic()):.1f}s"
            if deadline is not None else "disabled by //notimeout"
        )
        xlog(
            user,
            f"[Gemini diag {trace_id}] attempt {attempt_label} STILL WAITING "
            f"after {elapsed:.1f}s (GFJ budget remaining {remaining})",
        )


def _resolve_link(user: XUID, link: str) -> str:
    result = link

    try:
        if link.startswith(
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
        ):
            response = http_client.get(link)
            if response.status_code != 302:  # Found
                response.raise_for_status()
            if location := response.headers.get("Location"):
                result = location
    except httpx2.HTTPError as e:
        xlog(user, f"Could not resolve link:\n{e!r}")

    if result != link:
        xlog(user, "Link resolved")
    else:
        xlog(user, "Link not resolved")

    return result


def _get_finish_reason_feedback(response: dict[str, Any]) -> str | None:
    """Extracts a human-readable message from a failed GenerateContentResponse dictionary.

    Returns None if no message could be extracted."""

    prompt_feedback = response.get("promptFeedback")
    if isinstance(prompt_feedback, dict):
        if block_reason_message := prompt_feedback.get("blockReasonMessage"):
            return str(block_reason_message)
        if block_reason := prompt_feedback.get("blockReason"):
            return str(block_reason)

    try:
        return response["candidates"][0]["finishReason"]
    except (KeyError, IndexError, TypeError):
        return None


def _get_quota_violation_feedback(qid: str) -> str | None:
    """Converts a quota ID into a human-readable message.

    Returns None if an unknown quota ID is given."""

    if qid.startswith(
        (
            "GenerateContentInputTokensPerModelPerMinute",
            "GenerateContentPaidTierInputTokensPerModelPerMinute",
        )
    ):
        return "Input Tokens per Minute quota exceeded."

    if qid.startswith("GenerateContentInputTokensPerModelPerDay"):
        return "Input Tokens per Day quota exceeded."

    if qid.startswith("GenerateRequestsPerMinutePerProjectPerModel"):
        return "Requests per Minute quota exceeded."

    if qid.startswith("GenerateRequestsPerDayPerProjectPerModel"):
        return "Requests per Day quota exceeded."

    return None


def gemini_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around Google AI's Gemini.

    User paramater is only used for logging. Generation settings must all be passed inside the
    settings parameter."""

    generation_config: dict[str, Any] = {}

    gemini_request: dict[str, Any] = {
        "safetySettings": [
            {
                "threshold": "BLOCK_NONE",
                "category": "HARM_CATEGORY_HATE_SPEECH",
            },
            {
                "threshold": "BLOCK_NONE",
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            },
            {
                "threshold": "BLOCK_NONE",
                "category": "HARM_CATEGORY_HARASSMENT",
            },
            {
                "threshold": "BLOCK_NONE",
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            },
            {
                "threshold": "BLOCK_NONE",
                "category": "HARM_CATEGORY_JAILBREAK",
            },
        ],
        "generationConfig": generation_config,
        "contents": [
            {
                "role": "user" if msg.role == "user" else "model",
                "parts": [{"text": msg.content}],
            }
            for msg in messages
        ],
    }

    for key, value in (settings or {}).items():
        if key == "temperature":
            generation_config["temperature"] = value
        elif key == "max_tokens":
            generation_config["maxOutputTokens"] = value
        elif key == "top_k":
            generation_config["topK"] = value
        elif key == "top_p":
            generation_config["topP"] = value
        elif key == "frequency_penalty":
            generation_config["frequencyPenalty"] = value
        elif key == "repetition_penalty":
            generation_config["presencePenalty"] = value
        elif key == "search" and value:
            gemini_request["tools"] = [{"googleSearch": {}}]

    # Each generation attempt receives its OWN PROCESS_TIMEOUT window.
    # HTTP 429 rotates keys within that same attempt and is not a generation
    # retry. Only 503 responses and socket timeouts advance the attempt count.
    retries, retry_delay = gemini_retry_config()
    no_timeout = gemini_no_timeout.get()
    # Normal mode keeps the existing bounded retry policy unchanged.
    # //notimeout uses an unbounded attempt iterator instead.
    max_attempts = retries + 1
    configured_keys = gemini_key_candidates.get()
    keys = configured_keys or ((1, api_key),)
    keys = tuple((index, _google_key(key)) for index, key in keys)
    exhausted_keys: set[int] = set()  # Normal-mode 429s only.
    daily_exhausted: set[int] = set()  # //notimeout terminal quotas.
    key_cooldowns: dict[int, float] = {}  # //notimeout monotonic expiry.
    key_cursor = 0
    # A 429 is a key-specific quota event, not proof that the model overload ended.
    # Preserve 503 history across a 429 key switch. Once a different key is
    # actually attempted, begin its overload retry delays at five minutes
    # outside the 22:00-06:00 nighttime window.
    consecutive_503s = 0
    previous_key_cursor: int | None = None
    after_key_switch = False
    first_503_after_switch = False
    cancel_token = gemini_wait_token.get() if no_timeout else None
    if no_timeout and not cancel_token:
        # Avoid an un-cancellable infinite operation if the route contract is broken.
        raise RuntimeError("Missing //notimeout cancellation token")
    diagnostics = _diag_enabled()
    trace_id = secrets.token_hex(4) if diagnostics else ""

    if diagnostics:
        # Never log prompts, personal data or API credentials.
        try:
            payload_bytes = len(json.dumps(gemini_request, ensure_ascii=False).encode("utf-8"))
        except (TypeError, ValueError):
            payload_bytes = -1
        xlog(
            user,
            f"[Gemini diag {trace_id}] START model={model} "
            f"messages={len(messages)} payload_bytes={payload_bytes} "
            f"attempts={'unlimited' if no_timeout else max_attempts} "
            f"attempt_budget={'disabled' if no_timeout else str(PROCESS_TIMEOUT) + 's'} "
            f"key_count={len(keys)} "
            f"lock={'renewed' if no_timeout else str(gemini_max_lock_seconds()) + 's'}",
        )

    try:
        finished = False
        for attempt in (count() if no_timeout else range(max_attempts)):
            # Normal mode: every retry starts a fresh process-derived timeout;
            # 429 key switches share it. //notimeout skips the GFJ deadline.
            deadline = None if no_timeout else time.monotonic() + PROCESS_TIMEOUT
            attempt_label = (
                f"{attempt + 1}/unlimited" if no_timeout
                else f"{attempt + 1}/{max_attempts}"
            )

            while True:
                if no_timeout:
                    _no_timeout_check_cancelled(user, cancel_token)
                    if len(daily_exhausted) == len(keys):
                        xlog(user, "Gemini //notimeout: all keys hit daily quotas; stopping")
                        return _all_daily_keys_error(len(keys))
                    # Pick an eligible key, skipping daily-exhausted and cooling keys.
                    # If all remaining keys are cooling, sleep (cancellably)
                    # until the earliest one becomes available again.
                    chosen = None
                    now = time.monotonic()
                    for offset in range(len(keys)):
                        candidate = (key_cursor + offset) % len(keys)
                        if (
                            candidate not in daily_exhausted
                            and key_cooldowns.get(candidate, 0.0) <= now
                        ):
                            chosen = candidate
                            break
                    if chosen is None:
                        next_eligible = min(
                            expiry for candidate, expiry in key_cooldowns.items()
                            if candidate not in daily_exhausted
                        )
                        _no_timeout_sleep(
                            user, cancel_token, "all available keys cooling down",
                            max(0.0, next_eligible - now),
                        )
                        continue
                    key_cursor = chosen
                else:
                    if len(exhausted_keys) == len(keys):
                        xlog(user, f"Gemini all {len(keys)} keys returned 429; stopping")
                        return _quota_all_keys_error(len(keys))
                    # Skip keys that already returned 429 in this request.
                    while key_cursor in exhausted_keys:
                        key_cursor = (key_cursor + 1) % len(keys)

                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    if attempt < retries:
                        _sleep_for_retry(user, "attempt window exhausted", retry_delay)
                        break
                    track_stats("g.time_out")
                    return JaiResult(504, "Gateway Timeout")

                key_number, current_key = keys[key_cursor]
                if no_timeout:
                    # Only an actual key change activates the daytime 5-minute floor.
                    # In //notimeout, keys change after a 429, never after a 503.
                    if previous_key_cursor is not None and key_cursor != previous_key_cursor:
                        after_key_switch = True
                        first_503_after_switch = True
                    previous_key_cursor = key_cursor
                started = time.monotonic()
                connection_trace = _ConnectionTrace() if diagnostics else None
                stopped = None
                if diagnostics:
                    timeout_description = (
                        "disabled" if remaining is None else f"{remaining:.1f}s"
                    )
                    xlog(
                        user,
                        f"[Gemini diag {trace_id}] attempt {attempt_label} "
                        f"Key {key_number}/{len(keys)} START "
                        f"(HTTP timeout {timeout_description})",
                    )
                    stopped = threading.Event()
                    try:
                        threading.Thread(
                            target=_diag_waiting,
                            args=(
                                user, trace_id, attempt_label, started,
                                deadline, stopped, _diag_interval(),
                            ),
                            daemon=True,
                            name="gfj-gemini-diagnostics",
                        ).start()
                    except RuntimeError:
                        stopped.set()
                        xlog(user, f"[Gemini diag {trace_id}] heartbeat unavailable")

                try:
                    if no_timeout:
                        token = gemini_wait_token.get()
                        if not token:
                            raise RuntimeError("Missing //notimeout cancellation token")
                        tracing_token = _gemini_connection_trace.set(connection_trace)
                        try:
                            response = _cancellable_google_post(
                                user, token, model, current_key, gemini_request
                            )
                        finally:
                            _gemini_connection_trace.reset(tracing_token)
                    else:
                        response = http_client.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                            headers={"x-goog-api-key": current_key},
                            json=gemini_request,
                            timeout=remaining,
                            **({"extensions": {"trace": connection_trace}} if diagnostics else {}),
                        )
                except httpx2.TimeoutException as exc:
                    elapsed = time.monotonic() - started
                    if diagnostics:
                        # A configured timeout is expected; keep the existing
                        # concise timeout log, not the full HTTP transport trace.
                        xlog(
                            user,
                            f"[Gemini diag {trace_id}] attempt {attempt_label} "
                            f"Key {key_number}/{len(keys)} TIMEOUT "
                            f"type={type(exc).__name__}; elapsed {elapsed:.1f}s",
                        )
                    if no_timeout:
                        consecutive_503s = 0
                        _no_timeout_sleep(user, cancel_token, "external read timeout", retry_delay)
                        break
                    if attempt < retries:
                        _sleep_for_retry(user, "read timeout", retry_delay)
                        break
                    track_stats("g.time_out")
                    return JaiResult(504, "Gateway Timeout")
                except (httpx2.ProtocolError, httpx2.NetworkError) as exc:
                    # A transport failure is not a Google HTTP error response.
                    # In particular, HTTP/2 can report LocalProtocolError when
                    # a connection is closed while receiving a PING frame.
                    # Normal mode retains its retry budget; //notimeout retries
                    # indefinitely after a cancellable delay.
                    error_name = type(exc).__name__
                    elapsed = time.monotonic() - started
                    if diagnostics:
                        _log_connection_failure(
                            user, trace_id, attempt_label, key_number, exc, connection_trace
                        )
                    xlog(
                        user,
                        f"Gemini network/protocol failure on attempt {attempt_label} "
                        f"Key {key_number}/{len(keys)}: {error_name} "
                        f"after {elapsed:.1f}s",
                    )
                    if no_timeout:
                        consecutive_503s = 0
                        _no_timeout_sleep(
                            user, cancel_token, "connection/protocol failure", retry_delay
                        )
                        break
                    if attempt < retries:
                        _sleep_for_retry(user, "connection/protocol failure", retry_delay)
                        break
                    track_stats("g.failed.network")
                    return JaiResult(
                        502,
                        "Google connection failed (network/HTTP protocol error). "
                        "The request did not return a usable response. "
                        "Please retry manually.",
                    )
                finally:
                    if stopped is not None:
                        stopped.set()

                elapsed = time.monotonic() - started
                xlog(
                    user,
                    f"Gemini attempt {attempt_label} Key {key_number}/{len(keys)}: "
                    f"HTTP {response.status_code} ({elapsed:.1f}s)",
                )
                if diagnostics:
                    remaining_description = (
                        "disabled" if deadline is None
                        else f"{max(0.0, deadline - time.monotonic()):.1f}s"
                    )
                    xlog(
                        user,
                        f"[Gemini diag {trace_id}] attempt {attempt_label} "
                        f"END HTTP {response.status_code}; elapsed {elapsed:.1f}s; "
                        f"attempt remaining {remaining_description}",
                    )

                if no_timeout and response.status_code not in (429, 503):
                    # 429 does not resolve an ongoing model overload; it only
                    # changes which Google project can accept the next call.
                    consecutive_503s = 0

                if response.status_code == 429:
                    if no_timeout:
                        quota_kind, cooldown = _no_timeout_quota_info(response)
                        if quota_kind == "daily":
                            daily_exhausted.add(key_cursor)
                            xlog(
                                user, f"Gemini //notimeout Key {key_number}/{len(keys)} "
                                "daily quota exhausted; skipping for this operation",
                            )
                        else:
                            key_cooldowns[key_cursor] = time.monotonic() + cooldown
                            xlog(
                                user, f"Gemini //notimeout Key {key_number}/{len(keys)} "
                                f"HTTP 429 ({quota_kind}); cooldown {cooldown:.1f}s",
                            )
                        key_cursor = (key_cursor + 1) % len(keys)
                        continue
                    # Normal-mode 429 behavior is unchanged.
                    exhausted_keys.add(key_cursor)
                    if len(exhausted_keys) == len(keys):
                        xlog(user, f"Gemini all {len(keys)} keys returned 429; stopping")
                        return _quota_all_keys_error(len(keys))
                    key_cursor = (key_cursor + 1) % len(keys)
                    xlog(
                        user,
                        f"Gemini Key {key_number}/{len(keys)} rate-limited (429); "
                        f"trying another key within attempt {attempt_label}",
                    )
                    continue

                if no_timeout and response.status_code in (500, 502, 503, 504):
                    if response.status_code == 503:
                        consecutive_503s += 1
                        delay = _no_timeout_503_delay(
                            consecutive_503s,
                            retry_delay,
                            after_key_switch=after_key_switch,
                            first_503_after_switch=first_503_after_switch,
                        )
                        if after_key_switch and first_503_after_switch:
                            first_503_after_switch = False
                        cause = f"HTTP 503 overload (consecutive 503s: {consecutive_503s})"
                    else:
                        delay = retry_delay
                        cause = f"HTTP {response.status_code} overload/server error"
                    _no_timeout_sleep(user, cancel_token, cause, delay)
                    break
                if response.status_code == 503 and attempt < retries:
                    _sleep_for_retry(user, "503 overload", retry_delay)
                    break

                # Other failures are returned through the existing Google
                # error parser. No retry on safety/auth/model errors.
                response.raise_for_status()
                gemini_result = response.json()
                finished = True
                break

            if finished:
                break
        else:
            # Defensive fallback: all attempts were exhausted by 503/timeouts.
            track_stats("g.time_out")
            return JaiResult(504, "Gateway Timeout")

    except WaitCancelled:
        track_stats("g.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from Google AI"
        metadata = JaiResultMetadata()
        extras = ""

        if e.response.is_client_error:
            stats_key = "g.failed.client"
        elif e.response.is_server_error:
            stats_key = "g.failed.server"
        else:
            stats_key = "g.failed.unknown"

        if error := e.response.json():
            if "error" in error:
                error = error["error"]

            error_code = error.get("code", e.response.status_code)
            error_message = error.get("message", "")
            error_status = error.get("status", "")
            error_details = error.get("details", [])

            if error_status == "NOT_FOUND":
                stats_key += ".not_found"
                if error_message.startswith("models/"):
                    stats_key += ".model"
                    error_message = f"Invalid/unsupported model '{model}'"
                elif "no longer available" in error_message:
                    stats_key += ".unavailable"
                    error_message = f"Model '{model}' is no longer available"
            elif error_status == "UNAUTHENTICATED":
                stats_key += ".unauthenticated"
                if "invalid authentication credentials" in error_message:
                    stats_key += ".api_key"
                    error_message = (
                        "Request had invalid authentication credentials.\n"
                        "Your API key is probably not valid."
                    )
                    metadata.api_key_valid = False
            elif error_status == "INVALID_ARGUMENT":
                stats_key += ".invalid"
                if "API key not valid" in error_message:
                    stats_key += ".api_key"
                    metadata.api_key_valid = False
                elif "ending with a model turn" in error_message:
                    stats_key += ".turns"
                    extras = "Disable your prefills or use `//fixturns on`."
            elif error_status == "PERMISSION_DENIED":
                stats_key += ".denied"
                for detail in error_details:
                    if (
                        not isinstance(detail, dict)
                        or detail.get("@type")
                        != "type.googleapis.com/google.rpc.ErrorInfo"
                        or not (reason := detail.get("reason"))
                    ):
                        continue
                    if reason == "SERVICE_DISABLED":
                        stats_key += ".disabled"
                        error_message = "Generative Language API needs to be enabled"
                        break
                    if reason == "CONSUMER_SUSPENDED":
                        stats_key += ".suspended"
                        error_message = "Customer suspended. You might be banned."
                        break
                else:
                    stats_key += ".unknown"
            elif error_status == "RESOURCE_EXHAUSTED":
                stats_key += ".quota"
                for detail in error_details:
                    if (
                        not isinstance(detail, dict)
                        or detail.get("@type")
                        != "type.googleapis.com/google.rpc.QuotaFailure"
                    ):
                        continue
                    quota_ids = (
                        str(v.get("quotaId", ""))
                        for v in detail.get("violations", [])
                        if isinstance(v, dict)
                    )
                    for qid in quota_ids:
                        if feedback := _get_quota_violation_feedback(qid):
                            stats_key += f".{qid}"
                            error_message = feedback
                            break
                    else:
                        stats_key += ".unknown"
                        break

            message += f" ({error_code}): {error_status}\n{error_message}"
        elif error_text := e.response.text:
            xlog(user, f"{message}: {e.response.text!r}")

            message += (
                f" ({e.response.status_code}):\n{error_text[:100]}"
                f"{'...' if len(error_text) > 100 else ''}"
            )

        track_stats(stats_key)
        return JaiResult(
            e.response.status_code, message, extras=extras, metadata=metadata
        )
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))  # These are R E A L L Y anomalous
        track_stats("g.failed.unknown")
        return JaiResult(502, "Unhanded exception from Google AI.")

    text = ""
    extras = ""
    metadata = JaiResultMetadata()

    if (candidates := gemini_result.get("candidates")) and isinstance(candidates, list):
        if len(candidates) > 1:
            xlog(user, "Warning: more than one candidate found in response")
        candidate = candidates[0]
        if isinstance(candidate, dict):
            if (
                (content := candidate.get("content"))
                and isinstance(content, dict)
                and (parts := content.get("parts"))
                and isinstance(parts, list)
            ):
                text = ""
                for part in parts:
                    if isinstance(part, dict):
                        part_text = part.get("text")
                        part_thought = part.get("thought", False)
                        if isinstance(part_text, str) and not part_thought:
                            text += part_text

            if gm := candidate.get("groundingMetadata"):
                # U+3164 HANGUL FILLER

                if isinstance((wsqs := gm.get("webSearchQueries")), list):
                    xlog(user, f"Made {len(wsqs)} web searches")
                    extras += (
                        "Searches:\n"
                        + "\n".join(f"\u3164- {wsq}" for wsq in wsqs)
                        + "\n"
                    )

                if isinstance((gcs := gm.get("groundingChunks")), list):
                    links: list[str] = []
                    for gc in gcs:
                        if (
                            isinstance(gc, dict)
                            and (web := gc.get("web"))
                            and isinstance(web, dict)
                            and (uri := web.get("uri"))
                        ):
                            links.append(_resolve_link(user, uri))
                    xlog(user, f"Found {len(gcs)} grounding chunks {len(links)} links")
                    extras += (
                        "Links:\n"
                        + "\n".join(f"\u3164- {link}" for link in links)
                        + "\n"
                    )

    if not text:
        # Rejection

        feedback = _get_finish_reason_feedback(gemini_result)
        if not feedback:
            xlog(user, f"No result text: {gemini_result}")
            feedback = "UNKNOWN"

        track_stats(f"g.rejected.{feedback}")
        return JaiResult(
            502,
            f"Response blocked/empty due to {feedback}.",
            metadata=JaiResultMetadata(
                rejection_feedback=feedback,
            ),
        )

    if isinstance((usage := gemini_result.get("usageMetadata")), dict):
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
            reasoning_tokens=usage.get("thoughtsTokenCount"),
            total_tokens=usage.get("totalTokenCount"),
        )

    track_stats("g.succeeded")
    return JaiResult(200, text, extras=extras, metadata=metadata)
