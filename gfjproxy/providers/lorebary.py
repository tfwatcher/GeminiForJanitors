"""LoreBary's permanent provider routes as a separate GFJ provider.

JanitorAI still talks to GFJ. GFJ selects exactly one key from its multikey list,
then relays that key and the request to the selected LoreBary endpoint.
Endpoint URLs below are allowlisted and come from LoreBary's public Academy.
No LoreBary behavior is inserted into the native Gemini provider.
"""

import json
import os
import random
import secrets
import threading
import time
from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID

LOREBARY_ENDPOINTS = {
    "aistudio": "https://api.lorebary.com/aistudio",
    "openrouter": "https://api.lorebary.com/openrouter",
    "anthropic": "https://api.lorebary.com/anthropic",
    "openai": "https://api.lorebary.com/openai",
    "deepseek": "https://api.lorebary.com/deepseek",
    "mistral": "https://api.lorebary.com/mistral",
    "chutes": "https://api.lorebary.com/chutes",
    "lorellm": "https://api.lorebary.com/lorellm",
}

# Optional, explicit provider prefix on individual keys; not needed for
# ordinary comma-separated Google keys such as AQ.* and AIza*.
_ROUTE_KEY_PREFIX = {
    "aistudio": "google",
    "openrouter": "openrouter",
    "anthropic": "anthropic",
    "openai": "openai",
    "deepseek": "deepseek",
    "mistral": "mistral",
    "chutes": "chutes",
    "lorellm": "lorellm",
}


def _diagnostics_enabled() -> bool:
    return os.getenv("GFJPROXY_GEMINI_DIAGNOSTICS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _diagnostics_interval() -> float:
    try:
        value = float(os.getenv("GFJPROXY_GEMINI_DIAG_INTERVAL", "60"))
        return value if 10 <= value <= 300 else 60.0
    except (ValueError, OverflowError):
        return 60.0


def _diagnostic_heartbeat(
    user: XUID,
    trace_id: str,
    attempt_label: str,
    started: float,
    deadline: float,
    stopped: threading.Event,
) -> None:
    while not stopped.wait(_diagnostics_interval()):
        elapsed = time.monotonic() - started
        remaining = max(0.0, deadline - time.monotonic())
        xlog(
            user,
            f"[LoreBary diag {trace_id}] attempt {attempt_label} STILL WAITING "
            f"after {elapsed:.1f}s (budget remaining {remaining:.1f}s)",
        )


def _route_and_model(target: str) -> tuple[str, str] | None:
    route, sep, model = target.partition("/")
    route = route.lower()
    if not sep or route not in LOREBARY_ENDPOINTS or not model.strip():
        return None
    if route == "lorellm" and model.lower() == "lorellm":
        model = "LoreLLM"
    return route, model


def _provider_api_key(route: str, key: str) -> str | None:
    # The dispatcher adds 'lorebary/' internally. A second prefix may be
    # present if the user already configured e.g. 'google/AQ.abc'.
    if "/" in key:
        prefix, rest = key.split("/", 1)
        if prefix.lower() != _ROUTE_KEY_PREFIX[route]:
            return None
        key = rest
    return key if key else None


def _error_text(body: Any) -> str:
    if isinstance(body, dict):
        body = body.get("error", body)
        if isinstance(body, dict):
            return str(body.get("message") or body.get("status") or "upstream error")
    if isinstance(body, str):
        return body[:300]
    return "upstream error"


def _part_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = []
        for part in value:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "".join(texts)
    return ""


def _parse_sse(content: str) -> str:
    chunks = []
    for line in content.splitlines():
        if not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if value == "[DONE]":
            break
        try:
            event = json.loads(value)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices", []):
            if isinstance(choice, dict):
                chunks.append(_part_text(choice.get("delta", {}).get("content")))
    return "".join(chunks)


def _extract_response(body: Any) -> tuple[str, JaiResultMetadata]:
    metadata = JaiResultMetadata()
    if not isinstance(body, dict):
        return "", metadata
    if isinstance(body.get("error"), dict):
        return "", metadata
    if isinstance((usage := body.get("usage")), dict):
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ) if isinstance(usage.get("completion_tokens_details"), dict) else None,
            total_tokens=usage.get("total_tokens"),
        )
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                return _part_text(message.get("content")), metadata
            delta = first.get("delta")
            if isinstance(delta, dict):
                return _part_text(delta.get("content")), metadata
            return _part_text(first.get("text")), metadata
    # Defensive support for Google- or Anthropic-shaped replies.
    candidates = body.get("candidates")
    if isinstance(candidates, list) and candidates:
        candidate = candidates[0]
        if isinstance(candidate, dict):
            content = candidate.get("content")
            if isinstance(content, dict):
                return _part_text(content.get("parts")), metadata
    return _part_text(body.get("content")), metadata


def _retry_config() -> tuple[int, float]:
    try:
        retries = min(2, max(0, int(os.getenv("GFJPROXY_GEMINI_503_RETRIES", "0"))))
    except ValueError:
        retries = 0
    try:
        delay = min(300.0, max(0.0, float(os.getenv("GFJPROXY_GEMINI_503_RETRY_DELAY", "75"))))
    except (ValueError, OverflowError):
        delay = 75.0
    return retries, delay


def lorebary_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Call a selected permanent LoreBary endpoint with one GFJ-selected key."""
    parsed = _route_and_model(model)
    if parsed is None:
        return JaiResult(
            400,
            "LoreBary model must be lorebary/<route>/<model>. "
            f"Available routes: {', '.join(LOREBARY_ENDPOINTS)}",
        )
    route, model_id = parsed
    provider_key = _provider_api_key(route, api_key)
    if provider_key is None:
        return JaiResult(
            400,
            "API key prefix does not match the LoreBary route. "
            "Choose a key from that provider; Google keys work with aistudio.",
        )

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": False,
    }
    for source, dest in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("frequency_penalty", "frequency_penalty"),
        ("repetition_penalty", "presence_penalty"),
    ):
        if settings is not None and source in settings:
            payload[dest] = settings[source]

    if settings and settings.get("search"):
        xlog(user, "LoreBary: native GFJ Google Search flag is not forwarded")

    retries, retry_delay = _retry_config()
    deadline = time.monotonic() + PROCESS_TIMEOUT
    trace_id = secrets.token_hex(4) if _diagnostics_enabled() else ""
    attempt_label = "not started"
    started = time.monotonic()

    if trace_id:
        xlog(
            user,
            f"[LoreBary diag {trace_id}] START route={route} model={model_id} "
            f"messages={len(messages)} attempts={retries + 1} "
            f"total_budget={PROCESS_TIMEOUT}s",
        )

    try:
        for attempt in range(retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 5:
                track_stats("g.time_out")
                return JaiResult(504, "LoreBary gateway timeout (request budget exhausted).")

            started = time.monotonic()
            attempt_label = f"{attempt + 1}/{retries + 1}"
            stopped = threading.Event() if trace_id else None
            if trace_id:
                xlog(
                    user,
                    f"[LoreBary diag {trace_id}] attempt {attempt_label} START "
                    f"(HTTP timeout {remaining:.1f}s)",
                )
                assert stopped is not None
                try:
                    threading.Thread(
                        target=_diagnostic_heartbeat,
                        args=(user, trace_id, attempt_label, started, deadline, stopped),
                        daemon=True,
                        name="gfj-lorebary-diagnostics",
                    ).start()
                except RuntimeError:
                    stopped.set()
                    xlog(user, f"[LoreBary diag {trace_id}] heartbeat unavailable")

            try:
                response = http_client.post(
                    LOREBARY_ENDPOINTS[route],
                    headers={"Authorization": f"Bearer {provider_key}"},
                    json=payload,
                    timeout=remaining,
                )
            finally:
                if stopped is not None:
                    stopped.set()

            elapsed = time.monotonic() - started
            xlog(
                user,
                f"LoreBary {route} attempt {attempt_label}: "
                f"HTTP {response.status_code} ({elapsed:.1f}s)",
            )
            if trace_id:
                xlog(
                    user,
                    f"[LoreBary diag {trace_id}] attempt {attempt_label} END "
                    f"HTTP {response.status_code}; elapsed {elapsed:.1f}s; "
                    f"budget remaining {max(0.0, deadline - time.monotonic()):.1f}s",
                )

            if response.status_code == 503 and attempt < retries:
                delay = retry_delay + random.uniform(0, 3)
                if deadline - time.monotonic() > delay + 30:
                    xlog(user, f"LoreBary 503; retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue

            if response.status_code >= 400:
                try:
                    error = _error_text(response.json())
                except ValueError:
                    error = _error_text(response.text)
                # Don't disclose selected credentials in error messages/logs.
                error = error.replace(provider_key, "[redacted]")[:350]
                track_stats("g.failed.server" if response.status_code >= 500 else "g.failed.client")
                return JaiResult(
                    response.status_code,
                    f"Error from LoreBary ({response.status_code}): {error}",
                )

            try:
                body = response.json()
            except ValueError:
                body = None
            output, metadata = _extract_response(body)
            if not output and response.text.lstrip().startswith("data:"):
                output = _parse_sse(response.text)
            if not output:
                track_stats("g.failed.unknown")
                return JaiResult(502, "LoreBary returned an empty or unsupported response.")
            track_stats("g.succeeded")
            return JaiResult(200, output, metadata=metadata)
    except httpx2.TimeoutException as exc:
        if trace_id:
            xlog(
                user,
                f"[LoreBary diag {trace_id}] attempt {attempt_label} "
                f"TIMEOUT type={type(exc).__name__}; "
                f"elapsed {time.monotonic() - started:.1f}s",
            )
        track_stats("g.time_out")
        return JaiResult(504, "LoreBary gateway timeout.")
    except httpx2.HTTPError as exc:
        xlog(user, f"LoreBary request transport error: {type(exc).__name__}")
        track_stats("g.failed.unknown")
        return JaiResult(502, "LoreBary connection error.")
    except Exception as exc:  # ruff: ignore[BLE001]
        xlog(user, f"LoreBary unexpected error: {type(exc).__name__}")
        track_stats("g.failed.unknown")
        return JaiResult(502, "Unexpected LoreBary proxy error.")

    return JaiResult(502, "LoreBary retry loop ended without a response.")
