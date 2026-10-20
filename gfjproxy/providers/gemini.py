import os
import random
import time
from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID

################################################################################


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
