from traceback import print_exception

from flask import Blueprint, abort, request
from redis.exceptions import RedisError

from .._globals import PROCESS_TIMEOUT
from ..cooldown import get_cooldown
from ..handlers import PROVIDER_FUNCS, handle_chat_message, handle_proxy_test
from ..logging import xlog, xlogtime
from ..models import JaiRequest
from ..providers.lorebary import LOREBARY_ENDPOINTS, lorebary_generate_content
from ..response_recovery import (
    recovery_request,
    remember_response,
    restore_response,
    strip_saved_recovery_markers,
)
from ..storage import storage
from ..utils import MessageKind, ResponseHelper, comma_split, is_proxy_test
from ..xuid_secret import xuid_secret
from ..xuiduser import XUID, RedisUserStorage, UserSettings

# Install LoreBary as a separate provider in the existing dispatch table.
# No changes to gemini.py or to the original handlers.py are required.
PROVIDER_FUNCS["lorebary"] = lorebary_generate_content

proxy = Blueprint("proxy", __name__)


@proxy.route("/", methods=["POST"])
@proxy.route("/chat/completions", methods=["POST"])
@proxy.route("/quiet/", methods=["POST"])
@proxy.route("/quiet/chat/completions", methods=["POST"])
@proxy.route("/lorebary/<route_name>/chat/completions", methods=["POST"])
@proxy.route("/quiet/lorebary/<route_name>/chat/completions", methods=["POST"])
def handle(route_name: str | None = None):
    assert storage is not None  # Make type checkers happy

    request_json = request.get_json(silent=True)
    if not request_json:  # This should never happen.
        abort(400, "Bad Request. Missing or invalid JSON.")
        return  # Some type checkers don't realize abort exits the function

    request_path = request.path

    jai_req = JaiRequest.parse(request_json)
    jai_req.quiet = "/quiet/" in request_path

    if route_name is not None:
        # Optional dedicated GFJ proxy URL, e.g.
        #   /quiet/lorebary/aistudio/chat/completions
        # The JanitorAI model may remain "gemini-3.8-flash" on this URL.
        if route_name not in LOREBARY_ENDPOINTS:
            return ResponseHelper().build_error("Unknown LoreBary route.", 400)
        raw_model = request_json.get("model")
        if not isinstance(raw_model, str) or not raw_model.strip():
            return ResponseHelper().build_error("Missing LoreBary model.", 400)
        selected_model = raw_model.strip()
        if selected_model.lower().startswith("lorebary/"):
            selected_model = selected_model.split("/", maxsplit=1)[1]
            if selected_model.split("/", maxsplit=1)[0].lower() != route_name:
                return ResponseHelper().build_error(
                    "The LoreBary URL and model route do not match.", 400
                )
        else:
            # Strip a redundant provider prefix for non-OpenRouter routes.
            # OpenRouter needs its vendor/model syntax preserved.
            prefix = {
                "aistudio": "google/",
                "anthropic": "anthropic/",
                "openai": "openai/",
                "deepseek": "deepseek/",
                "mistral": "mistral/",
                "chutes": "chutes/",
                "lorellm": "lorellm/",
            }.get(route_name)
            if prefix and selected_model.lower().startswith(prefix):
                selected_model = selected_model[len(prefix):]
            selected_model = f"{route_name}/{selected_model}"
        jai_req.models = {"lorebary": selected_model}

    proxy_test = is_proxy_test(request_json)

    response = ResponseHelper(use_stream=jai_req.stream, wrap_errors=proxy_test)

    # JanitorAI provides the user's API key through HTTP Bearer authentication.
    # Google AI cannot be used without an API key and neither can this proxy.

    request_auth = request.headers.get("authorization", "").split(" ", maxsplit=1)
    if len(request_auth) != 2 or request_auth[0].lower() != "bearer":
        return response.build_error("Unauthorized. API key required.", 401)

    api_keys = comma_split(request_auth[1])
    xuid = XUID(api_keys[0], xuid_secret)

    # The URL's selected route is part of the recovery fingerprint. Otherwise
    # identical request bodies sent to different LoreBary URLs could collide.
    cache_request = {**request_json, "_gfj_route_path": request_path}

    # Explicit recovery bypasses cooldown, key rotation, and upstream entirely.
    # Ordinary identical requests (including Regenerate) always reach Google.
    if not proxy_test:
        is_recovery, original_request = recovery_request(cache_request)
        if is_recovery:
            cached = restore_response(xuid, original_request)
            if cached is None:
                xlog(xuid, "Response recovery MISS (no matching saved response)")
                return response.build_error(
                    "No matching saved reply. Recovery requires exactly the "
                    "original conversation, provider route, and settings. "
                    "Remove //recover to request a new reply.",
                    404,
                )
            xlog(xuid, "Response recovery HIT (upstream was not contacted)")
            return response.add_message(cached).build()

    # An earlier recovered message may be part of later JanitorAI history.
    # Keep its control marker out of future prompts while preserving the
    # original request body for strict cache matching.
    strip_saved_recovery_markers(jai_req.messages)

    if not storage.lock(xuid):
        xlog(xuid, "User attempted concurrent use")
        return response.build_error(
            "Concurrent use is not allowed. Please wait a moment.", 403
        )

    # Redis's upstream per-user lock expires after 60 seconds, even if the
    # Google request is still running. Extend this *owned* lock to cover the
    # configured process timeout plus a grace period. With a 750s Gunicorn
    # timeout the effective 740s budget yields an 860s lock. This prevents a concurrent
    # JanitorAI retry from starting another costly Google request.
    # Normal processing still releases the lock immediately at completion.
    if isinstance(storage, RedisUserStorage):
        try:
            lock_lease_seconds = PROCESS_TIMEOUT + 120
            storage._locks[xuid.lockid()].extend(
                lock_lease_seconds, replace_ttl=True
            )
        except (KeyError, RedisError) as exc:
            xlog(xuid, f"Could not extend Redis request lock: {type(exc).__name__}")
            storage.unlock(xuid)
            return response.build_error(
                "Could not secure request lock. Please retry later.", 503
            )

    user = UserSettings(storage, xuid)

    # Cheap and easy rate limiting

    if (
        (seconds := user.last_seen())
        and (cooldown := get_cooldown())
        and (delay := cooldown - seconds) > 0
    ):
        xlog(user, f"User told to wait {delay} seconds")
        storage.unlock(xuid)
        return response.build_error(f"Please wait {delay} seconds.", 429)

    # Handle user's request

    rcounter = user.get_rcounter()
    api_key_index = rcounter % len(api_keys)
    user.inc_rcounter()

    # Do not require users to rewrite their comma-separated API key lists.
    # The internal prefix selects the dedicated LoreBary provider; it is not
    # forwarded to LoreBary itself, which receives the ONE selected key.
    selected_key = api_keys[api_key_index]
    if "lorebary" in jai_req.models:
        jai_req.api_key = f"lorebary/{selected_key}"
    else:
        jai_req.api_key = selected_key
    jai_req.api_key_index = api_key_index
    jai_req.api_key_count = len(api_keys)

    log_details = [
        f"User {user.last_seen_msg()}",
        f"Request #{user.get_rcounter()}",
    ]

    if len(api_keys) > 1:
        log_details.append(f"Key {api_key_index + 1}/{len(api_keys)}")

    ref_time = xlogtime(
        user,
        f"Processing {'stream ' if jai_req.stream else ''}{request_path} ({', '.join(log_details)})",
    )

    try:
        if not jai_req.models:
            response.add_error("Please specify a model.", 400)
        if unknown := jai_req.models.get("unknown"):
            response.add_error(
                (
                    f"Unknown model(s): {unknown}\n"
                    + "Make sure to use OpenRouter syntax `provider/model`.\n"
                    + "Examples: `google/gemini-2.5-flash`, `cerebras/llama3.1-8b`, `deepseek/deepseek-chat`, etc."
                ),
                400,
            )
        elif proxy_test:
            response = handle_proxy_test(user, jai_req, response)
        else:
            response = handle_chat_message(user, jai_req, response)
    except Exception as e:  # ruff: ignore[BLE001]
        response.add_error("Internal Proxy Error", 500)
        print_exception(e)

    if 200 <= response.status <= 299:
        xlogtime(user, "Processing succeeded", ref_time)

        if not proxy_test and (announcement := storage.announcement):
            response.add_proxy_message(f"***\n{announcement}\n***")
    else:
        messages = response.message.split("\n")
        xlogtime(user, f"Processing failed: {messages[0]}", ref_time)
        for message in messages[1:]:
            xlog(user, f"> {message}")

    # Save only a completed, successful assistant reply, not error responses,
    # proxy-only command output, or API-key/model test responses. Caching is
    # best effort: its failure must never prevent the real response returning.
    if (
        not proxy_test
        and response.status == 200
        and any(item.kind is MessageKind.CHAT for item in response._messages)
    ):
        if remember_response(xuid, cache_request, response.message):
            xlog(user, "Response recovery: saved successful reply in Redis")

    if user.valid:
        user.save()
    else:
        xlog(user, "Invalid user not saved")

    storage.unlock(xuid)

    return response.build()
