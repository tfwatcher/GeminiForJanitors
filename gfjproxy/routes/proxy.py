import threading
from traceback import print_exception

from flask import Blueprint, abort, request
from redis.exceptions import RedisError

from .._globals import PROCESS_TIMEOUT
from ..cooldown import get_cooldown
from ..handlers import PROVIDER_FUNCS, handle_chat_message, handle_proxy_test
from ..logging import xlog, xlogtime
from ..models import JaiRequest
from ..providers.gemini import (
    gemini_key_candidates,
    gemini_max_lock_seconds,
    gemini_no_timeout,
    gemini_wait_token,
)
from ..providers.lorebary import LOREBARY_ENDPOINTS, lorebary_generate_content
from ..response_recovery import (
    extract_control_requests_with_stop,
    remember_response,
    restore_response,
    strip_saved_recovery_markers,
)
from ..storage import storage
from ..utils import MessageKind, ResponseHelper, comma_split, is_proxy_test
from ..wait_control import (
    WAIT_LEASE_SECONDS,
    WaitControlUnavailable,
    begin_wait,
    end_wait,
    stop_wait,
)
from ..xuid_secret import xuid_secret
from ..xuiduser import XUID, RedisUserStorage, UserSettings

# Install LoreBary as a separate provider in the existing dispatch table.
# No changes to gemini.py or to the original handlers.py are required.
PROVIDER_FUNCS["lorebary"] = lorebary_generate_content

proxy = Blueprint("proxy", __name__)

# Redis request locks are renewed ONLY for explicit //notimeout requests.
# If the worker dies, the lock expires shortly after the last renewal.
_NO_TIMEOUT_LOCK_LEASE = WAIT_LEASE_SECONDS  # seconds
_NO_TIMEOUT_LOCK_RENEW = 60  # seconds
_LOCK_RENEW_LUA = """
-- Never extend a lock that belongs to somebody else. Verify that this
-- no-timeout request still owns BOTH the lock and the active cancellation
-- marker before renewing the lease for either one.
if redis.call('get', KEYS[1]) == ARGV[1]
   and redis.call('get', KEYS[2]) == ARGV[2] then
    redis.call('pexpire', KEYS[1], ARGV[3])
    redis.call('expire', KEYS[2], ARGV[4])
    return 1
end
return 0
"""


def _renew_no_timeout_lock(
    redis_storage: RedisUserStorage,
    xuid: XUID,
    owned_token: bytes,
    wait_token: str,
    stopped: threading.Event,
) -> None:
    """Renew only a lock still owned by this request; no unsafe TTL resets.

    The redis-py lock token is normally thread-local, so a background thread
    cannot call Lock.extend() directly. Lua checks the captured owner token
    atomically before extending, preventing extension of another user's lock.
    """
    while not stopped.wait(_NO_TIMEOUT_LOCK_RENEW):
        try:
            result = redis_storage._client.eval(
                _LOCK_RENEW_LUA, 2, xuid.lockid(),
                f":notimeout:active:{xuid!r}",
                owned_token, wait_token,
                _NO_TIMEOUT_LOCK_LEASE * 1000, WAIT_LEASE_SECONDS,
            )
            if not result:
                xlog(xuid, "//notimeout Redis lock no longer owned; renewal stopped")
                return
        except RedisError as exc:
            xlog(xuid, f"//notimeout Redis lock renewal failed: {type(exc).__name__}")
            return


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

    # Extract special controls from the RAW latest user message before GFJ's
    # regular command parser modifies content. Older messages cannot activate
    # either command. Ordinary prompts never change the timeout behavior.
    proxy_test = is_proxy_test(request_json)
    if proxy_test:
        wants_recovery, wants_no_timeout, wants_stop_wait, cleaned_json = (
            False, False, False, request_json
        )
    else:
        wants_recovery, wants_no_timeout, wants_stop_wait, cleaned_json = (
            extract_control_requests_with_stop(request_json)
        )
    jai_req = JaiRequest.parse(cleaned_json)
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

    response = ResponseHelper(use_stream=jai_req.stream, wrap_errors=proxy_test)

    # JanitorAI provides the user's API key through HTTP Bearer authentication.
    # Google AI cannot be used without an API key and neither can this proxy.

    request_auth = request.headers.get("authorization", "").split(" ", maxsplit=1)
    if len(request_auth) != 2 or request_auth[0].lower() != "bearer":
        return response.build_error("Unauthorized. API key required.", 401)

    api_keys = comma_split(request_auth[1])
    xuid = XUID(api_keys[0], xuid_secret)

    # //stopwait bypasses the existing user's request lock, so a second
    # JanitorAI request can stop a FIRST request that is still awaiting Google.
    # This route doesn't call Google and changes no future timeout settings.
    if wants_stop_wait:
        try:
            stopped = stop_wait(xuid)
        except WaitControlUnavailable:
            return response.build_error(
                "Cannot cancel the pending wait: Redis is unavailable.", 503
            )
        if stopped:
            xlog(xuid, "//stopwait cancellation requested")
            return response.add_proxy_message(
                "Cancellation requested for the active //notimeout request. "
                "Its original request will return Gateway Timeout (504) if "
                "JanitorAI is still connected."
            ).build()
        return response.add_proxy_message(
            "No active //notimeout request was found; nothing to stop."
        ).build()

    # Recovery uses the latest successful reply for this proxy user, regardless
    # of changes to the original conversation, model, or provider route.
    cache_request = request_json

    # Explicit recovery bypasses cooldown, key rotation, and upstream entirely.
    # Ordinary identical requests (including Regenerate) always reach Google.
    if wants_recovery:
        cached = restore_response(xuid, cleaned_json)
        if cached is None:
            xlog(xuid, "Response recovery MISS (no saved reply available)")
            return response.build_error(
                "No saved reply available for this proxy user. "
                "It may have been evicted or saved under a different first API key. "
                "Remove //recover to generate a new reply.",
                404,
            )
        xlog(xuid, "Response recovery HIT (upstream was not contacted)")
        return response.add_message(cached).build()

    # An earlier recovered message may be part of later JanitorAI history.
    # Keep its control marker out of future model prompts.
    strip_saved_recovery_markers(jai_req.messages)

    # The override is deliberately limited to the direct Google Gemini route.
    # It never changes LoreBary, provider tests, or other models.
    no_timeout_mode = (
        wants_no_timeout and not proxy_test and "google" in jai_req.models
        and "lorebary" not in jai_req.models
    )
    if wants_no_timeout and not no_timeout_mode:
        xlog(xuid, "//notimeout ignored: selected provider is not direct Google")

    if no_timeout_mode and not isinstance(storage, RedisUserStorage):
        return response.build_error(
            "//notimeout with //stopwait requires Redis storage.", 503
        )

    if not storage.lock(xuid):
        xlog(xuid, "User attempted concurrent use")
        return response.build_error(
            "Concurrent use is not allowed. Please wait a moment.", 403
        )

    # Extend this owned Redis lock to cover every possible Gemini attempt,
    # including retry delays and a cleanup margin. 429 key switches share an
    # attempt's timeout, so the number of keys does not multiply the lock.
    # Other providers retain the original PROCESS_TIMEOUT + 120s lease.
    # Finishing early still releases the lock immediately.
    if isinstance(storage, RedisUserStorage):
        try:
            lock_lease_seconds = (
                _NO_TIMEOUT_LOCK_LEASE
                if no_timeout_mode
                else gemini_max_lock_seconds()
                if not proxy_test and "google" in jai_req.models
                else PROCESS_TIMEOUT + 120
            )
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

    # Register a unique cross-worker cancellation target while holding
    # the per-user lock. Never overwrite another active waiting request.
    wait_token = None
    if no_timeout_mode:
        try:
            wait_token = begin_wait(xuid)
        except WaitControlUnavailable:
            storage.unlock(xuid)
            return response.build_error(
                "Cannot register cancellable //notimeout request in Redis.", 503
            )
        if wait_token is None:
            storage.unlock(xuid)
            return response.build_error(
                "An earlier //notimeout wait is still registered. "
                "Send //stopwait or allow its lease to expire.", 409
            )

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

    # In no-timeout mode, renew the Redis lease while waiting. The renewal
    # stops when processing ends, and a dead worker cannot extend it.
    lease_stop = None
    if no_timeout_mode and isinstance(storage, RedisUserStorage):
        try:
            owned_token = storage._locks[xuid.lockid()].local.token
            if not isinstance(owned_token, bytes):
                raise TypeError("Cannot get Redis lock owner token")
            lease_stop = threading.Event()
            threading.Thread(
                target=_renew_no_timeout_lock,
                args=(storage, xuid, owned_token, wait_token, lease_stop),
                daemon=True,
                name="gfj-notimeout-redis-lock",
            ).start()
        except (KeyError, TypeError, RuntimeError) as exc:
            if lease_stop is not None:
                lease_stop.set()
            if wait_token is not None:
                end_wait(xuid, wait_token)
            storage.unlock(xuid)
            xlog(xuid, f"//notimeout cannot secure lock renewal: {type(exc).__name__}")
            return response.build_error(
                "Unable to establish the //notimeout request lock renewal.", 503
            )

    # Only the direct-Google provider gets a per-request ordered key list.
    # A ContextVar keeps concurrent users' API keys isolated from each other.
    key_token = None
    if not proxy_test and "google" in jai_req.models and "lorebary" not in jai_req.models:
        order = list(range(api_key_index, len(api_keys))) + list(range(api_key_index))
        candidates = tuple((i + 1, api_keys[i]) for i in order)
        key_token = gemini_key_candidates.set(candidates)

    no_timeout_token = gemini_no_timeout.set(True) if no_timeout_mode else None
    wait_token_context = gemini_wait_token.set(wait_token) if wait_token else None
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
    finally:
        if key_token is not None:
            gemini_key_candidates.reset(key_token)
        if no_timeout_token is not None:
            gemini_no_timeout.reset(no_timeout_token)
        if wait_token_context is not None:
            gemini_wait_token.reset(wait_token_context)
        if lease_stop is not None:
            lease_stop.set()

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
        and remember_response(xuid, cache_request, response.message)
    ):
        xlog(user, "Response recovery: saved successful reply in Redis")

    if user.valid:
        user.save()
    else:
        xlog(user, "Invalid user not saved")

    # Clear only this request's cancellation marker; normal requests are
    # unaffected, and a stale //stopwait cannot cancel later attempts.
    if wait_token is not None:
        end_wait(xuid, wait_token)
    storage.unlock(xuid)

    return response.build()
