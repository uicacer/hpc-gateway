"""
STREAM Proxy — Authentication & Authorization

This module handles two authentication modes that can coexist on the same proxy:

MODE A: Globus Token Auth (preferred for university-wide deployment)
----------------------------------------------------------------------
The caller presents a Globus access token in the Authorization header.
The proxy validates it against Globus Auth's introspect endpoint and extracts
the caller's identity (e.g., nassar@uic.edu). The identity is used for:
  - Access control (domain check, e.g. @uic.edu only)
  - Per-request attribution in proxy logs
  - Rate limiting per individual user identity

NOTE: The proxy currently submits all Globus Compute jobs under its own stored
credentials (~/.globus_compute/storage.db), not under the caller's token.
Wiring caller.globus_token through to globus_compute_sdk for true per-user
SLURM attribution is a planned extension.

MODE B: API Key Auth (for external services like AWS/Amplify)
--------------------------------------------------------------
The caller presents a pre-issued service key (e.g., "sk-stream-amplify").
The proxy validates it against a local key table and logs the service name.
Used when the caller is a server (not a human) that authenticates its own users
separately (e.g., AWS Cognito). Per-user attribution lives in the caller's own
logs, not on Lakeshore.

The @uic.edu problem with Amplify users:
-----------------------------------------
If an Amplify user logs in with AWS Cognito, the proxy sees the Amplify server's
service key — not the user's UIC identity. There is no way for the proxy or
Lakeshore to know who that end user is unless Amplify implements one of:

  Option 1 (recommended): Amplify adds "Login with Globus" as an auth option.
    The user links their UIC/Globus account during Amplify signup. Amplify then
    holds a Globus token for that user and passes it to the proxy. Full per-user
    attribution on Lakeshore, no changes needed on the proxy side.

  Option 2 (simpler): Amplify authenticates as a service (Mode B above).
    Per-user attribution lives in Amplify's own logs. Acceptable for an
    institutional service operated by a trusted team.

AUTH HEADER FORMAT:
  Globus token:  Authorization: Bearer <globus_access_token>
  API key:       Authorization: Bearer sk-stream-<key>

The proxy distinguishes them by attempting Globus introspection first. If the
token is not a valid Globus token, it falls back to API key validation.
"""

import hashlib
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import httpx
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Globus Auth introspect endpoint — public, no special access needed
GLOBUS_INTROSPECT_URL = "https://auth.globus.org/v2/oauth2/token/introspect"

# Your Globus application's client_id and client_secret.
# Register at https://developers.globus.org → "Register a new app"
# The proxy uses these to authenticate *itself* when calling the introspect endpoint.
# These are NOT the user's credentials — they identify the proxy as a trusted app.
GLOBUS_CLIENT_ID = os.getenv("GLOBUS_CLIENT_ID", "")
GLOBUS_CLIENT_SECRET = os.getenv("GLOBUS_CLIENT_SECRET", "")  # pragma: allowlist secret

# Allowed identity domains for direct Globus auth.
# Users whose Globus email matches any of these patterns are allowed.
# Empty list = allow any valid Globus identity (less restrictive).
# Example: ["uic.edu", "illinois.edu"] allows both UIC and UIUC users.
ALLOWED_DOMAINS = [
    d.strip() for d in os.getenv("PROXY_ALLOWED_DOMAINS", "uic.edu").split(",") if d.strip()
]

# API key table for service-to-service callers (e.g., Amplify server on AWS).
# Format: "key": "service_name"
# Add new callers here and rotate keys via environment variables.
# In production, load this from a secrets manager (AWS Secrets Manager, Vault, etc.)
#
# Keys are stored as-is in memory but only their SHA-256 hash appears in logs,
# so log files never contain raw credentials.
_RAW_API_KEY_TABLE: dict[str, str] = {}

# Populate from environment — each key is PROXY_API_KEY_<NAME>=<value>
# Example .env:
#   PROXY_API_KEY_AMPLIFY=sk-stream-amplify-xxxx
#   PROXY_API_KEY_LANGCHAIN=sk-stream-langchain-yyyy
for _env_name, _env_val in os.environ.items():
    if _env_name.startswith("PROXY_API_KEY_") and _env_val:
        _service_name = _env_name[len("PROXY_API_KEY_") :].lower()
        _RAW_API_KEY_TABLE[_env_val] = _service_name

# Fallback: support single legacy PROXY_API_KEY for backwards compatibility
_legacy_key = os.getenv("PROXY_API_KEY", "")
if _legacy_key and _legacy_key not in _RAW_API_KEY_TABLE:
    _RAW_API_KEY_TABLE[_legacy_key] = "legacy"

# =============================================================================
# RATE LIMITING
# =============================================================================
#
# Simple in-memory rate limiter — per caller identity (Globus email or service name).
# For a multi-process deployment, replace with Redis-backed rate limiting.
#
# Limits are intentionally conservative to protect the shared HPC cluster.
# Each request dispatches a Globus Compute job that runs a 72B model on an H100.
# A single misbehaving caller could monopolize the cluster for everyone.

# Requests allowed per caller per time window
RATE_LIMIT_REQUESTS = int(os.getenv("PROXY_RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("PROXY_RATE_LIMIT_WINDOW", "60"))

# In-memory store: caller_id → list of request timestamps in the current window
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(caller_id: str) -> None:
    """
    Enforce per-caller rate limit. Raises HTTP 429 if the caller exceeds
    RATE_LIMIT_REQUESTS requests within RATE_LIMIT_WINDOW_SECONDS seconds.

    Uses a sliding window — only counts requests within the last N seconds,
    so the limit resets naturally without a cron job.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    # Remove timestamps outside the current window (sliding window cleanup)
    timestamps = _rate_limit_store[caller_id]
    _rate_limit_store[caller_id] = [t for t in timestamps if t > window_start]

    if len(_rate_limit_store[caller_id]) >= RATE_LIMIT_REQUESTS:
        logger.warning(
            f"Rate limit exceeded: caller={caller_id}, "
            f"requests={len(_rate_limit_store[caller_id])}/{RATE_LIMIT_REQUESTS} "
            f"in {RATE_LIMIT_WINDOW_SECONDS}s window"
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per "
                f"{RATE_LIMIT_WINDOW_SECONDS}s. Please slow down."
            ),
        )

    _rate_limit_store[caller_id].append(now)


# =============================================================================
# CALLER IDENTITY
# =============================================================================


@dataclass
class CallerIdentity:
    """
    Represents an authenticated caller — either a human user (Globus) or a
    service (API key). The rest of the proxy uses this to:
      - Log requests with attribution
      - Pass the correct token to Globus Compute
      - Apply per-caller rate limits
    """

    # Human-readable name for logs: "nassar@uic.edu" or "amplify-service"
    name: str

    # Auth mode used: "globus" or "api_key"
    auth_mode: str

    # For Globus auth: the raw access token to pass to globus_compute_sdk.
    # For API key auth: None — the proxy uses its own stored Globus credentials.
    globus_token: str | None = None

    # SHA-256 hash of the credential — safe to write to logs
    credential_hash: str = field(default="")

    def log_safe_id(self) -> str:
        """Returns a log-safe string identifying this caller."""
        return f"{self.auth_mode}:{self.name}:{self.credential_hash[:8]}"


# =============================================================================
# GLOBUS TOKEN VALIDATION
# =============================================================================


async def _validate_globus_token(token: str) -> CallerIdentity | None:
    """
    Validate a Globus access token by calling Globus Auth's introspect endpoint.

    The introspect endpoint is an OAuth2 standard — it tells us whether the
    token is valid, who it belongs to, and what scopes it has.

    We authenticate the introspect request with our own Globus client credentials
    (GLOBUS_CLIENT_ID + GLOBUS_CLIENT_SECRET). This proves to Globus that the
    proxy is a registered, trusted application — not just anyone calling introspect.

    Returns CallerIdentity if the token is valid and the caller is authorized.
    Returns None if the token is invalid, expired, or from a disallowed domain.
    """
    if not GLOBUS_CLIENT_ID or not GLOBUS_CLIENT_SECRET:
        # Globus auth is not configured on this proxy instance — skip
        return None

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                GLOBUS_INTROSPECT_URL,
                # HTTP Basic Auth: proxy authenticates itself to Globus
                auth=(GLOBUS_CLIENT_ID, GLOBUS_CLIENT_SECRET),
                # The token we want to validate
                data={"token": token, "include": "identities_set"},
            )

        if resp.status_code != 200:
            return None

        info = resp.json()

        # "active": false means the token is expired or revoked
        if not info.get("active", False):
            logger.debug("Globus token is inactive (expired or revoked)")
            return None

        # Extract the user's email from their Globus identity
        # Globus uses the email field from the identity provider (e.g., UIC's SSO)
        email = info.get("email", "") or info.get("username", "")
        if not email:
            logger.warning("Globus token valid but no email in identity")
            return None

        # Enforce domain restriction if PROXY_ALLOWED_DOMAINS is configured.
        # This is where @uic.edu filtering happens.
        if ALLOWED_DOMAINS:
            domain = email.split("@")[-1].lower() if "@" in email else ""
            if domain not in [d.lower() for d in ALLOWED_DOMAINS]:
                logger.warning(
                    f"Globus auth rejected: {email} not in allowed domains {ALLOWED_DOMAINS}"
                )
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Access restricted to users from: {', '.join(ALLOWED_DOMAINS)}. "
                        f"Your identity ({email}) is not from an allowed institution."
                    ),
                )

        token_hash = hashlib.sha256(token.encode()).hexdigest()
        logger.info(f"Globus token validated: identity={email}")

        return CallerIdentity(
            name=email,
            auth_mode="globus",
            globus_token=token,  # Stored for future per-user job submission (not yet wired through)
            credential_hash=token_hash,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.debug(f"Globus token validation failed: {e}")
        return None


# =============================================================================
# API KEY VALIDATION
# =============================================================================


def _validate_api_key(token: str) -> CallerIdentity | None:
    """
    Validate a pre-issued service API key against the key table.

    Used for server-to-server callers (Amplify, LangChain apps, etc.) that
    cannot present a Globus token because they authenticate their own users
    through a separate system (e.g., AWS Cognito).

    The raw key never appears in logs — only its SHA-256 hash is recorded.
    """
    service_name = _RAW_API_KEY_TABLE.get(token)
    if not service_name:
        return None

    key_hash = hashlib.sha256(token.encode()).hexdigest()
    logger.info(f"API key validated: service={service_name}, key_hash={key_hash[:16]}")

    return CallerIdentity(
        name=service_name,
        auth_mode="api_key",
        globus_token=None,  # Proxy will use its own stored Globus credentials
        credential_hash=key_hash,
    )


# =============================================================================
# MAIN AUTH DEPENDENCY
# =============================================================================

security = HTTPBearer()


async def authenticate(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> CallerIdentity:
    """
    FastAPI dependency — authenticates every request to the proxy.

    Authentication order:
      1. Try Globus token introspection (preferred — enables domain-based access control)
      2. Fall back to API key validation (for service callers)
      3. Reject with 401 if neither succeeds

    After authentication, applies per-caller rate limiting.

    Usage in routes:
        @router.post("/v1/chat/completions")
        async def chat(caller: CallerIdentity = Depends(authenticate)):
            ...
    """
    token = credentials.credentials

    # Step 1: Try Globus token validation
    # This covers: UIC researchers, any @uic.edu user, anyone who authenticates
    # via Globus (including users from other InCommon institutions if ALLOWED_DOMAINS
    # is broadened to include their domain).
    caller = await _validate_globus_token(token)

    # Step 2: Fall back to API key validation
    # This covers: the Amplify server, LangChain apps, any service caller
    # that was issued a pre-shared key.
    if caller is None:
        caller = _validate_api_key(token)

    # Step 3: Reject if neither worked
    if caller is None:
        logger.warning(
            f"Authentication failed from {request.client.host if request.client else 'unknown'}"
        )
        raise HTTPException(
            status_code=401,
            detail=(
                "Authentication required. Provide either:\n"
                "  • A valid Globus access token (for UIC/institutional users)\n"
                "  • A pre-issued service API key (for application integrations)\n"
                "Contact your STREAM administrator for access."
            ),
        )

    # Step 4: Rate limiting — applied after auth so we can rate-limit per identity
    _check_rate_limit(caller.name)

    # Step 5: Log the request with full attribution
    # Log the service/user name and credential hash — never the raw token/key
    logger.info(
        f"Authenticated request: caller={caller.log_safe_id()}, "
        f"path={request.url.path}, "
        f"client={request.client.host if request.client else 'unknown'}"
    )

    return caller


# =============================================================================
# INPUT VALIDATION
# =============================================================================


def validate_messages(messages: list) -> list:
    """
    Validate and sanitize the messages array before forwarding to Globus Compute.

    Why this matters: the proxy forwards messages to a Globus Compute function
    that runs on the HPC cluster. Malformed payloads could crash vLLM, cause
    unexpected behavior, or consume excessive resources.

    Checks:
      - messages is a list of dicts
      - each message has "role" (one of user/assistant/system) and "content"
      - content is a string or a list (for multimodal messages with images)
      - individual text content is under 100K characters
      - total message count is under 500 turns (prevents context window abuse)

    Returns the validated messages list.
    Raises HTTP 400 with a descriptive message for any violation.
    """
    allowed_roles = {"user", "assistant", "system"}
    max_content_chars = 100_000  # ~75K tokens — well within any model's context
    max_messages = 500  # 250 full back-and-forth turns

    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="'messages' must be a list")

    if len(messages) == 0:
        raise HTTPException(status_code=400, detail="'messages' cannot be empty")

    if len(messages) > max_messages:
        raise HTTPException(
            status_code=400, detail=f"Too many messages: {len(messages)} > {max_messages} limit"
        )

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise HTTPException(
                status_code=400, detail=f"Message {i}: must be a dict, got {type(msg).__name__}"
            )

        role = msg.get("role")
        if role not in allowed_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i}: invalid role '{role}'. Must be one of: {allowed_roles}",
            )

        content = msg.get("content")
        if content is None:
            raise HTTPException(status_code=400, detail=f"Message {i}: missing 'content' field")

        # Content can be a string (text only) or a list (multimodal: text + images)
        if isinstance(content, str):
            if len(content) > max_content_chars:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Message {i}: content too large "
                        f"({len(content):,} chars > {max_content_chars:,} limit)"
                    ),
                )
        elif isinstance(content, list):
            # Multimodal content: list of {"type": "text", "text": "..."} or
            # {"type": "image_url", "image_url": {"url": "data:image/..."}}
            # We just check it's a list of dicts — deep validation is handled
            # by vLLM itself when the request arrives on the cluster.
            for j, part in enumerate(content):
                if not isinstance(part, dict):
                    raise HTTPException(
                        status_code=400, detail=f"Message {i}, content part {j}: must be a dict"
                    )
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Message {i}: 'content' must be a string or list, "
                    f"got {type(content).__name__}"
                ),
            )

    return messages
