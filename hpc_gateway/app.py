"""
HPC Gateway — FastAPI application for routing LLM requests to HPC clusters.

DUAL-USE DESIGN:
----------------
This module serves two roles:

1. STANDALONE SERVICE:
   Run as its own process with uvicorn. The caller sends OpenAI-compatible
   /v1/chat/completions requests; the gateway dispatches them to the HPC cluster
   via Globus Compute and streams tokens back through the WebSocket relay.
   → Start with: uvicorn hpc_gateway.app:app --host 0.0.0.0 --port 8001

2. EMBEDDED ROUTER:
   Import `router` and mount it in any existing FastAPI application:
       from hpc_gateway.app import router
       app.include_router(router, prefix="/hpc")
   Same routes, same logic, no separate process needed.

CONFIGURATION:
--------------
All settings come from environment variables. No config files to manage:

  GLOBUS_COMPUTE_ENDPOINT_ID   UUID of the HPC cluster's Globus endpoint
  HPC_MODELS                   JSON dict mapping model names to their config
                                (see GlobusComputeClient for the schema)
  RELAY_URL                    WebSocket URL of the relay server
  RELAY_SECRET                 Shared secret for relay authentication
  RELAY_ENCRYPTION_KEY         AES-256 key (hex) for E2E relay encryption

  HPC_PROXY_HOST               Host to bind to (default: 0.0.0.0)
  HPC_PROXY_PORT               Port to listen on (default: 8001)
  USE_GLOBUS_COMPUTE           "true"/"false" (default: true)
  VLLM_SERVER_URL              Fallback vLLM URL when not using Globus
  LOG_LEVEL                    Logging level (default: INFO)

AUTHENTICATION:
---------------
Every /v1/* endpoint requires authentication. Two modes are supported:
  - Globus token: Bearer token from Globus Auth (checked via introspection)
  - API key: Static API key from the HPC_API_KEYS env var

Set USE_GLOBUS_AUTH=true to enable Globus token validation.
Set HPC_API_KEYS to a comma-separated list of valid API keys.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from hpc_gateway.auth import CallerIdentity, authenticate, validate_messages
from hpc_gateway.compute import GlobusComputeClient
from hpc_gateway.crypto import decrypt_message

# =========================================================================
# Configuration — all from environment variables
# =========================================================================
PROXY_HOST = os.getenv("HPC_PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.getenv("HPC_PROXY_PORT", "8001"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

USE_GLOBUS_COMPUTE = os.getenv("USE_GLOBUS_COMPUTE", "true").lower() == "true"
GLOBUS_COMPUTE_ENDPOINT_ID = os.getenv("GLOBUS_COMPUTE_ENDPOINT_ID")

# Fallback vLLM URL (used when USE_GLOBUS_COMPUTE=false — direct SSH tunnel mode)
LAKESHORE_VLLM_ENDPOINT = os.getenv("LAKESHORE_VLLM_ENDPOINT", "http://localhost:8000")

# WebSocket relay for token streaming
RELAY_URL = os.getenv("RELAY_URL", "")
RELAY_SECRET = os.getenv("RELAY_SECRET", "")
RELAY_ENCRYPTION_KEY = os.getenv("RELAY_ENCRYPTION_KEY", "")

# Model registry — JSON string mapping model names to their HPC-side config.
# Example (set as environment variable):
#   HPC_MODELS='{"qwen25-vl-72b": {"hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
#                                   "url": "http://ghi2-002:8000",
#                                   "context_reserve_output": 4096}}'
_HPC_MODELS: dict = {}
_raw_models = os.getenv("HPC_MODELS", "{}")
try:
    _HPC_MODELS = json.loads(_raw_models)
except json.JSONDecodeError:
    logging.warning("HPC_MODELS env var is not valid JSON — no models registered")

logger = logging.getLogger(__name__)

# =========================================================================
# Globus Compute client — initialized once at startup
# =========================================================================
globus_client: GlobusComputeClient | None = None


def _init_globus_client() -> GlobusComputeClient | None:
    """Create and return a GlobusComputeClient, or None if Globus is disabled/unconfigured."""
    if not USE_GLOBUS_COMPUTE:
        return None
    if not GLOBUS_COMPUTE_ENDPOINT_ID:
        logger.warning(
            "USE_GLOBUS_COMPUTE=true but GLOBUS_COMPUTE_ENDPOINT_ID is not set — "
            "Globus Compute will be unavailable."
        )
        return None
    try:
        client = GlobusComputeClient(
            endpoint_id=GLOBUS_COMPUTE_ENDPOINT_ID,
            models=_HPC_MODELS,
            relay_secret=RELAY_SECRET,
        )
        logger.info("Globus Compute client initialized")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Globus Compute client: {e}")
        return None


# =========================================================================
# FastAPI lifespan — replaces deprecated @app.on_event("startup")
# =========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    global globus_client
    globus_client = _init_globus_client()

    logger.info("=" * 60)
    logger.info("HPC Gateway Starting")
    logger.info("=" * 60)
    logger.info(f"Mode: {'Globus Compute' if USE_GLOBUS_COMPUTE else 'SSH / Direct'}")
    if globus_client:
        logger.info(f"Globus Endpoint: {GLOBUS_COMPUTE_ENDPOINT_ID}")
        logger.info(f"Models: {list(_HPC_MODELS.keys())}")
    if RELAY_URL:
        logger.info(f"Relay: {RELAY_URL}")
    logger.info(f"Listening on: {PROXY_HOST}:{PROXY_PORT}")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown: clean up the persistent Globus executor
    if globus_client:
        logger.info("Shutting down Globus Compute client...")
        globus_client.shutdown()


# =========================================================================
# APIRouter — the actual route definitions
# =========================================================================
# Using a router lets this module be embedded into any FastAPI app:
#   app.include_router(hpc_gateway.app.router, prefix="/hpc")
router = APIRouter()


@router.get("/health")
async def health_check():
    """Return service health — no authentication required."""
    return {
        "status": "healthy",
        "service": "HPC Gateway",
        "mode": "globus_compute" if USE_GLOBUS_COMPUTE else "direct",
        "globus_configured": bool(globus_client and globus_client.is_available()),
        "models": list(_HPC_MODELS.keys()),
        "relay_configured": bool(RELAY_URL),
    }


@router.get("/v1/models")
async def list_models(caller: CallerIdentity = Depends(authenticate)):
    """
    List available HPC models in OpenAI-compatible format.

    Returns the same schema as GET /v1/models on OpenAI's API, so any
    OpenAI-compatible client can discover what models are available.
    """
    from time import time as now

    models = []
    for name, info in _HPC_MODELS.items():
        models.append(
            {
                "id": info.get("hf_name", name),
                "object": "model",
                "created": int(now()),
                "owned_by": "hpc-gateway",
                # Include the gateway-internal name as metadata.
                # Callers can use either name — hf_name or the registry key.
                "gateway_name": name,
            }
        )

    return {"object": "list", "data": models}


@router.post("/reload-auth")
async def reload_authentication():
    """
    Force reload of Globus credentials from ~/.globus_compute/storage.db.

    Call this after authenticating on the host machine. The Globus SDK caches
    credentials in a module-level singleton; this clears the cache so fresh
    tokens are re-read from disk.
    """
    if not USE_GLOBUS_COMPUTE or not globus_client:
        return {"success": False, "message": "Globus Compute not configured"}

    try:
        success, message = globus_client.reload_credentials()
        return {"success": success, "message": message}
    except Exception as e:
        logger.error(f"Failed to reload credentials: {e}")
        return {"success": False, "message": f"Failed to reload: {str(e)}"}


@router.post("/v1/chat/completions")
async def proxy_chat_completions(
    request: Request,
    caller: CallerIdentity = Depends(authenticate),
):
    """
    Forward a chat completion request to the HPC cluster.

    Accepts the same request body as OpenAI's POST /v1/chat/completions.
    Responses are also OpenAI-compatible — either a streaming SSE response
    or a complete JSON response depending on the `stream` field.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from e

    raw_model = body.get("model", "")
    # LiteLLM adds an "openai/" prefix when forwarding. Strip it so the model
    # name matches keys in the HPC_MODELS registry.
    model = raw_model.removeprefix("openai/")

    # Validate and sanitize messages before sending to the HPC cluster.
    messages = validate_messages(body.get("messages", []))
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens")  # None means "use model default"

    logger.info(
        f"Chat request: caller={caller.log_safe_id()}, model={model}, "
        f"messages={len(messages)}, stream={stream}"
    )

    if USE_GLOBUS_COMPUTE:
        return await _route_via_globus_compute(model, messages, temperature, max_tokens, stream)
    else:
        return await _route_via_direct(model, messages, temperature, max_tokens, stream)


# =========================================================================
# Internal routing helpers
# =========================================================================

async def _route_via_globus_compute(model, messages, temperature, max_tokens, stream):
    """Route a request to the HPC cluster via Globus Compute."""
    if not globus_client or not globus_client.is_available():
        raise HTTPException(status_code=503, detail="Globus Compute not configured")

    # True streaming: submit job, open relay channel, yield tokens as they arrive.
    # Falls back to batch mode if the relay connection fails.
    if stream and RELAY_URL:
        try:
            return await _route_via_globus_compute_streaming(
                model, messages, temperature, max_tokens
            )
        except Exception as e:
            logger.warning(f"Relay streaming failed — falling back to batch mode: {e}")
            # Fall through to batch mode

    # Batch mode: submit job, wait for complete result, return as JSON (or simulated stream).
    try:
        logger.info(f"Submitting batch job to Globus endpoint: {GLOBUS_COMPUTE_ENDPOINT_ID}")
        result = await globus_client.submit_inference(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

        if "error" in result:
            error_msg = result.get("error", "Unknown error")
            error_type = result.get("error_type", "UnknownError")
            if error_type == "AuthenticationError":
                raise HTTPException(
                    status_code=401,
                    detail=f"Globus Compute authentication required: {error_msg}",
                )
            raise HTTPException(status_code=503, detail=f"HPC inference failed: {error_msg}")

        logger.info("Batch inference completed successfully")

        if stream:
            # Caller wants streaming but relay is unavailable.
            # Simulate streaming by splitting the complete response into word chunks.
            return _convert_json_to_sse_stream(result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Globus Compute routing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal gateway error: {str(e)}") from e


async def _route_via_globus_compute_streaming(model, messages, temperature, max_tokens):
    """
    True streaming from the HPC cluster via the WebSocket relay.

    1. Submit a streaming job to Globus Compute → returns channel_id immediately
    2. Connect to relay as consumer on that channel
    3. Receive tokens in real-time, convert to SSE, yield to caller

    The remote function on the HPC side connects to the relay as producer and
    streams tokens through it. The encryption key (RELAY_ENCRYPTION_KEY) is read
    from the endpoint's environment — it never travels over the Globus AMQP channel.
    """
    from websockets.asyncio.client import connect as ws_connect

    result = await globus_client.submit_streaming_inference(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        relay_url=RELAY_URL,
    )

    if "error" in result:
        error_msg = result.get("error", "Unknown error")
        error_type = result.get("error_type", "UnknownError")
        if error_type == "AuthenticationError":
            raise HTTPException(
                status_code=401,
                detail=f"Globus Compute authentication required: {error_msg}",
            )
        raise HTTPException(status_code=503, detail=f"HPC streaming failed: {error_msg}")

    channel_id = result["channel_id"]
    logger.info(f"Relay streaming: channel={channel_id[:8]}, relay={RELAY_URL}")

    async def sse_generator():
        """Connect to relay and yield SSE events as tokens arrive."""
        try:
            relay_consume_url = f"{RELAY_URL}/consume/{channel_id}"
            async with ws_connect(relay_consume_url) as ws:
                # Post-handshake auth: send secret as the first message, not in the URL.
                # This way it never appears in HTTP access logs or proxied headers.
                if RELAY_SECRET:
                    await ws.send(json.dumps({"type": "auth", "secret": RELAY_SECRET}))

                async for msg_str in ws:
                    # E2E decryption: if RELAY_ENCRYPTION_KEY is set, the producer
                    # encrypted each token payload before sending. decrypt_message()
                    # unwraps the {"type":"enc","d":"..."} envelope and returns the
                    # original plaintext JSON. If no key is configured, passthrough.
                    if RELAY_ENCRYPTION_KEY:
                        msg_str = decrypt_message(RELAY_ENCRYPTION_KEY, msg_str)

                    msg = json.loads(msg_str)

                    if msg["type"] == "token":
                        chunk = {
                            "choices": [{"index": 0, "delta": {"content": msg["content"]}}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    elif msg["type"] == "done":
                        usage = msg.get("usage", {})
                        if usage:
                            final_chunk = {
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                                ],
                                "usage": usage,
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break

                    elif msg["type"] == "error":
                        logger.error(
                            f"Relay error on channel {channel_id[:8]}: {msg.get('message')}"
                        )

        except Exception as e:
            logger.error(f"Relay connection failed: {e}", exc_info=True)
            error_chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


async def _route_via_direct(model, messages, temperature, max_tokens, stream):
    """
    Route directly to a vLLM server — no Globus Compute.

    Used when USE_GLOBUS_COMPUTE=false (e.g., SSH tunnel is already open and
    you want to skip the Globus auth layer). LAKESHORE_VLLM_ENDPOINT should
    point to the tunnel's local end.
    """
    if max_tokens is None:
        max_tokens = 2048  # safe default when no model registry is available

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    target_url = f"{LAKESHORE_VLLM_ENDPOINT}/v1/chat/completions"
    logger.info(f"Direct vLLM request: {target_url}")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                async with client.stream("POST", target_url, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"vLLM error: {error_text.decode()}",
                        )

                    async def stream_generator():
                        async for line in response.aiter_lines():
                            if line.strip():
                                yield line + "\n"

                    return StreamingResponse(stream_generator(), media_type="text/event-stream")
            else:
                response = await client.post(target_url, json=payload)
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"vLLM error: {response.text}",
                    )
                return response.json()

    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to vLLM. Is the tunnel running? Error: {str(e)}",
        ) from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail="vLLM request timed out") from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal gateway error: {str(e)}") from e


def _convert_json_to_sse_stream(json_response: dict):
    """
    Simulate streaming by splitting a complete response into word-sized SSE chunks.

    Globus Compute is batch-only — the remote function returns the full response
    at once. When the caller requested `stream=true` but the relay is unavailable,
    we still return an SSE response to keep the API contract, yielding a few words
    at a time with small delays to create a natural typing effect.

    Splitting on words rather than characters avoids cutting through multi-byte
    Unicode sequences and keeps each chunk visually coherent.
    """
    words_per_chunk = 2       # 2 words → smooth appearance without excessive events
    delay_between_chunks = 0.05  # 50 ms → ~40 words/second reading pace

    async def sse_generator():
        choices = json_response.get("choices", [])
        if not choices:
            yield "data: [DONE]\n\n"
            return

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        role = message.get("role", "assistant")

        chunk_base = {
            "id": json_response.get("id", ""),
            "object": "chat.completion.chunk",
            "created": json_response.get("created", 0),
            "model": json_response.get("model", ""),
        }

        # First chunk carries the role (OpenAI protocol requires this)
        if role:
            chunk = {
                **chunk_base,
                "choices": [
                    {"index": 0, "delta": {"role": role, "content": ""}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Content chunks: group into small word batches
        if content:
            words = content.split(" ")
            for i in range(0, len(words), words_per_chunk):
                word_group = words[i : i + words_per_chunk]
                # Space before words except the very first chunk
                text_chunk = " ".join(word_group) if i == 0 else " " + " ".join(word_group)
                chunk = {
                    **chunk_base,
                    "choices": [
                        {"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(delay_between_chunks)

        # Final chunk signals completion and includes usage stats
        chunk = {
            **chunk_base,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}
            ],
            "usage": json_response.get("usage", {}),
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# =========================================================================
# Standalone FastAPI app (used when running as a service)
# =========================================================================
app = FastAPI(
    title="HPC Gateway",
    description="OpenAI-compatible API gateway for HPC clusters via Globus Compute",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)


def main():
    import uvicorn

    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level=LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
