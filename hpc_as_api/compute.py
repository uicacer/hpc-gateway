"""
Globus Compute client for submitting vLLM inference tasks to HPC clusters.

This module is the core of hpc-as-api. It provides:
  1. Batch inference: submit a job to Globus Compute, wait for the full result
  2. Streaming inference: submit a job, receive tokens in real-time via relay

ARCHITECTURE OVERVIEW
=====================

                  ┌─────────────────────────────────┐
                  │         Your Application         │
                  │  (FastAPI, LangChain, any client) │
                  └──────────────┬──────────────────┘
                                 │ awaits result
                                 ▼
                  ┌─────────────────────────────────┐
                  │       GlobusComputeClient        │
                  │  (this module — runs locally)   │
                  └──────────────┬──────────────────┘
                                 │ submits job via AMQP
                                 ▼
                  ┌─────────────────────────────────┐
                  │         Globus Cloud             │
                  │  (routes task to HPC endpoint)  │
                  └──────────────┬──────────────────┘
                                 │ executes on
                                 ▼
                  ┌─────────────────────────────────┐
                  │       HPC Cluster (SLURM)        │
                  │  remote_vllm_* runs on GPU node │
                  │  calls vLLM HTTP API directly   │
                  └──────────────┬──────────────────┘
                                 │ tokens flow via WebSocket relay
                                 ▼
                  ┌─────────────────────────────────┐
                  │      WebSocket Relay Server      │
                  │  (streamrelay — public URL)     │
                  └──────────────┬──────────────────┘
                                 │ streamed to consumer
                                 ▼
                  ┌─────────────────────────────────┐
                  │      Consumer (your app)         │
                  │  SSE events → user's browser    │
                  └─────────────────────────────────┘

GLOBUS COMPUTE IN ONE PARAGRAPH
=================================
Globus Compute is a Function-as-a-Service system for HPC clusters. You give it
a Python function and arguments; it serializes them, sends them to the HPC
endpoint (a daemon running on the cluster's login node), which dispatches them
to a SLURM job running on a GPU compute node. The result comes back through the
same channel. It requires no open ports on the HPC side — everything is outbound
from the cluster to Globus cloud via AMQP.
"""

import asyncio
import logging
import os
import time
import warnings
from typing import Any

from globus_compute_sdk import Executor
from globus_compute_sdk.errors.error_types import DeserializationError, TaskExecutionFailed
from globus_compute_sdk.serialize import AllCodeStrategies, ComputeSerializer
from globus_sdk import GlobusAPIError
from globus_sdk.login_flows.command_line_login_flow_manager import CommandLineLoginFlowEOFError

from hpc_as_api.utils import strip_old_images

logger = logging.getLogger(__name__)

# Suppress the Globus SDK's "Environment differences detected" warning.
# This fires when the local Python version (e.g., 3.12.12) differs slightly
# from the endpoint workers (e.g., 3.12.3). Minor patch differences are harmless.
warnings.filterwarnings(
    "ignore", message=r"[\s\S]*Environment differences detected", category=UserWarning
)

# Default task timeout — how long to wait for a Globus Compute job to return.
# Override with GLOBUS_TASK_TIMEOUT env var or pass timeout= to submit methods.
_DEFAULT_TASK_TIMEOUT = int(os.getenv("GLOBUS_TASK_TIMEOUT", "240"))


# =============================================================================
# REMOTE FUNCTION: BATCH INFERENCE (executes on the HPC cluster)
# =============================================================================
#
# This function is serialized as source code and executed REMOTELY on the HPC
# cluster via Globus Compute. It must be completely self-contained: all imports
# inside the function body, no references to anything outside.
#
# WHY exec() FROM A SOURCE STRING:
# ---------------------------------
# PyInstaller (used for desktop builds) bundles .pyc bytecode with internal
# import references. When Globus Compute serializes a normal function, it
# captures that bytecode. The HPC endpoint doesn't have PyInstaller, so
# deserialization fails with "No module named 'pyimod02_importers'".
#
# By defining the function from a source STRING via exec() at runtime, Python's
# standard compiler produces clean bytecode with no bundler references.
#
# This pattern was tested against: CombinedCode strategy, __module__='__main__',
# and normal dill by-reference. Only the exec()-from-string approach works
# reliably across both normal Python and PyInstaller-bundled environments.

_REMOTE_FN_SOURCE = """\
def remote_vllm_inference(vllm_url, model, messages, temperature, max_tokens, stream=False):
    \"\"\"
    Execute a vLLM inference request on the HPC cluster.

    This function runs on the GPU compute node. It calls the vLLM HTTP API
    directly (no network hops needed — vLLM is on the same node or LAN).

    Args:
        vllm_url: HTTP URL of the vLLM server (e.g., "http://ghi2-002:8000")
        model: HuggingFace model name (e.g., "Qwen/Qwen2.5-VL-72B-Instruct-AWQ")
        messages: OpenAI-format message list
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        stream: Whether to request SSE streaming from vLLM (not used in batch mode)

    Returns:
        OpenAI-format response dict on success, or {"error": ..., "error_type": ...} on failure.
    \"\"\"
    try:
        import requests
        endpoint = f"{vllm_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=180)
            if response.status_code >= 400:
                try:
                    error_body = response.json()
                except Exception:
                    error_body = response.text
                return {
                    "error": f"{response.status_code} Error: {error_body}",
                    "error_type": "HTTPError",
                    "status_code": response.status_code,
                    "response_body": error_body,
                    "request_payload": payload,
                }
            return response.json()
        except requests.exceptions.RequestException as e:
            error_response = None
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_response = e.response.json()
                except Exception:
                    error_response = e.response.text if hasattr(e.response, "text") else str(e.response)
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "status_code": getattr(e.response, "status_code", None) if hasattr(e, "response") else None,
                "response_body": error_response,
            }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "error_type": type(e).__name__}
"""

_ns = {}
exec(compile(_REMOTE_FN_SOURCE, "<remote_vllm_inference>", "exec"), _ns)  # nosec B102
remote_vllm_inference = _ns["remote_vllm_inference"]


# =============================================================================
# REMOTE FUNCTION: STREAMING INFERENCE (executes on the HPC cluster)
# =============================================================================
#
# Unlike the batch function above, this one doesn't return the response via
# Globus Compute. Instead it streams ALL data (tokens, usage, errors) through
# the WebSocket relay in real-time:
#
#   Step 1: Connect to relay as PRODUCER (outbound WebSocket from the GPU node)
#   Step 2: Call vLLM with stream=True → get Server-Sent Events one token at a time
#   Step 3: Forward each token through the relay to the waiting consumer
#   Step 4: Send "done" + usage stats through relay → consumer reads and closes
#
# The Globus Compute return value ({ok: True, tokens_sent: N}) is just a receipt.
# The consumer doesn't wait for it — it already got everything via the relay.

_REMOTE_STREAMING_FN_SOURCE = """\
def remote_vllm_streaming(vllm_url, model, messages, temperature, max_tokens, relay_url, channel_id, relay_secret=""):
    \"\"\"
    Execute a streaming vLLM inference on the HPC cluster.

    Streams tokens to the consumer via a WebSocket relay. The consumer
    connects to relay_url/consume/channel_id and receives tokens in real-time
    as the GPU generates them.

    The encryption key is read from os.environ on the HPC endpoint — it is
    NOT passed as a function argument, so it never travels over Globus Compute's
    AMQP channel. Set RELAY_ENCRYPTION_KEY in the Globus endpoint's config.yaml.
    \"\"\"
    import json
    import os
    import requests
    from websockets.sync.client import connect as ws_connect

    encryption_key = os.environ.get("RELAY_ENCRYPTION_KEY", "")

    def _encrypt(plaintext_json):
        \"\"\"
        AES-256-GCM encrypt a JSON string.
        Wire format: {"type": "enc", "d": "<base64(nonce + ciphertext + tag)>"}
        The relay forwards this opaque blob — it cannot read the plaintext.
        \"\"\"
        import base64
        import os as _os
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        key = bytes.fromhex(encryption_key)  # 32 bytes (AES-256)
        nonce = _os.urandom(12)              # fresh random nonce per message
        aesgcm = AESGCM(key)
        ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext_json.encode(), None)
        blob = base64.b64encode(nonce + ciphertext_with_tag).decode()
        return json.dumps({"type": "enc", "d": blob})

    def _send(ws, payload_dict):
        \"\"\"Encrypt-then-send if key configured, otherwise send plaintext JSON.\"\"\"
        raw = json.dumps(payload_dict)
        ws.send(_encrypt(raw) if encryption_key else raw)

    ws = None
    try:
        # Connect to relay as PRODUCER.
        # Secret is sent as first JSON message AFTER the WebSocket handshake
        # (not in the URL as ?secret=) so it never appears in HTTP access logs.
        ws = ws_connect(f"{relay_url}/produce/{channel_id}")
        if relay_secret:
            import json as _json
            ws.send(_json.dumps({"type": "auth", "secret": relay_secret}))

        # Call vLLM with stream=True to get tokens as SSE events.
        response = requests.post(
            f"{vllm_url}/v1/chat/completions",
            json={"model": model, "messages": messages, "temperature": temperature,
                  "max_tokens": max_tokens, "stream": True},
            stream=True,
            timeout=180,
        )

        if response.status_code >= 400:
            error_msg = f"vLLM HTTP {response.status_code}: {response.text[:300]}"
            _send(ws, {"type": "error", "message": error_msg})
            _send(ws, {"type": "done"})
            return {"error": error_msg}

        # Parse vLLM's SSE stream line by line.
        # Each line looks like: "data: {"choices":[{"delta":{"content":"Hello"}}]}"
        # Blank lines are SSE event separators — skip them.
        usage = {}
        tokens_sent = 0
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                _send(ws, {"type": "token", "content": content})
                tokens_sent += 1
            if chunk.get("usage"):
                usage = chunk["usage"]

        # Signal stream completion. Consumer reads this and closes its connection.
        _send(ws, {"type": "done", "usage": usage})

    except Exception as e:
        if ws:
            try:
                _send(ws, {"type": "error", "message": str(e)})
                _send(ws, {"type": "done"})
            except Exception:
                pass
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        if ws:
            try:
                ws.close()
            except Exception:
                pass

    # Globus Compute requires a return value. The consumer already received
    # everything via the relay — this is just a receipt.
    return {"ok": True, "tokens_sent": tokens_sent}
"""

_ns2 = {}
exec(compile(_REMOTE_STREAMING_FN_SOURCE, "<remote_vllm_streaming>", "exec"), _ns2)  # nosec B102
remote_vllm_streaming = _ns2["remote_vllm_streaming"]


# =============================================================================
# GLOBUS COMPUTE CLIENT CLASS
# =============================================================================


class GlobusComputeClient:
    """
    Client for submitting vLLM inference tasks to an HPC cluster via Globus Compute.

    Configuration is passed via constructor arguments (not imported from a
    centralized config module). This makes the client self-contained and usable
    as a library without any STREAM-specific dependencies.

    Quick start:
        client = GlobusComputeClient(
            endpoint_id="8d978809-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            models={
                "qwen25-vl-72b": {
                    "hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
                    "url": "http://ghi2-002:8000",
                    "context_total": 65536,
                    "context_reserve_output": 4096,
                }
            },
            relay_secret=os.getenv("RELAY_SECRET", ""),
        )

        if client.is_available():
            result = await client.submit_inference(
                messages=[{"role": "user", "content": "Hello!"}],
                temperature=0.7,
                max_tokens=512,
                model="qwen25-vl-72b",
            )

    The `models` dict maps your model names to their configuration.
    Each entry needs at minimum: "hf_name" and "url".
    "context_reserve_output" sets the default max_tokens when the caller
    doesn't specify one.
    """

    def __init__(
        self,
        endpoint_id: str | None = None,
        models: dict | None = None,
        relay_secret: str = "",
        max_payload_bytes: int | None = None,
        task_timeout: int | None = None,
    ):
        """
        Initialize the Globus Compute client.

        Args:
            endpoint_id: Globus Compute endpoint UUID for the HPC cluster.
                         Falls back to GLOBUS_COMPUTE_ENDPOINT_ID env var.
            models: Dict mapping model names to their configuration.
                    Each entry: {"hf_name": str, "url": str,
                                 "context_reserve_output": int, ...}
                    Falls back to HPC_MODELS env var (JSON string) if not provided.
            relay_secret: Shared secret for WebSocket relay authentication.
                          Falls back to RELAY_SECRET env var.
            max_payload_bytes: Max serialized payload size before rejecting a request.
                               Globus Compute enforces a 10 MB hard limit.
                               Default 8 MB (leaves headroom). Falls back to
                               HPC_MAX_PAYLOAD_BYTES env var.
            task_timeout: Seconds to wait for a Globus Compute job to return.
                          Falls back to GLOBUS_TASK_TIMEOUT env var (default 240s).
        """
        # Endpoint ID — the UUID that identifies this HPC cluster's Globus endpoint.
        # You get this when you register the endpoint with `globus-compute-endpoint start`.
        self.endpoint_id = endpoint_id or os.getenv("GLOBUS_COMPUTE_ENDPOINT_ID")

        # Model registry — maps your model names to their HPC-side configuration.
        # If not passed, try to read from HPC_MODELS env var (JSON string).
        import json as _json
        if models is not None:
            self.models = models
        else:
            raw = os.getenv("HPC_MODELS", "{}")
            try:
                self.models = _json.loads(raw)
            except _json.JSONDecodeError:
                logger.warning("HPC_MODELS env var is not valid JSON — using empty model registry")
                self.models = {}

        # Relay secret — sent as the first WebSocket message after handshake.
        # Must match RELAY_SECRET on the relay server.
        self.relay_secret = relay_secret or os.getenv("RELAY_SECRET", "")

        # Payload size limit — reject messages that would exceed Globus's hard limit.
        # 8 MB default leaves 2 MB headroom below the 10 MB hard limit.
        self.max_payload_bytes = max_payload_bytes or int(
            os.getenv("HPC_MAX_PAYLOAD_BYTES", str(8 * 1024 * 1024))
        )

        # Task timeout — how long to wait for a Globus job to return.
        self.task_timeout = task_timeout or _DEFAULT_TASK_TIMEOUT

        # Internal state
        self._executor = None
        self._globus_app = None

        if self.endpoint_id:
            logger.info(
                f"GlobusComputeClient initialized: endpoint={self.endpoint_id}, "
                f"models={list(self.models.keys())}"
            )
        else:
            logger.warning(
                "No Globus Compute endpoint configured. "
                "Pass endpoint_id= or set GLOBUS_COMPUTE_ENDPOINT_ID."
            )

    def is_available(self) -> bool:
        """Return True if an endpoint ID is configured."""
        return bool(self.endpoint_id and self.endpoint_id.strip())

    def _resolve_model(self, model: str) -> tuple[str, str, int]:
        """
        Resolve a model name to its HPC-side (hf_name, vllm_url, default_max_tokens).

        The model registry maps the names callers use (e.g., "qwen25-vl-72b") to
        the HuggingFace name vLLM expects ("Qwen/Qwen2.5-VL-72B-Instruct-AWQ"),
        the URL of the vLLM server on the HPC cluster, and the default context limits.

        Returns:
            (hf_name, vllm_url, default_max_tokens)

        Raises:
            KeyError: if model is not in the registry and no fallback exists.
        """
        info = self.models.get(model)
        if info:
            hf_name = info.get("hf_name", model)
            url = info.get("url", os.getenv("HPC_VLLM_URL", "http://localhost:8000"))
            default_max_tokens = info.get("context_reserve_output", 2048)
        else:
            # Fallback: treat model as the HF name directly, use default URL
            logger.warning(
                f"Model '{model}' not in registry — using as HF name directly. "
                "Add it to the models dict for explicit configuration."
            )
            hf_name = model
            url = os.getenv("HPC_VLLM_URL", "http://localhost:8000")
            default_max_tokens = 2048
        return hf_name, url, default_max_tokens

    # =========================================================================
    # PERSISTENT EXECUTOR
    # =========================================================================
    #
    # The Globus Compute Executor maintains a persistent AMQP connection to
    # Globus cloud. Creating a new one per request costs ~1-2 seconds for the
    # TCP + AMQP handshake. By keeping one Executor alive across requests,
    # subsequent requests reuse the existing connection (~100ms overhead instead
    # of ~1500ms).
    #
    # If the AMQP connection dies (network glitch, token expiry, Globus restart),
    # _reset_executor() clears it so the next call to _get_executor() creates
    # a fresh connection.

    def _get_executor(self) -> Executor:
        """Get or create the persistent Globus Compute Executor."""
        if self._executor is None:
            logger.info("Creating persistent Globus Compute Executor (AMQP connect)...")
            self._executor = Executor(endpoint_id=self.endpoint_id)
            # AllCodeStrategies tries multiple serialization methods (dill by-value,
            # dill by-reference, cloudpickle) to find one that works across Python
            # version differences between local and endpoint.
            self._executor.serializer = ComputeSerializer(strategy_code=AllCodeStrategies())
            logger.info("Executor ready")
        return self._executor

    def _reset_executor(self):
        """Close the current Executor and clear it. Next call recreates it."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.debug(f"Executor shutdown error (expected if connection was dead): {e}")
            self._executor = None
            logger.info("Executor reset — will reconnect on next request")

    def shutdown(self):
        """Clean up the persistent Executor. Call during app shutdown."""
        logger.info("Shutting down GlobusComputeClient...")
        self._reset_executor()

    def _get_globus_app(self, force_refresh: bool = False):
        """Get the Globus app instance for authentication checks."""
        if force_refresh:
            try:
                from globus_compute_sdk.sdk.auth import globus_app as globus_app_module
                if hasattr(globus_app_module, "_globus_app"):
                    globus_app_module._globus_app = None
                if hasattr(globus_app_module, "GLOBUS_APP"):
                    globus_app_module.GLOBUS_APP = None
            except Exception:
                pass
            self._globus_app = None

        if self._globus_app is None:
            from globus_compute_sdk.sdk.auth.globus_app import get_globus_app
            self._globus_app = get_globus_app()
        return self._globus_app

    def reload_credentials(self) -> tuple[bool, str]:
        """
        Force reload of Globus credentials from ~/.globus_compute/storage.db.

        Call this after the user authenticates on the host machine to pick up
        the newly saved tokens. The Globus SDK caches credentials in a singleton;
        this method clears that cache so fresh tokens are read from disk.

        Returns:
            (success: bool, message: str)
        """
        try:
            logger.info("Reloading Globus credentials from disk...")
            self._globus_app = None
            try:
                from globus_compute_sdk.sdk.auth import globus_app as m
                if hasattr(m, "_globus_app"):
                    m._globus_app = None
                if hasattr(m, "GLOBUS_APP"):
                    m.GLOBUS_APP = None
            except Exception as e:
                logger.debug(f"Could not clear SDK cache: {e}")

            app = self._get_globus_app(force_refresh=True)
            if app.login_required():
                return False, "Credentials not found. Please authenticate first."
            return True, "Credentials reloaded successfully"
        except Exception as e:
            logger.error(f"Failed to reload credentials: {e}")
            return False, f"Failed to reload credentials: {str(e)}"

    def ensure_authenticated(self, force_refresh: bool = False) -> tuple[bool, str | None]:
        """
        Check if user is authenticated with Globus Compute.

        Returns:
            (is_authenticated, error_message_or_None)
        """
        try:
            app = self._get_globus_app(force_refresh=force_refresh)
            if app.login_required():
                logger.warning("Globus Compute authentication required")
                return False, (
                    "Globus Compute authentication required. "
                    "Run: globus-compute-endpoint login
"
                    "Or visit: https://app.globus.org/"
                )
            return True, None
        except Exception as e:
            logger.error(f"Authentication check failed: {e}")
            return False, f"Authentication check failed: {str(e)}"

    def _estimate_payload_size(self, messages: list[dict]) -> int:
        """
        Estimate the serialized payload size in bytes.

        Globus Compute enforces a 10 MB payload limit. Base64-encoded images are
        the main size contributor (~1.33x the original file size). This estimate
        uses JSON length as a proxy — accurate enough for the size check.
        """
        import json
        try:
            return len(json.dumps(messages).encode("utf-8"))
        except (TypeError, ValueError):
            return len(str(messages).encode("utf-8"))

    # =========================================================================
    # BATCH INFERENCE
    # =========================================================================

    async def submit_inference(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str = "",
        _retry: bool = False,
    ) -> dict[str, Any]:
        """
        Submit a batch inference job to the HPC cluster and wait for the full result.

        HOW IT WORKS:
        1. Strip old images from history (keeps payload under Globus's 8 MB limit)
        2. Validate payload size
        3. Check Globus authentication
        4. Submit remote_vllm_inference to Globus Compute via the persistent Executor
        5. Wait for the result in a background thread (asyncio.to_thread keeps event
           loop responsive during the 5-30 second GPU inference wait)
        6. Return the vLLM response (OpenAI format) or an error dict

        Args:
            messages: Conversation history in OpenAI format
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Max tokens to generate. If None, uses model's
                        context_reserve_output from the models registry.
            model: Model name as registered in the models dict
            _retry: Internal flag — do not pass. Prevents infinite retry loops.

        Returns:
            OpenAI-format response dict on success.
            {"error": str, "error_type": str, ...} on failure.
        """
        hf_name, vllm_url, default_max_tokens = self._resolve_model(model)
        if max_tokens is None:
            max_tokens = default_max_tokens

        if not self.is_available():
            raise RuntimeError("No endpoint configured. Pass endpoint_id= to GlobusComputeClient.")

        # Strip images from all but the latest user message to stay under payload limit.
        messages = strip_old_images(messages)

        # Reject before submitting if payload is already too large.
        estimated_size = self._estimate_payload_size(messages)
        if estimated_size > self.max_payload_bytes:
            size_mb = estimated_size / (1024 * 1024)
            limit_mb = self.max_payload_bytes / (1024 * 1024)
            logger.error(f"Payload too large: {size_mb:.1f} MB > {limit_mb:.0f} MB limit")
            return {
                "error": (
                    f"Image payload too large ({size_mb:.1f} MB). "
                    f"Globus Compute limit is {limit_mb:.0f} MB. "
                    "Reduce image size/quality or use fewer images."
                ),
                "error_type": "payload_too_large",
            }

        is_authenticated, auth_message = self.ensure_authenticated()
        if not is_authenticated:
            return {
                "error": auth_message or "Globus Compute authentication required",
                "error_type": "AuthenticationError",
                "auth_required": True,
            }

        logger.info(
            f"Submitting batch inference: model={model} → {hf_name}, "
            f"url={vllm_url}, messages={len(messages)}, max_tokens={max_tokens}"
        )
        t_start = time.perf_counter()

        try:
            gce = self._get_executor()
            t_executor = time.perf_counter()

            future = gce.submit(
                remote_vllm_inference,
                vllm_url,
                hf_name,
                messages,
                temperature,
                max_tokens,
                False,  # stream=False — batch mode
            )
            t_submit = time.perf_counter()

            # asyncio.to_thread() moves the blocking future.result() call to a
            # background thread so the async event loop stays responsive during
            # the GPU inference wait (5-30+ seconds for large models).
            result = await asyncio.to_thread(future.result, timeout=self.task_timeout)
            t_result = time.perf_counter()

            logger.info(
                f"Batch inference timing — "
                f"executor={t_executor - t_start:.2f}s, "
                f"submit={t_submit - t_executor:.2f}s, "
                f"wait={t_result - t_submit:.2f}s, "
                f"total={t_result - t_start:.2f}s"
            )

            if isinstance(result, dict) and "error" in result:
                logger.error(f"Remote inference error: {result['error']}")
                return result

            logger.info("Batch inference completed successfully")
            return result

        except TimeoutError:
            logger.error(f"Globus task timed out after {self.task_timeout}s")
            return {"error": f"Task timeout after {self.task_timeout}s", "error_type": "TimeoutError"}

        except GlobusAPIError as e:
            if e.http_status in (401, 403):
                logger.warning(f"Globus auth error (HTTP {e.http_status}) — resetting executor")
                self._reset_executor()
                return {
                    "error": "Globus Compute session expired. Please re-authenticate.",
                    "error_type": "AuthenticationError",
                    "auth_required": True,
                }
            logger.error(f"Globus API error: {e}")
            return {"error": str(e), "error_type": "GlobusAPIError"}

        except CommandLineLoginFlowEOFError:
            logger.warning("Globus SDK tried interactive auth in non-interactive environment")
            self._reset_executor()
            return {
                "error": "Globus Compute authentication required.",
                "error_type": "AuthenticationError",
                "auth_required": True,
            }

        except (DeserializationError, TaskExecutionFailed) as e:
            # DeserializationError: Python/dill version mismatch between local and endpoint
            # TaskExecutionFailed > ManagerLost: SLURM job expired or compute node crashed
            error_str = str(e)
            error_lower = error_str.lower()

            if "managerlost" in error_lower or "loss of manager" in error_lower:
                last_line = error_str.strip().split("
")[-1].strip().rstrip("-")
                logger.error(f"HPC compute node lost: {last_line}")
                return {
                    "error": (
                        "HPC compute node is unavailable (worker manager lost). "
                        "The SLURM job may have expired. Check squeue on the cluster."
                    ),
                    "error_type": "ManagerLost",
                }

            logger.error(f"Globus deserialization failed: {e}", exc_info=True)
            return {
                "error": (
                    "Result deserialization failed — likely a Python version mismatch "
                    "between local and the HPC endpoint. "
                    "Try: pip install --upgrade globus-compute-sdk"
                ),
                "error_type": "DeserializationError",
            }

        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Unexpected Globus error: {e}", exc_info=True)

            if (
                "unable to open database file" in error_str
                or "login_required" in error_str
                or ("eof" in error_str and "authorization" in error_str)
            ):
                self._reset_executor()
                return {
                    "error": "Globus Compute authentication required.",
                    "error_type": "AuthenticationError",
                    "auth_required": True,
                }

            # Stale connection retry: reset executor and try once more.
            # The _retry flag prevents infinite loops — if this IS the retry, give up.
            if not _retry:
                logger.warning(f"Retrying after unexpected error ({type(e).__name__})...")
                self._reset_executor()
                return await self.submit_inference(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                    _retry=True,
                )

            return {"error": str(e), "error_type": type(e).__name__}

    # =========================================================================
    # STREAMING INFERENCE (via WebSocket relay)
    # =========================================================================

    async def submit_streaming_inference(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str = "",
        relay_url: str = "",
    ) -> dict[str, Any]:
        """
        Submit a streaming inference job. Returns a channel_id immediately.

        The actual tokens flow through the WebSocket relay — connect to
        relay_url/consume/{channel_id} to receive them in real-time.

        Workflow:
            result = await client.submit_streaming_inference(
                messages=messages, model="qwen25-vl-72b",
                relay_url="wss://relay.example.com"
            )
            channel_id = result["channel_id"]
            # Now connect: ws://relay.example.com/consume/{channel_id}
            # Tokens arrive as: {"type": "token", "content": "..."}
            # Stream ends with: {"type": "done", "usage": {...}}

        Args:
            messages: Conversation history in OpenAI format
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            model: Model name as registered in the models dict
            relay_url: WebSocket URL of the relay server

        Returns:
            {"channel_id": "uuid"} on success
            {"error": str, ...} on failure
        """
        import uuid

        hf_name, vllm_url, default_max_tokens = self._resolve_model(model)
        if max_tokens is None:
            max_tokens = default_max_tokens

        if not self.is_available():
            return {"error": "No endpoint configured", "error_type": "ConfigError"}

        if not relay_url:
            return {"error": "relay_url is required for streaming", "error_type": "ConfigError"}

        messages = strip_old_images(messages)

        estimated_size = self._estimate_payload_size(messages)
        if estimated_size > self.max_payload_bytes:
            size_mb = estimated_size / (1024 * 1024)
            limit_mb = self.max_payload_bytes / (1024 * 1024)
            return {
                "error": (
                    f"Image payload too large ({size_mb:.1f} MB). "
                    f"Globus Compute limit is {limit_mb:.0f} MB."
                ),
                "error_type": "payload_too_large",
            }

        is_authenticated, auth_message = self.ensure_authenticated()
        if not is_authenticated:
            return {
                "error": auth_message or "Globus Compute authentication required",
                "error_type": "AuthenticationError",
                "auth_required": True,
            }

        # Generate a UUID for this streaming session.
        # Both producer (HPC) and consumer (your app) use this to join the same relay channel.
        channel_id = str(uuid.uuid4())

        try:
            gce = self._get_executor()

            logger.info(
                f"Submitting streaming inference: model={model} → {hf_name}, "
                f"channel={channel_id[:8]}, relay={relay_url}"
            )

            # Submit the streaming function — returns immediately (~100ms).
            # The remote function will:
            #   1. Connect to relay as producer
            #   2. Call vLLM with stream=True
            #   3. Forward tokens through relay to the consumer
            # We don't wait for Globus to deliver the job's return value.
            gce.submit(
                remote_vllm_streaming,
                vllm_url,
                hf_name,
                messages,
                temperature,
                max_tokens,
                relay_url,
                channel_id,
                self.relay_secret,
                # RELAY_ENCRYPTION_KEY is NOT passed here — the remote function
                # reads it from os.environ on the endpoint. This way the encryption
                # key never travels over Globus Compute's AMQP channel.
            )

            logger.info(f"Streaming job submitted (channel={channel_id[:8]})")
            return {"channel_id": channel_id}

        except Exception as e:
            logger.error(f"Failed to submit streaming job: {e}", exc_info=True)
            error_str = str(e).lower()
            if "unable to open database file" in error_str or "login_required" in error_str:
                self._reset_executor()
                return {
                    "error": "Globus Compute authentication required.",
                    "error_type": "AuthenticationError",
                    "auth_required": True,
                }
            return {"error": str(e), "error_type": type(e).__name__}

    def check_model_health(self, model: str, timeout: int = 20) -> tuple[bool, str | None]:
        """
        Synchronous health check: submits a 1-token inference and waits.

        Returns (True, None) if the model responds, (False, error_msg) otherwise.

        Why sync: health checks are called from both async and sync contexts.
        The 1-token probe is cheap enough to block for (~6s round-trip via Globus).
        """
        if not self.is_available():
            return False, "No endpoint configured"

        is_authenticated, auth_message = self.ensure_authenticated()
        if not is_authenticated:
            return False, auth_message or "Authentication required"

        try:
            hf_name, vllm_url, _ = self._resolve_model(model)
            gce = self._get_executor()

            logger.info(f"[Health] Checking {model} → {hf_name} at {vllm_url} (timeout={timeout}s)")
            t_start = time.perf_counter()

            future = gce.submit(
                remote_vllm_inference,
                vllm_url,
                hf_name,
                [{"role": "user", "content": "hi"}],
                0.0,  # temperature=0 (deterministic, no wasted randomness)
                1,    # max_tokens=1 (just need to know if the model responds)
                False,
            )
            result = future.result(timeout=timeout)

            elapsed = time.perf_counter() - t_start
            logger.info(f"[Health] {model} check completed in {elapsed:.1f}s")

            if isinstance(result, dict) and "error" in result:
                return False, f"Model not responding: {result.get('error', '')[:150]}"

            return True, None

        except TimeoutError:
            return False, f"Model not responding (timed out after {timeout}s)"
        except Exception as e:
            logger.error(f"[Health] {model} check failed: {e}", exc_info=True)
            return False, f"Health check error: {str(e)[:150]}"
