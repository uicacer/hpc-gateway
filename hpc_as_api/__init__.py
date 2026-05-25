"""
hpc-as-api — OpenAI-compatible API gateway for HPC clusters via Globus Compute.

Quick start (programmatic use):
    from hpc_as_api.compute import GlobusComputeClient

    client = GlobusComputeClient(
        endpoint_id="8d978809-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        models={
            "qwen25-vl-72b": {
                "hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
                "url": "http://ghi2-002:8000",
                "context_reserve_output": 4096,
            }
        },
    )
    result = await client.submit_inference(
        messages=[{"role": "user", "content": "Hello!"}],
        model="qwen25-vl-72b",
    )

Quick start (FastAPI service):
    # Set env vars: GLOBUS_COMPUTE_ENDPOINT_ID, HPC_MODELS, RELAY_URL, ...
    # Then run: uvicorn hpc_as_api.app:app --host 0.0.0.0 --port 8001

    # Or embed the router in your existing FastAPI app:
    from hpc_as_api.app import router
    app.include_router(router, prefix="/hpc")
"""

from hpc_as_api.utils import (
    count_images,
    extract_text_content,
    has_images,
    strip_old_images,
)

# GlobusComputeClient depends on globus_compute_sdk and globus_sdk, which are
# optional (hpc-as-api[globus]). Import lazily so the base package works
# without them installed.
try:
    from hpc_as_api.compute import GlobusComputeClient
    _GLOBUS_AVAILABLE = True
except ImportError:
    _GLOBUS_AVAILABLE = False

__version__ = "0.1.0"
__all__ = [
    "GlobusComputeClient",
    "extract_text_content",
    "has_images",
    "count_images",
    "strip_old_images",
]
