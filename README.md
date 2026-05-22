# hpc-gateway

[![PyPI](https://img.shields.io/pypi/v/hpc-gateway)](https://pypi.org/project/hpc-gateway/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/uicacer/hpc-gateway/blob/main/LICENSE)
[![Tests](https://github.com/uicacer/hpc-gateway/actions/workflows/tests.yml/badge.svg)](https://github.com/uicacer/hpc-gateway/actions)

**OpenAI-compatible API gateway for HPC clusters via Globus Compute.**

`hpc-gateway` exposes any vLLM-served model running on an HPC cluster (SLURM, PBS, etc.) as a standard OpenAI-compatible REST API. It handles authentication, rate limiting, payload size management, and real-time token streaming — so your existing OpenAI clients work without modification.

```python
from hpc_gateway.compute import GlobusComputeClient

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
    messages=[{"role": "user", "content": "Explain quantum entanglement."}],
    model="qwen25-vl-72b",
)
```

## Why

HPC clusters run the largest open-source LLMs (72B+ parameters) on GPU hardware that typical cloud users can't afford. But HPC infrastructure has no standard API surface — each cluster has its own SLURM scripts, SSH tunnels, and authentication systems. `hpc-gateway` provides a uniform OpenAI-compatible interface over any vLLM-served model, using [Globus Compute](https://www.globus.org/compute) for authentication and job dispatch (no open ports required on the HPC side).

## Architecture

```
Your App / OpenAI Client
        │  POST /v1/chat/completions
        ▼
  hpc-gateway (FastAPI)
        │  Globus Compute (AMQP — no HPC firewall holes)
        ▼
  HPC Cluster (SLURM)
        │  vLLM HTTP API (internal LAN)
        ▼
  GPU Compute Node
        │  tokens flow via WebSocket relay (streamrelay)
        ▼
  hpc-gateway → SSE stream → Your App
```

Key design points:
- **No open ports on HPC**: Globus Compute is outbound-only from the cluster
- **Real-time streaming**: Tokens stream back via [streamrelay](https://github.com/uicacer/streamrelay) WebSocket relay
- **E2E encryption**: Optional AES-256-GCM encryption between HPC and consumer (relay sees only ciphertext)
- **OpenAI-compatible**: Drop-in for any client using the OpenAI SDK

## Installation

```bash
# Base package (no Globus SDK)
pip install hpc-gateway

# With Globus Compute support
pip install "hpc-gateway[globus]"
```

## Quickstart: Run as a service

Set environment variables and start:

```bash
export GLOBUS_COMPUTE_ENDPOINT_ID="your-endpoint-uuid"
export HPC_MODELS='{"qwen25-vl-72b": {"hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "url": "http://ghi2-002:8000", "context_reserve_output": 4096}}'
export RELAY_URL="wss://relay.example.com"
export RELAY_SECRET="your-relay-secret"

uvicorn hpc_gateway.app:app --host 0.0.0.0 --port 8001
```

The gateway is now reachable at `http://localhost:8001/v1/chat/completions` with the standard OpenAI API schema.

## Embed in an existing FastAPI app

```python
from fastapi import FastAPI
from hpc_gateway.app import router

app = FastAPI()
app.include_router(router, prefix="/hpc")
```

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `GLOBUS_COMPUTE_ENDPOINT_ID` | — | Globus endpoint UUID for the HPC cluster |
| `HPC_MODELS` | `{}` | JSON dict: model name → HPC config |
| `RELAY_URL` | — | WebSocket relay URL for token streaming |
| `RELAY_SECRET` | — | Shared secret for relay auth |
| `RELAY_ENCRYPTION_KEY` | — | AES-256 hex key for E2E encryption |
| `USE_GLOBUS_COMPUTE` | `true` | `false` to route directly via SSH tunnel |
| `LAKESHORE_VLLM_ENDPOINT` | `http://localhost:8000` | Direct vLLM URL (SSH mode) |
| `HPC_PROXY_HOST` | `0.0.0.0` | Bind host |
| `HPC_PROXY_PORT` | `8001` | Bind port |

### HPC_MODELS schema

```json
{
  "my-model-name": {
    "hf_name": "org/ModelName",
    "url": "http://compute-node:8000",
    "context_reserve_output": 4096
  }
}
```

## Authentication

The gateway supports two auth modes (configured in `hpc_gateway/auth.py`):

- **Globus token**: Bearer token from Globus Auth, validated via introspection
- **API key**: Static key from `HPC_API_KEYS` env var (comma-separated)

## Development

```bash
git clone https://github.com/uicacer/hpc-gateway
cd hpc-gateway
uv sync --extra dev
uv run pytest
```

## Related

- [streamrelay](https://github.com/uicacer/streamrelay) — WebSocket relay for real-time token streaming from Globus Compute
- [STREAM](https://github.com/uicacer/stream) — Full tiered LLM routing system that uses hpc-gateway

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

If you use hpc-gateway in research, please cite:

```bibtex
@software{nassar2025hpcgateway,
  author = {Nassar, Anas},
  title  = {hpc-gateway: OpenAI-compatible API gateway for HPC clusters via Globus Compute},
  year   = {2025},
  url    = {https://github.com/uicacer/hpc-gateway}
}
```
