---
title: 'hpc-as-api: An OpenAI-Compatible API Gateway for HPC Clusters via Globus Compute'
tags:
  - Python
  - HPC
  - API gateway
  - Globus Compute
  - LLM
  - OpenAI
  - SLURM
  - vLLM
  - federated authentication
  - scientific computing
authors:
  - name: Anas Nassar
    orcid: 0009-0008-4225-5745
    corresponding: true
    affiliation: 1
affiliations:
  - name: Advanced Cyberinfrastructure for Education and Research (ACER), University of Illinois Chicago, USA
    index: 1
date: 2026-05-22
bibliography: paper.bib
---

# Summary

HPC clusters run the largest open-source AI models and scientific simulation codes
available — but they expose no standard API surface. Each cluster has its own SLURM
scripts, SSH tunnels, authentication systems, and job submission conventions.
Researchers and developers who want to call an HPC-hosted model or service must
navigate these heterogeneous interfaces directly, requiring HPC expertise that most
application developers do not have.

`hpc-as-api` solves this by wrapping HPC resources behind a standard
OpenAI-compatible REST API. It accepts `POST /v1/chat/completions` requests
(the same format used by OpenAI, Anthropic, and every modern LLM client library),
dispatches them to the HPC cluster via Globus Compute [@globuscompute2024], and
streams tokens back in real time through the `streamrelay` WebSocket relay
[@nassar2026streamrelay]. From the caller's perspective, invoking a 72-billion
parameter model on an H100 cluster is identical to calling any cloud provider:

```python
import openai
client = openai.OpenAI(base_url="https://hpc-api.institution.edu/v1", api_key="sk-xxxx")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
    messages=[{"role": "user", "content": "Explain transformer attention."}],
    stream=True,
)
```

The gateway handles authentication, rate limiting, payload size management, and
end-to-end encryption transparently. The HPC cluster requires no public IP, no
open inbound ports, and no changes to its scheduler or firewall configuration.

# Statement of Need

## The accessibility gap in HPC resources

Institutional HPC clusters provide GPU and CPU resources unavailable on personal
hardware at no marginal cost to researchers. The largest open-source models
(70B+ parameters) can only be run practically on this hardware. Yet these
resources are structurally inaccessible to the majority of developers and
researchers:

- **HPC expertise barrier**: Accessing a cluster requires knowledge of SLURM,
  SSH key management, module systems, batch scripts, and cluster-specific
  conventions. Application developers building LLM-powered tools do not
  typically have this expertise.
- **No standard interface**: Each HPC center exposes resources differently.
  Code written for one cluster does not transfer to another.
- **No streaming**: Standard HPC job dispatch returns a single result when a
  job completes. Applications that need incremental output — chat interfaces
  seeing the first token, monitoring dashboards, real-time data pipelines —
  are incompatible with the batch execution model.

Several HPC centers have deployed LLM inference services [@first2025; @dartmouth2025;
@purdue2025; @chatai2024], but a recurring limitation is the absence of
streaming and the requirement for HPC accounts. Users must interact through
cluster-specific interfaces rather than standard tools.

## What hpc-as-api provides

`hpc-as-api` closes this gap by providing:

1. **A standard API surface**: Any OpenAI-compatible client (LangChain,
   LlamaIndex, OpenWebUI, Cursor, AWS Amplify) works without modification.

2. **No HPC knowledge required by the caller**: The caller sends an HTTP POST
   with a bearer token. The gateway handles Globus authentication, job
   dispatch, relay connection, and SSE streaming internally.

3. **Dual-mode authentication**: Globus Token Auth for direct university users
   (per-user SLURM attribution via email domain mapping); pre-issued API keys
   for external service callers (e.g., an AWS backend authenticating its own
   users via Cognito). Both modes coexist on the same endpoint.

4. **Real-time streaming**: Integration with `streamrelay` [@nassar2026streamrelay]
   provides sub-second time-to-first-token from HPC via the dual-channel
   WebSocket relay architecture. A batch fallback handles relay-unavailable
   scenarios.

5. **End-to-end encryption**: Optional AES-256-GCM encryption between the HPC
   node and the gateway consumer, so the relay operator never sees plaintext
   token payloads even if the relay VM is compromised.

6. **Embeddable router**: The FastAPI `router` object can be mounted in any
   existing application (`app.include_router(router, prefix="/hpc")`), enabling
   integration without running a separate process.

# Design and Implementation

## Architecture

`hpc-as-api` is built around three separation-of-concerns principles:

**Control plane stays unchanged.** Job authentication, dispatch, and scheduling
continue to use Globus Compute [@globuscompute2024]. The gateway adds no new
dependencies to the HPC cluster — it reuses the Globus Compute endpoint already
required for job submission.

**Data plane via streamrelay.** Token streaming uses the dual-channel WebSocket
relay architecture from `streamrelay` [@nassar2026streamrelay]. The HPC compute
node connects outbound to the relay as producer; the gateway connects as consumer.
Neither side opens an inbound port.

**Configuration via environment.** All settings (endpoint ID, model registry,
relay URL, secrets) come from environment variables. There are no config files
to manage. The model registry (`HPC_MODELS` env var, JSON) maps caller-visible
model names to HuggingFace names and vLLM URLs:

```bash
export HPC_MODELS='{"qwen25-vl-72b": {
  "hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
  "url": "http://ghi2-002:8000",
  "context_reserve_output": 4096
}}'
```

## Key components

**`GlobusComputeClient`** (`hpc_as_api/compute.py`): Manages the persistent
Globus Compute Executor (AMQP connection reuse saves 1–2 s per request),
handles authentication checks and credential reload, manages payload size
(stripping images from older conversation history to stay under Globus's 10 MB
limit), and submits both batch and streaming inference jobs. Remote functions
are defined as source strings and compiled via `exec()` to produce clean
bytecode — a workaround for PyInstaller-bundled environments where standard
serialization fails with missing internal modules.

**`authenticate` / `validate_messages`** (`hpc_as_api/auth.py`): FastAPI
dependency that validates every request. Accepts either a Globus access token
(introspected against Globus Auth's public endpoint, email domain checked) or
a pre-issued API key (constant-time comparison). Input validation enforces
message count limits, role constraints, and content length bounds before any
job reaches the cluster. Per-caller rate limiting uses a sliding window
(default 20 requests/60 s).

**`decrypt_message`** (`hpc_as_api/crypto.py`): AES-256-GCM decryption for
end-to-end encrypted relay payloads. The encryption key is set as an environment
variable on the Globus Compute endpoint (`worker_init`) and never transmitted
as a task argument, so it does not travel over Globus's AMQP channel.

**Message utilities** (`hpc_as_api/utils.py`): Handles multimodal OpenAI
messages (text + base64 images). `strip_old_images()` removes image content
from all but the latest user message before Globus submission, reducing payload
size while preserving text context for the model.

## Deployment

The recommended deployment places the gateway on the same VM as the
`streamrelay` relay server (lower latency, single TLS certificate, one security
perimeter). Caddy handles TLS automatically. The full deployment requires:

- One small public VM (e.g., AWS t3.micro) running `streamrelay` and `hpc-as-api`
- One Globus Compute endpoint on the HPC cluster (outbound AMQP only)
- One pre-issued API key per external calling service

Complete deployment instructions and a threat model covering all five attack
surfaces (Globus AMQP, relay channel, proxy endpoint, API key storage, TLS
termination) are provided in `docs/deployment.md`.

# Performance

Deployed in the STREAM system [@nassar2026stream] at the University of Illinois
Chicago against a Qwen 2.5 72B AWQ model on an NVIDIA H100 NVL GPU on the
Lakeshore HPC cluster (50-run medians):

| Metric | Value |
|---|---|
| Median time-to-first-token (relay streaming, steady state) | **0.60 s** ± 0.20 s |
| Median end-to-end latency (batch, warm cache) | **11.17 s** ± 2.0 s |
| Throughput | ~25 tok/s (vLLM with `--enforce-eager` on CUDA 12.4) |

The 0.60 s TTFT includes Globus Compute authentication and job dispatch on a
dedicated single-user endpoint. The relay itself adds no measurable per-message
overhead — it is a memory-copy forwarder with no parsing on message content.

# Acknowledgements

`hpc-as-api` was developed as part of the STREAM project at the Advanced
Cyberinfrastructure for Education and Research (ACER) group at the University
of Illinois Chicago. We thank Marius Horga (Assistant Director of Advanced
Platforms for Research, ACER) for support of this work, and the UIC ACER team
for providing and maintaining the Lakeshore HPC cluster used in development
and evaluation.

# AI Usage Disclosure

Claude Code (Anthropic, claude-sonnet-4-6) was used to assist with: code
generation and refactoring (`GlobusComputeClient`, FastAPI routes, authentication
module, encryption module, test scaffolding), documentation drafting (README,
deployment guide, CONTRIBUTING), and paper text editing. All architectural
decisions — the configuration-injectable client design, dual-mode authentication
architecture, end-to-end encryption key separation (endpoint env var vs. task
argument), embeddable FastAPI router pattern, and multimodal payload management
strategy — are the author's original work. All AI-assisted outputs were reviewed,
validated, and revised by the author. The author takes full responsibility for
the accuracy, correctness, and integrity of all submitted materials.

# References
