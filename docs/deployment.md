# Production Deployment Guide

This guide covers deploying `hpc-gateway` as a production service behind TLS,
with `streamrelay` on the same VM, accessible from any OpenAI-compatible client.

## Prerequisites

- A small public VM (AWS t3.micro or equivalent) — this is the only component needing a public IP
- A Globus Compute endpoint already running on your HPC cluster
- Domain name pointing to your VM (e.g., `hpc-api.institution.edu`)
- Python 3.11+ on the VM

## Architecture

```
Internet (caller)
      │  HTTPS/WSS port 443
      ▼
    Caddy (TLS termination, auto Let's Encrypt)
      ├── / → streamrelay (WebSocket relay, port 8765 internal)
      └── :8001 → hpc-gateway (proxy, port 8002 internal)
                        │  Globus Compute AMQP (outbound only)
                        ▼
                  HPC Cluster
```

## Step 1: Install on the relay VM

```bash
# Install Python 3.11+ and pip
sudo apt update && sudo apt install -y python3.11 python3.11-pip caddy

# Install hpc-gateway with Globus support
python3.11 -m pip install "hpc-gateway[globus]"

# Or install from source
git clone https://github.com/uicacer/hpc-gateway
cd hpc-gateway
python3.11 -m pip install -e ".[globus]"
```

## Step 2: Configure environment

Create `/home/ubuntu/proxy.env` (chmod 600):

```bash
GLOBUS_COMPUTE_ENDPOINT_ID=your-endpoint-uuid-here
HPC_MODELS={"qwen25-vl-72b": {"hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "url": "http://ghi2-002:8000", "context_reserve_output": 4096}}
RELAY_URL=wss://your-relay-domain.com
RELAY_SECRET=<hex string — generate with: python3 -c "import secrets; print(secrets.token_hex(32))">
RELAY_ENCRYPTION_KEY=<hex string — generate with: python3 -c "import secrets; print(secrets.token_hex(32))">
HPC_API_KEY_SERVICE1=sk-your-api-key-here
HPC_PROXY_HOST=127.0.0.1
HPC_PROXY_PORT=8002
```

Set the `RELAY_ENCRYPTION_KEY` in the Globus Compute endpoint's `worker_init` block
(not as a task argument) so it never travels over AMQP.

## Step 3: systemd service

Create `/etc/systemd/system/hpcgateway.service`:

```ini
[Unit]
Description=HPC Gateway
After=network.target streamrelay.service

[Service]
User=ubuntu
EnvironmentFile=/home/ubuntu/proxy.env
ExecStart=/usr/bin/python3.11 -m uvicorn hpc_gateway.app:app \
    --host 127.0.0.1 --port 8002 --log-level info
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hpcgateway
```

## Step 4: Configure Caddy

`/etc/caddy/Caddyfile`:

```caddyfile
your-relay-domain.com {
    reverse_proxy localhost:8765
}

your-relay-domain.com:8001 {
    reverse_proxy localhost:8002
}
```

```bash
sudo systemctl restart caddy
```

## Step 5: Open firewall ports

Open inbound TCP 443 and 8001 in your cloud provider's security group.

## Step 6: Verify

```bash
curl https://your-relay-domain.com:8001/health
# Expected: {"status":"healthy","service":"HPC Gateway",...}
```

## Threat model

| Attack vector | Defense |
|---|---|
| Eavesdropping on caller→proxy | TLS (Caddy, auto Let's Encrypt) |
| Relay operator reading token payloads | AES-256-GCM end-to-end encryption |
| Unauthorized relay connections | Shared secret (post-handshake, not in URL) |
| Unauthorized proxy access | Bearer token (Globus) or API key auth |
| API key exposure in logs | Keys validated in-memory, never logged |
| Globus credentials leaving proxy VM | Credentials stored only in `~/.globus_compute/`; never transmitted |

## Generating keys and secrets

```bash
# Relay shared secret
python3 -c "import secrets; print(secrets.token_hex(32))"

# E2E encryption key (set both on relay VM and in Globus endpoint worker_init)
python3 -c "import secrets; print(secrets.token_hex(32))"

# API key for a calling service
python3 -c "import secrets; print('sk-' + secrets.token_hex(20))"
```

All keys are 64-character hex strings. Hex avoids the `+`, `/`, `=` characters
that cause parsing problems in `.env` files, shell exports, and YAML `worker_init` blocks.
