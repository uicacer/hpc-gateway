"""Tests for hpc_as_api.app — FastAPI routes and health check."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture()
def mock_globus_and_auth(monkeypatch):
    """
    Patch Globus SDK imports and the auth dependency so the FastAPI app loads
    without real Globus credentials or API keys.
    """
    # Stub globus_compute_sdk and globus_sdk
    for mod in [
        "globus_compute_sdk",
        "globus_compute_sdk.errors",
        "globus_compute_sdk.errors.error_types",
        "globus_compute_sdk.serialize",
        "globus_sdk",
        "globus_sdk.login_flows",
        "globus_sdk.login_flows.command_line_login_flow_manager",
    ]:
        monkeypatch.setitem(__import__("sys").modules, mod, MagicMock())

    # Evict any previously imported hpc_as_api modules so patches take effect
    import sys
    for key in list(sys.modules.keys()):
        if key.startswith("hpc_as_api"):
            sys.modules.pop(key)


@pytest.fixture()
def client(mock_globus_and_auth, monkeypatch):
    """Return a TestClient with a real (no-Globus) app instance."""
    monkeypatch.setenv("USE_GLOBUS_COMPUTE", "false")
    monkeypatch.setenv("HPC_MODELS", json.dumps({
        "test-model": {"hf_name": "org/TestModel", "url": "http://fake:8000"}
    }))

    from hpc_as_api.app import app
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["service"] == "HPC Gateway"


def test_health_lists_models(client):
    resp = client.get("/health")
    assert "test-model" in resp.json()["models"]


# ---------------------------------------------------------------------------
# /v1/models — requires auth
# ---------------------------------------------------------------------------

def test_models_requires_auth(client):
    resp = client.get("/v1/models")
    # Should return 401/403, not 200
    assert resp.status_code in (401, 403, 422)
