"""Tests for GlobusComputeClient — config, model resolution, payload size check."""

import json
import os
import pytest

# GlobusComputeClient imports globus_compute_sdk at module level — mock it before import
from unittest.mock import MagicMock, patch


@pytest.fixture()
def mock_globus_modules(monkeypatch):
    """
    Patch globus_compute_sdk imports so tests run without the Globus SDK installed.
    Only the package-level symbols used in compute.py are stubbed.
    """
    fake_sdk = MagicMock()
    fake_sdk.Executor = MagicMock
    fake_sdk.errors.error_types.DeserializationError = Exception
    fake_sdk.errors.error_types.TaskExecutionFailed = Exception
    fake_sdk.serialize.AllCodeStrategies = MagicMock
    fake_sdk.serialize.ComputeSerializer = MagicMock

    fake_globus_sdk = MagicMock()
    fake_globus_sdk.GlobusAPIError = Exception
    fake_globus_sdk.login_flows.command_line_login_flow_manager.CommandLineLoginFlowEOFError = Exception

    monkeypatch.setitem(__import__("sys").modules, "globus_compute_sdk", fake_sdk)
    monkeypatch.setitem(__import__("sys").modules, "globus_compute_sdk.errors", fake_sdk.errors)
    monkeypatch.setitem(
        __import__("sys").modules,
        "globus_compute_sdk.errors.error_types",
        fake_sdk.errors.error_types,
    )
    monkeypatch.setitem(
        __import__("sys").modules, "globus_compute_sdk.serialize", fake_sdk.serialize
    )
    monkeypatch.setitem(__import__("sys").modules, "globus_sdk", fake_globus_sdk)
    monkeypatch.setitem(
        __import__("sys").modules,
        "globus_sdk.login_flows",
        fake_globus_sdk.login_flows,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "globus_sdk.login_flows.command_line_login_flow_manager",
        fake_globus_sdk.login_flows.command_line_login_flow_manager,
    )
    return fake_sdk


def make_client(mock_globus_modules, **kwargs):
    """Import GlobusComputeClient after mocks are in place and instantiate it."""
    # Force re-import in case a cached version is already in sys.modules
    import importlib
    import sys
    sys.modules.pop("hpc_gateway.compute", None)
    from hpc_gateway.compute import GlobusComputeClient
    return GlobusComputeClient(**kwargs)


# ---------------------------------------------------------------------------
# Constructor / config resolution
# ---------------------------------------------------------------------------

def test_endpoint_from_arg(mock_globus_modules):
    client = make_client(mock_globus_modules, endpoint_id="test-uuid-123")
    assert client.endpoint_id == "test-uuid-123"


def test_endpoint_from_env(mock_globus_modules, monkeypatch):
    monkeypatch.setenv("GLOBUS_COMPUTE_ENDPOINT_ID", "env-uuid-456")
    client = make_client(mock_globus_modules)
    assert client.endpoint_id == "env-uuid-456"


def test_models_from_arg(mock_globus_modules):
    models = {"mymodel": {"hf_name": "org/Model", "url": "http://node:8000"}}
    client = make_client(mock_globus_modules, endpoint_id="x", models=models)
    assert client.models == models


def test_models_from_env(mock_globus_modules, monkeypatch):
    models = {"m1": {"hf_name": "org/M1", "url": "http://node:8000"}}
    monkeypatch.setenv("HPC_MODELS", json.dumps(models))
    client = make_client(mock_globus_modules, endpoint_id="x")
    assert client.models == models


def test_models_invalid_json_env(mock_globus_modules, monkeypatch):
    monkeypatch.setenv("HPC_MODELS", "not-json")
    client = make_client(mock_globus_modules, endpoint_id="x")
    assert client.models == {}


def test_is_available_true(mock_globus_modules):
    client = make_client(mock_globus_modules, endpoint_id="some-id")
    assert client.is_available() is True


def test_is_available_false(mock_globus_modules):
    client = make_client(mock_globus_modules, endpoint_id=None)
    assert client.is_available() is False


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------

def test_resolve_known_model(mock_globus_modules):
    models = {
        "qwen72b": {
            "hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
            "url": "http://ghi2-002:8000",
            "context_reserve_output": 4096,
        }
    }
    client = make_client(mock_globus_modules, endpoint_id="x", models=models)
    hf_name, url, max_tok = client._resolve_model("qwen72b")
    assert hf_name == "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    assert url == "http://ghi2-002:8000"
    assert max_tok == 4096


def test_resolve_unknown_model_uses_name_directly(mock_globus_modules, monkeypatch):
    monkeypatch.setenv("HPC_VLLM_URL", "http://fallback:8000")
    client = make_client(mock_globus_modules, endpoint_id="x", models={})
    hf_name, url, max_tok = client._resolve_model("some/Unknown-Model")
    assert hf_name == "some/Unknown-Model"
    assert url == "http://fallback:8000"
    assert max_tok == 2048


# ---------------------------------------------------------------------------
# _estimate_payload_size
# ---------------------------------------------------------------------------

def test_estimate_payload_size(mock_globus_modules):
    client = make_client(mock_globus_modules, endpoint_id="x")
    messages = [{"role": "user", "content": "Hello world"}]
    size = client._estimate_payload_size(messages)
    assert size > 0
    assert size < 1024  # small message, definitely under 1 KB
