"""CPU control-flow regressions; these fixtures do not qualify CUDA execution."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from deepseekrpc import bootstrap, runtime_identity, runtime_policy
from deepseekrpc.deployment_config import DeploymentConfig
from deepseekrpc.jsonutil import ValidationError


class BoundaryReached(Exception):
    """Stop before an isolated hardware or model-data boundary."""


def configuration(tmp_path, role):
    examples = Path(__file__).parents[1] / "config"
    shared = json.loads((examples / "cluster.example.json").read_text())
    shared["cluster_id"] = "a" * 32
    shared["workers"]["B"]["address"] = "192.168.50.144"
    local = json.loads((examples / f"{role}.example.json").read_text())
    local.update(model_root=str(tmp_path / "model"), state_root=str(tmp_path / f"state-{role}"))
    if role == "A":
        local["engram_root"] = str(tmp_path / "engram")
    (tmp_path / "cluster.json").write_text(json.dumps(shared))
    path = tmp_path / f"{role}.json"
    path.write_text(json.dumps(local))
    return DeploymentConfig.read(path)


def load(config, **overrides):
    arguments = dict(role=config.role, run_id=config.run, state_root=config.state_root,
                     engram_root=config.engram_root, device=config.device)
    arguments.update(overrides)
    return bootstrap.load_stage(config.model_root, **arguments)


@pytest.mark.parametrize("role", "ABC")
def test_configuration_run_crosses_the_actual_loader_gate(tmp_path, monkeypatch, role):
    config = configuration(tmp_path, role)
    assert len(config.run) == 64
    monkeypatch.setattr(runtime_identity, "require_windows_runtime", lambda *a, **kw: {"component_identity": "fixture"})
    # Isolate the optional native adapter import, not the actual loader gate.
    monkeypatch.setitem(sys.modules, "deepseekrpc.vllm_experts", SimpleNamespace(OwnedExl3Config=object))

    def metadata_boundary(*args, **kwargs):
        raise BoundaryReached("local metadata")

    monkeypatch.setattr(bootstrap, "local_deployment", metadata_boundary)
    with pytest.raises(BoundaryReached, match="local metadata"):
        load(config)


@pytest.mark.parametrize("run", ["a" * 32, "a" * 63, "a" * 65, "g" * 64, None])
def test_loader_refuses_bad_runs_before_runtime_initialization(tmp_path, monkeypatch, run):
    config = configuration(tmp_path, "B")

    def forbidden_runtime(*args, **kwargs):
        raise AssertionError("invalid run reached expensive runtime initialization")

    monkeypatch.setattr(runtime_identity, "require_windows_runtime", forbidden_runtime)
    with pytest.raises(ValidationError, match="worker run"):
        load(config, run_id=run)


@pytest.mark.parametrize("entry", ["load", "configured-self-test"])
def test_nested_gpu_check_retains_the_actual_worker_cache_owner(tmp_path, monkeypatch, entry):
    import torch
    from deepseekrpc import gpu_selftest, host_selftest, triton_policy, worker

    config = configuration(tmp_path, "B")
    config.model_root.mkdir()
    # Keep the real metadata ownership guard. Only hardware observers and the
    # compiler verifier are isolated, stopping before any operation checks.
    monkeypatch.setattr(runtime_policy, "_vllm_cache_root", None)
    monkeypatch.setattr(runtime_policy, "_vllm_cache_environment", None)
    monkeypatch.setenv("VLLM_CACHE_ROOT", "inherited fixture cache")
    monkeypatch.setenv("VLLM_CONFIG_ROOT", "inherited fixture config")
    monkeypatch.setattr(runtime_identity, "sys", SimpleNamespace(platform="win32", prefix=sys.prefix))
    monkeypatch.setattr(triton_policy, "configure_windows_triton", lambda *a, **kw: {})

    def hardware_boundary():
        raise BoundaryReached("identity hardware observers")

    monkeypatch.setattr(runtime_identity, "configure_exact_cuda_math", hardware_boundary)
    actual_identity = runtime_identity.require_windows_runtime
    calls = []

    def observe_identity(*args, **kwargs):
        with pytest.raises(BoundaryReached, match="identity hardware observers"):
            actual_identity(*args, **kwargs)
        calls.append(kwargs.get("cache_root"))
        return {"component_identity": "CPU policy fixture only"}

    monkeypatch.setattr(runtime_identity, "require_windows_runtime", observe_identity)
    monkeypatch.setattr(gpu_selftest, "configure_exact_cuda_math", lambda: None)
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 9))
    monkeypatch.setitem(sys.modules, "deepseekrpc.vllm_experts", SimpleNamespace(OwnedExl3Config=object))
    monkeypatch.setattr(bootstrap, "local_deployment", lambda *a, **kw: ({}, {}, object(), "b" * 64))

    def operation_boundary():
        raise BoundaryReached("native operation checks")

    monkeypatch.setattr(host_selftest, "run_host_checks", operation_boundary)
    with pytest.raises(BoundaryReached, match="native operation checks"):
        if entry == "load":
            load(config)
        else:
            worker._owned_device_preflight(config)
    assert calls == [config.state_root / "kernels/triton", None]
    owner = config.state_root / "kernels/vllm"
    assert Path(os.environ["VLLM_CACHE_ROOT"]) == owner / "cache"
    assert Path(os.environ["VLLM_CONFIG_ROOT"]) == owner / "config"
    observe_identity(config.device)
    with pytest.raises(ValidationError, match="ownership cannot change"):
        actual_identity(config.device, cache_root=tmp_path / "different/kernels/triton")
    assert not (tmp_path / "different").exists()
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "redirected"))
    with pytest.raises(ValidationError, match="policy changed"):
        actual_identity(config.device)


def test_standalone_identity_still_selects_a_private_interpreter_cache(tmp_path, monkeypatch):
    from deepseekrpc import triton_policy
    private = tmp_path / "private-interpreter"
    monkeypatch.setattr(runtime_policy, "_vllm_cache_root", None)
    monkeypatch.setattr(runtime_policy, "_vllm_cache_environment", None)
    monkeypatch.setenv("VLLM_CACHE_ROOT", "inherited fixture cache")
    monkeypatch.setenv("VLLM_CONFIG_ROOT", "inherited fixture config")
    monkeypatch.setattr(runtime_policy, "sys", SimpleNamespace(prefix=str(private), modules=sys.modules))
    monkeypatch.setattr(runtime_identity, "sys", SimpleNamespace(platform="win32", prefix=str(private)))

    def compiler_boundary(cache_root):
        assert cache_root is None
        raise BoundaryReached("standalone compiler observer")

    monkeypatch.setattr(triton_policy, "configure_windows_triton", compiler_boundary)
    for _ in range(2):
        with pytest.raises(BoundaryReached, match="standalone compiler observer"):
            runtime_identity.require_windows_runtime()
    root = private / ".deepseekrpc-vllm"
    assert Path(os.environ["VLLM_CACHE_ROOT"]) == root / "cache"
    assert Path(os.environ["VLLM_CONFIG_ROOT"]) == root / "config"
