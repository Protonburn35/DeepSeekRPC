import json
import socket
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from deepseekrpc.api_runtime import APIRuntime
from deepseekrpc.deployment_config import DeploymentConfig
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.planner import Role
from deepseekrpc.worker import APIServer, own_state


def files(tmp_path):
    shared = {"format": 1, "cluster_id": "c" * 32, "split": [18, 5, 17], "context": 1048576,
              "kv_dtype": "bf16", "index_dtype": "bf16", "layer_stage_wire": "fp32", "replay": "exact",
              "workers": {"B": {"address": "127.0.0.1", "port": 39011},
                          "C": {"address": "127.0.0.1", "port": 39012}},
              "api": {"address": "127.0.0.1", "port": 8082}}
    local = {"format": 1, "stage": "A", "shared": "cluster.json", "model_root": "model",
             "engram_root": "engram", "state_root": "state-A",
             "tls": {"ca": "tls/ca.pem", "certificate": "tls/A.pem", "private_key": "tls/A.key"}}
    (tmp_path / "cluster.json").write_text(json.dumps(shared))
    (tmp_path / "A.json").write_text(json.dumps(local))
    return shared, local


def test_shared_identity_ignores_local_paths_and_gpu_but_binds_precision_and_ports(tmp_path):
    shared, local = files(tmp_path)
    a = DeploymentConfig.read(tmp_path / "A.json")
    local.update(stage="C", model_root="another-model", state_root="state-C", device=1)
    del local["engram_root"]
    (tmp_path / "C.json").write_text(json.dumps(local))
    c = DeploymentConfig.read(tmp_path / "C.json")
    assert a.role == Role.HOST and c.role == Role.TERMINAL and a.run == c.run
    assert "another-model" not in json.dumps(c.effective())
    shared["api"]["port"] = 8081
    (tmp_path / "cluster.json").write_text(json.dumps(shared))
    assert DeploymentConfig.read(tmp_path / "A.json").run != a.run


@pytest.mark.parametrize("field,value", [("kv_dtype", "fp16"), ("index_dtype", "int8"),
    ("layer_stage_wire", "bf16"), ("replay", "bounded"), ("context", True), ("split", [14, 9, 17])])
def test_precision_and_ownership_choices_are_not_silently_translated(tmp_path, field, value):
    shared, _ = files(tmp_path)
    shared[field] = value
    (tmp_path / "cluster.json").write_text(json.dumps(shared))
    with pytest.raises(ValidationError):
        DeploymentConfig.read(tmp_path / "A.json")


def test_state_cannot_alias_model_and_unresolved_addresses_refuse_startup(tmp_path):
    shared, local = files(tmp_path)
    local["state_root"] = "model/cache"
    (tmp_path / "A.json").write_text(json.dumps(local))
    with pytest.raises(ValidationError, match="separate directory"):
        DeploymentConfig.read(tmp_path / "A.json")
    local["state_root"] = "state-A"
    (tmp_path / "A.json").write_text(json.dumps(local))
    shared["workers"]["B"]["address"] = "REPLACE_LAPTOP_ADDRESS"
    (tmp_path / "cluster.json").write_text(json.dumps(shared))
    with pytest.raises(ValidationError, match="placeholder"):
        DeploymentConfig.read(tmp_path / "A.json")


def test_runtime_directory_has_real_exclusive_ownership_and_preserves_other_files(tmp_path):
    state = tmp_path / "state"
    with own_state(state):
        with pytest.raises(RuntimeError, match="already owns"):
            with own_state(state):
                raise AssertionError("second owner admitted")
    with own_state(state):
        pass
    unrelated = tmp_path / "existing"
    unrelated.mkdir()
    (unrelated / "do-not-touch").write_bytes(b"preserve")
    with pytest.raises(ValidationError, match="dedicated empty"):
        with own_state(unrelated):
            pass
    assert (unrelated / "do-not-touch").read_bytes() == b"preserve"


@pytest.mark.parametrize("failed", [False, True])
def test_device_diagnostics_claim_state_before_cache_writes_and_release_afterward(tmp_path, monkeypatch, failed):
    from deepseekrpc import worker
    config = SimpleNamespace(state_root=tmp_path / "fresh-state")

    def lifetime_observer(config, state):
        # An I/O/ownership fixture only; no device/model result is fabricated.
        assert (state / ".owner").is_file()
        with pytest.raises(RuntimeError, match="already owns"):
            with own_state(state):
                raise AssertionError("concurrent stage owner admitted during diagnostics")
        (state / "kernels/triton").mkdir(parents=True)
        if failed:
            raise OSError("synthetic diagnostic lifetime interruption")
        return {"scope": "ownership fixture only"}

    monkeypatch.setattr(worker, "_device_preflight", lifetime_observer)
    if failed:
        with pytest.raises(OSError, match="lifetime interruption"):
            worker._owned_device_preflight(config)
    else:
        assert worker._owned_device_preflight(config)["scope"] == "ownership fixture only"
    # Normal startup can claim the same real directory despite the JIT cache.
    with own_state(config.state_root):
        assert (config.state_root / "kernels/triton").is_dir()


def test_actual_background_http_listener_reports_loading_without_a_model():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = sock.getsockname()
    server = APIServer(APIRuntime(), endpoint).start()
    try:
        assert server.alive
        with urllib.request.urlopen(f"http://127.0.0.1:{endpoint[1]}/v1/models", timeout=2) as response:
            assert json.loads(response.read())["data"][0]["max_model_len"] == 1048576
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"http://127.0.0.1:{endpoint[1]}/readyz", timeout=2)
        assert error.value.code == 503
    finally:
        server.close()
