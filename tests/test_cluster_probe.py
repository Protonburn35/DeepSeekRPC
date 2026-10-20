from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from deepseekrpc.cluster_probe import sampler_checks, validate_replicas
from deepseekrpc.coordinator import ClusterCoordinator
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.planner import Geometry, Role, StagePlan
from deepseekrpc.request_settings import RequestSettings


def test_shipped_sampler_microdiagnostics_execute_real_verifier_equations():
    assert sampler_checks() == {"checks": 13, "status": "passed",
        "scope": "independent tiny probability/branch equations; no model execution"}


def test_speculation_gate_defaults_closed_and_expires_on_mesh_or_source_change():
    # Metadata-only gate fixture; no StageService/model is emulated or run.
    coordinator = object.__new__(ClusterCoordinator)
    stage = SimpleNamespace(identity="a" * 64)
    coordinator.host = SimpleNamespace(execution=SimpleNamespace(stage=stage), mesh_nonce="b" * 32)
    coordinator._speculation_proof = coordinator._qualification_thread = None
    assert not RequestSettings(1, 32).speculative
    settings = RequestSettings(1, 32, speculative=True)
    with pytest.raises(ValidationError, match="selftest-cluster"):
        coordinator.require_speculation(settings)
    coordinator._speculation_proof = ((stage.identity, coordinator.host.mesh_nonce), "c" * 64)
    coordinator.require_speculation(settings)
    coordinator.host.mesh_nonce = "d" * 32
    with pytest.raises(ValidationError):
        coordinator.require_speculation(settings)
    coordinator._qualification_thread = threading.get_ident()
    coordinator.require_speculation(settings)
    coordinator._qualification_thread = None
    stage.identity = "e" * 64
    assert not coordinator.speculation_status()["qualified"]


def test_probe_receipt_validator_refuses_replica_or_timeline_disagreement():
    # Pure receipt validation fixtures, explicitly not distributed GPU evidence.
    plan = StagePlan(Geometry.from_config(json.loads((Path(__file__).parents[1] / "provenance/model-config.json").read_text())), (18, 5, 17))
    peers = {role: {"stage": int(role), "identity": "a" * 64, "mesh": "b" * 32,
        "history": "c" * 64, "processed": 256, "pending_token": 9, "pools": {},
        "locked_weight_pages": {"pages_checked": 1} if role == Role.MIDDLE else None} for role in Role}
    for dependency in plan.dependencies():
        if dependency.field not in {"kv", "index_k"}:
            continue
        key = f"{'sources' if dependency.field == 'kv' else 'index_keys'}.{dependency.source_layer}"
        for role in (dependency.owner, dependency.recipient):
            peers[role]["pools"][key] = {"sha256": "d" * 64}
    validate_replicas(peers, plan)
    altered = deepcopy(peers)
    altered[Role.MIDDLE]["processed"] += 1
    with pytest.raises(ValidationError, match="timeline"):
        validate_replicas(altered, plan)
    altered = deepcopy(peers)
    dependency = next(item for item in plan.dependencies() if item.field == "kv")
    altered[dependency.recipient]["pools"][f"sources.{dependency.source_layer}"]["sha256"] = "e" * 64
    with pytest.raises(ValidationError, match="shadow"):
        validate_replicas(altered, plan)
