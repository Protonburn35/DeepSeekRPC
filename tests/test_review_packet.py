"""Evidence-metadata fixtures only; no Windows runtime is simulated here."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("review_packet_tool", Path(__file__).parents[1] / "tools/build_review_packet.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def evidence_fixture(tmp_path):
    commit = "a" * 40
    cpu_path = tmp_path / "cpu.json"
    cpu_path.write_text(json.dumps({"source_commit": commit, "worktree_clean": True, "platform": "Windows-metadata-fixture",
        "checks": [{"name": name, "exit_code": 0} for name in ("pytest", "workflow-syntax", "wheel", "install-wheel", "installed-cli")]}))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    reports = {name: {"scope": "synthetic report metadata fixture"} for name in
        ("evidence/vllm.json", "evidence/exllama.json", "evidence/native.json", "evidence/plugin.json",
         "evidence/installed-runtime/parser.json")}
    reports["evidence/installed-runtime/runtime.json"] = {"source_commit": commit, "python": "3.12.14",
        "CUDA_execution": "not run", "full_model_validation": "not run",
        "checks": [{"name": name, "status": "passed"} for name in ("versions", "application_source_identity",
            "private_compilers_without_sdk", "native_without_sdk", "complete_python_integration", "actual_original_parser")]}
    manifest = {"kind": "runtime", "source_commit": commit, "architectures": ["sm_89", "sm_120"],
        "installed_qualification": {"status": "passed"}, "GPU_execution": "not run", "full_model_validation": "not run", "files": {}}
    for name, report in reports.items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(report).encode()
        path.write_bytes(data)
        manifest["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (bundle / "BUNDLE.json").write_text(json.dumps(manifest))
    return bundle, cpu_path, commit


def test_review_gate_requires_matching_commit_in_both_windows_reports(tmp_path):
    bundle, cpu, commit = evidence_fixture(tmp_path)
    assert review.validated_evidence(bundle, cpu, commit)[0]["source_commit"] == commit
    with pytest.raises(ValueError, match="exact clean source"):
        review.validated_evidence(bundle, cpu, "b" * 40)


def test_review_gate_refuses_failed_checks_and_unsealed_installation(tmp_path):
    bundle, cpu, commit = evidence_fixture(tmp_path)
    record = json.loads(cpu.read_text())
    record["checks"][0]["exit_code"] = 1
    cpu.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Windows CPU evidence"):
        review.validated_evidence(bundle, cpu, commit)
    record["checks"][0]["exit_code"] = 0
    cpu.write_text(json.dumps(record))
    manifest = json.loads((bundle / "BUNDLE.json").read_text())
    del manifest["installed_qualification"]
    (bundle / "BUNDLE.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="installation qualification"):
        review.validated_evidence(bundle, cpu, commit)


def test_review_gate_refuses_report_tampering_before_copying_any_artifact(tmp_path):
    bundle, cpu, commit = evidence_fixture(tmp_path)
    path = bundle / "evidence/installed-runtime/runtime.json"
    path.write_bytes(path.read_bytes().replace(b'"passed"', b'"failed"', 1))
    with pytest.raises(ValueError, match="sealed bundle manifest"):
        review.validated_evidence(bundle, cpu, commit)
