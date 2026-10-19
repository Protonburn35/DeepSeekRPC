import gzip
import json
from pathlib import Path

import pytest

from deepseekrpc.inventory import TensorInfo, ownership
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.planner import Geometry, Role, StagePlan
from deepseekrpc.runtime_memory import GIB, plan_runtime_memory


@pytest.fixture(scope="module")
def physical_metadata():
    root = Path(__file__).parents[1]
    plan = StagePlan(Geometry.from_config(json.loads((root / "provenance/model-config.json").read_text())), (18, 5, 17))
    entries = {role: {} for role in Role}
    with gzip.open(root / "provenance/tensors.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if record["model"] != "pack":
                continue
            info = TensorInfo(record["name"], record["dtype"], tuple(record["shape"]), record["offset"], record["nbytes"])
            entries[ownership(info.name, plan)[0]][info.name] = info
    entries[Role.TERMINAL]["embed.weight"] = entries[Role.HOST]["embed.weight"]
    return plan, entries


def test_whole_pack_admission_includes_all_precisions_and_the_c_embedding_replica(physical_metadata):
    plan, entries = physical_metadata
    reports = []
    for role, capacity in zip(Role, (24, 8, 16)):
        report = plan_runtime_memory(entries[role], plan, role,
                                     available_gpu_bytes=capacity * GIB, available_host_bytes=24 * GIB)
        assert sum(report.gpu.values()) <= capacity * GIB
        assert sum(report.host.values()) <= 24 * GIB
        assert report.expert_slots > 0
        assert report.gpu["permanent_native_weights"] > 0
        reports.append(report)
    assert sum(report.gpu["BF16_source_index_shadow_payload"] for report in reports) == 5 * GIB
    assert sum(report.packed_matrix_count for report in reports) == 47900
    assert sum(report.native_tensor_count for report in reports) == 896  # 895 originals plus C's replica
    assert reports[2].gpu["permanent_native_weights"] > entries[Role.TERMINAL]["embed.weight"].nbytes
    assert reports[1].host["locked_owned_weight_store"] >= 17602641264
    assert reports[1].host["locked_owned_weight_store"] % 4096 == 0
    assert reports[1].ram_tier_bytes == 0


def test_admission_refuses_device_or_physical_host_shortfall_and_missing_experts(physical_metadata):
    plan, entries = physical_metadata
    with pytest.raises(ValidationError, match="device budget"):
        plan_runtime_memory(entries[Role.MIDDLE], plan, Role.MIDDLE,
                            available_gpu_bytes=2 * GIB, available_host_bytes=24 * GIB)
    with pytest.raises(ValidationError, match="physical host"):
        plan_runtime_memory(entries[Role.MIDDLE], plan, Role.MIDDLE,
                            available_gpu_bytes=8 * GIB, available_host_bytes=16 * GIB)
    incomplete = {name: info for name, info in entries[Role.MIDDLE].items()
                  if not name.startswith("layers.18.ffn.experts.0.")}
    with pytest.raises(ValidationError, match="every owned"):
        plan_runtime_memory(incomplete, plan, Role.MIDDLE,
                            available_gpu_bytes=8 * GIB, available_host_bytes=24 * GIB)
