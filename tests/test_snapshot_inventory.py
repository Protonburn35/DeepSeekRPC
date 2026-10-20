import copy
import json
from pathlib import Path

import pytest

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.planner import Geometry, Role, StagePlan
from deepseekrpc.request_settings import RequestSettings
from deepseekrpc.sampling import RandomStreams
from deepseekrpc.snapshot_state import pool_ranges, state_inventory, validate_timeline


def plan(split=(18, 5, 17)):
    config = json.loads((Path(__file__).parents[1] / "provenance/model-config.json").read_text())
    return StagePlan(Geometry.from_config(config), split)


def metadata(role, *, processed=153):
    return {"abi": 1, "stage": int(role), "identity": "a" * 64, "run": "b" * 64,
            "timeline": {"session": "c" * 32, "request": "d" * 32, "epoch": 7,
                "history": "e" * 64, "encoder_tokens": 129, "processed": processed,
                "pending_token": 11 if processed < 1048576 else None,
                "stop_reason": None if processed < 1048576 else "length", "next_operation": 42},
            "settings": RequestSettings(41017, 1048576).as_dict(), "images": [], "assets": [],
            "prefix": None, "rng": RandomStreams(41017).snapshot() if role == Role.TERMINAL else None,
            "stops": {"pending": "", "tail": "", "prefixes": [], "ended": False} if role == Role.TERMINAL else None}


def test_snapshot_shapes_include_owned_draft_and_replica_rows_at_the_true_processed_cursor():
    p = plan()
    for role in Role:
        value = metadata(role)
        records, settings, images = state_inventory(p, role, value, identity="a" * 64, run="b" * 64)
        assert records[0].shape == (154,)  # One pending output has no KV row.
        assert records[1].shape == (154,)
        for kind, layer, first, count, width in pool_ranges(p, role, 153):
            record, = [record for record in records if record.name == f"{kind}.{layer}"]
            assert record.dtype == "bf16" and record.shape == (count, width)
            if kind == "windows":
                assert (first, count) == (25, 128)
        names = {record.name for record in records}
        assert all((f"windows.{layer}" in names) == (role == Role.TERMINAL) for layer in (40, 41, 42))
        assert "pair.20" not in names  # Ratio 1 has no FP32 open pair.
        assert sum(name.startswith("pair.") for name in names) == (3 if role == Role.HOST else 0)


def test_near_capacity_snapshot_payload_matches_actual_five_gib_cluster_inventory():
    p = plan()
    total = 0
    for role in Role:
        records, _, _ = state_inventory(p, role, metadata(role, processed=1048576), identity="a" * 64, run="b" * 64)
        total += sum(record.nbytes for record in records if record.name.startswith(("sources.", "index_keys.")))
    assert total == 5 * 1024 ** 3


@pytest.mark.parametrize("change", ["identity", "dtype", "timeline", "assets"])
def test_foreign_or_incomplete_snapshot_metadata_fails_before_gpu_allocation(change):
    value = metadata(Role.MIDDLE)
    if change == "identity":
        value["identity"] = "f" * 64
    elif change == "dtype":
        value["cache_dtype"] = "fp16"
    elif change == "timeline":
        value["timeline"]["pending_token"] = None
    else:
        value["assets"] = [{}]
    with pytest.raises(ValidationError):
        state_inventory(plan(), Role.MIDDLE, value, identity="a" * 64, run="b" * 64)


def test_saved_timeline_rejects_boolean_positions_and_over_capacity_pending_inputs():
    value = metadata(Role.HOST)["timeline"]
    value["processed"] = True
    with pytest.raises(ValidationError):
        validate_timeline(value)
    value["processed"] = 1048576
    with pytest.raises(ValidationError, match="pending"):
        validate_timeline(value)
