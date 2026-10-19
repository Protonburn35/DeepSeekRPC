import json
from pathlib import Path
import struct

import numpy as np
import pytest
from safetensors.numpy import save_file

from deepseekrpc.inventory import MAX_READ, TensorFile, ownership, parse_header
from deepseekrpc.jsonutil import ValidationError, loads
from deepseekrpc.native_windows import mark_sparse
from deepseekrpc.planner import Geometry, Role, StagePlan


def plan(split=(18, 5, 17)):
    config = json.loads((Path(__file__).parents[1] / "provenance/model-config.json").read_text())
    return StagePlan(Geometry.from_config(config), split)


def test_actual_source_map_and_replica_bytes():
    p = plan()
    assert [p.bounds(r) for r in Role] == [(0, 18), (18, 23), (23, 40)]
    assert {(d.field, d.source_layer, int(d.owner), int(d.recipient)) for d in p.dependencies()} == {
        ("kv", 14, 0, 1), ("index_k", 14, 0, 1), ("topk", 14, 0, 1),
        ("kv", 20, 1, 2), ("index_k", 20, 1, 2), ("topk", 20, 1, 2), ("candidates", 20, 1, 2),
    }
    assert [p.persistent_source_bytes(r) for r in Role] == [2013265920, 2013265920, 1342177280]


def test_alternate_partition_regenerates_dependencies():
    p = plan((20, 4, 16))
    assert not [d for d in p.dependencies() if d.recipient == Role.MIDDLE]
    assert {d.source_layer for d in p.dependencies()} == {20}


@pytest.mark.parametrize("split", [(18, 5, 16), (18, 0, 22), (14, 9, 17), (True, 20, 19), (20, 20)])
def test_bad_split_rejected(split):
    with pytest.raises(ValidationError):
        plan(split)


def test_tensor_selective_read_and_closed_handle(tmp_path):
    path = tmp_path / "fixture.safetensors"
    values = np.arange(32, dtype=np.float16)
    save_file({"layers.18.ffn.experts.0.w1.suh": values, "other": np.ones(4, dtype=np.int64)}, path)
    with TensorFile(path) as source:
        name = "layers.18.ffn.experts.0.w1.suh"
        assert source.read_range(name, 4, 8) == values.tobytes()[4:12]
        assert b"".join(source.chunks(name, 10)) == values.tobytes()
        with pytest.raises(ValidationError):
            source.read_range(name, 63, 2)
        with pytest.raises(ValidationError):
            source.chunks(name, 0).__next__()
    with pytest.raises(ValidationError):
        source.read_range(name, 0, 1)
    assert ownership(name, plan()) == (Role.MIDDLE, "routed_experts")


def test_large_sparse_offsets_without_large_allocation(tmp_path):
    path = tmp_path / "large.safetensors"
    size = (1 << 32) + 17
    header = json.dumps({"huge": {"dtype": "U8", "shape": [size], "data_offsets": [0, size]},
                         "tail": {"dtype": "U8", "shape": [4], "data_offsets": [size, size + 4]}}).encode()
    with path.open("wb") as file:
        mark_sparse(file)
        file.write(struct.pack("<Q", len(header)))
        file.write(header)
        file.seek(size, 1)
        file.write(b"tail")
    with TensorFile(path) as source:
        assert source.read_range("tail", 0, 4) == b"tail"
        with pytest.raises(ValidationError):
            source.read_range("huge", 0, MAX_READ + 1)


@pytest.mark.parametrize("descriptor", [
    {"dtype": "F16", "shape": [2], "data_offsets": [0, 2]},
    {"dtype": "F16", "shape": [-1], "data_offsets": [0, 2]},
    {"dtype": "F16", "shape": [1 << 40, 1 << 40], "data_offsets": [0, 2]},
    {"dtype": "UNKNOWN", "shape": [2], "data_offsets": [0, 2]},
    {"dtype": "F16", "shape": [2], "data_offsets": [0, 4], "extra": True},
])
def test_malformed_tensor_headers(descriptor):
    data = json.dumps({"x": descriptor}).encode()
    with pytest.raises(ValidationError):
        parse_header(data, len(data) + 8 + 4)


def test_payload_overlap_rejected():
    data = json.dumps({k: {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]} for k in ("x", "y")}).encode()
    with pytest.raises(ValidationError, match="overlapping"):
        parse_header(data, len(data) + 12)


@pytest.mark.parametrize("data", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e9999}', b'\xff', b'[' * 2000])
def test_strict_json(data):
    with pytest.raises(ValidationError):
        loads(data, limit=4096)
