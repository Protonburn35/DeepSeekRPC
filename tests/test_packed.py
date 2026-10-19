import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest
from safetensors.numpy import save_file

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.packed import MUL1_MARKER, PackedLinearLayout
from deepseekrpc.planner import Geometry, Role, StagePlan
from deepseekrpc.weights import SelectiveWeights, ShardPin


def fixture(tmp_path, k, *, multiplier=MUL1_MARKER):
    prefix = "layers.18.ffn.experts.0.w1"
    tensors = {f"{prefix}.trellis": np.zeros((8, 16, 16 * k), np.int16),
               f"{prefix}.suh": np.ones(128, np.float16),
               f"{prefix}.svh": np.ones(256, np.float16),
               f"{prefix}.mul1": np.array(multiplier, np.int32)}
    name = "model-00001-of-00001.safetensors"
    save_file(tensors, tmp_path / name)
    data = (tmp_path / name).read_bytes()
    size, = struct.unpack("<Q", data[:8])
    pin = ShardPin(name, "a" * 40, len(data), hashlib.sha256(data).hexdigest(), size,
                   hashlib.sha256(data[8:8 + size]).hexdigest())
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"weight_map": {n: name for n in tensors}}))
    plan = StagePlan(Geometry.from_config(json.loads((Path(__file__).parents[1] / "provenance/model-config.json").read_text())), (18, 5, 17))
    source = SelectiveWeights(tmp_path, index, index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
                              pins={name: pin}, plan=plan, role=Role.MIDDLE)
    return source, prefix


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8])
def test_actual_pack_k_layouts_include_rotations_and_exact_multiplier(tmp_path, k):
    source, prefix = fixture(tmp_path, k)
    with source:
        layout = PackedLinearLayout.from_source(source, prefix)
        assert (layout.in_features, layout.out_features, layout.k) == (128, 256, k)
        assert layout.nbytes % 256 == 0
        assert {part.info.name.rsplit(".", 1)[1] for part in layout.parts} == {"trellis", "suh", "svh", "mul1"}
        assert all(part.offset % 16 == 0 for part in layout.parts)


def test_marker_value_is_checked_even_when_file_integrity_matches(tmp_path):
    source, prefix = fixture(tmp_path, 2, multiplier=-877912083)
    with source, pytest.raises(ValidationError, match="multiplier marker"):
        PackedLinearLayout.from_source(source, prefix)


def test_non_pack_k_cannot_silently_select_another_kernel(tmp_path):
    source, prefix = fixture(tmp_path, 7)
    with source, pytest.raises(ValidationError, match="physical K"):
        PackedLinearLayout.from_source(source, prefix)
