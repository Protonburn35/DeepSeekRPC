import gzip
import json
from pathlib import Path

import pytest

from deepseekrpc.dense_weights import DenseBinding, layer_bindings, stage_bindings
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.inventory import ownership
from deepseekrpc.planner import Geometry, Role, StagePlan


def test_every_real_pack_non_routed_matrix_has_an_explicit_module_binding():
    with gzip.open(Path(__file__).parents[1] / "provenance/tensors.jsonl.gz", "rt") as stream:
        prefixes = {record["name"].removesuffix(".trellis") for line in stream
                    if (record := json.loads(line))["name"].endswith(".trellis")
                    and ".ffn.experts." not in record["name"]}
    bindings = {"lm_head": DenseBinding(("head",), output_fp32=True)}
    for layer in range(40):
        bindings.update(layer_bindings(prefixes, source_layer=f"layers.{layer}", module_layer=f"model.layers.{layer}"))
    for layer in range(3):
        bindings.update(layer_bindings(prefixes, source_layer=f"mtp.{layer}", module_layer=f"draft.{layer}"))
    consumed = [prefix for binding in bindings.values() for prefix in binding.prefixes]
    assert len(consumed) == len(set(consumed)) == 668
    assert set(consumed) == prefixes
    for layer in range(40):
        binding = bindings[f"model.layers.{layer}.attn.wo_a"]
        assert binding.mode == "grouped" and len(binding.prefixes) == 8
    assert bindings["model.layers.20.attn.compressor.fused_wkv_wgate"].output_fp32
    assert len(bindings["model.layers.20.attn.compressor.fused_wkv_wgate"].prefixes) == 1
    assert len(bindings["model.layers.2.attn.compressor.fused_wkv_wgate"].prefixes) == 2
    plan = StagePlan(Geometry.from_config(json.loads(
        (Path(__file__).parents[1] / "provenance/model-config.json").read_text())), (18, 5, 17))
    for role in Role:
        owned = {prefix for prefix in prefixes if ownership(prefix + ".trellis", plan)[0] == role}
        actual = stage_bindings(plan, role, owned)
        assert {p for binding in actual.values() for p in binding.prefixes} == owned
        if role == Role.TERMINAL:
            assert actual["model.mtp.0.main_proj"].prefixes == ("mtp.0.main_proj",)
            assert actual["model.layers.40.attn.fused_wqa_wkv"].prefixes == ("mtp.0.attn.wq_a", "mtp.0.attn.wkv")


def test_unknown_or_incomplete_physical_projection_never_becomes_a_dense_fallback():
    with pytest.raises(ValidationError, match="incomplete"):
        layer_bindings({"layers.0.attn.wq_a"}, source_layer="layers.0", module_layer="model.layers.0")
