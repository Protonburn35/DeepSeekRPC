"""Header-derived whole-stage admission arithmetic before any model allocation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .dense_weights import stage_bindings
from .inventory import MAX_READ, TensorInfo, ownership
from .jsonutil import ValidationError, integer
from .packed import PACK_K
from .planner import Role, StagePlan

GIB = 1024 ** 3
_EXPERT = re.compile(r"(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\Z")


def native_element_bytes(name: str):
    """Actual native parameter storage chosen by OwnedStageGraph/vLLM."""
    if (name in {"image_start", "image_end", "image_newline"} or ".ffn.gate." in name or
            ".hc_" in name or name.endswith(".attn_sink") or ".confidence_head." in name):
        return 4
    return 2


@dataclass(frozen=True)
class PackedFootprint:
    inputs: int
    outputs: int
    nbytes: int


def packed_footprint(prefix, entries):
    try:
        trellis, suh, svh, marker = (entries[f"{prefix}.{suffix}"] for suffix in ("trellis", "suh", "svh", "mul1"))
    except KeyError as exc:
        raise ValidationError("admission encountered an incomplete packed linear") from exc
    if (trellis.dtype != "I16" or len(trellis.shape) != 3 or
            suh.dtype != "F16" or len(suh.shape) != 1 or svh.dtype != "F16" or len(svh.shape) != 1 or
            marker.dtype != "I32" or marker.shape != () or marker.nbytes != 4):
        raise ValidationError("packed admission physical dtype/shape mismatch")
    inputs, outputs = suh.shape[0], svh.shape[0]
    k, remainder = divmod(trellis.shape[-1], 16)
    if (inputs <= 0 or outputs <= 0 or inputs % 128 or outputs % 128 or k not in PACK_K or remainder or
            trellis.shape[:2] != (inputs // 16, outputs // 16)):
        raise ValidationError("packed admission dimensions or codebook geometry are unsupported")
    size = 0
    for item, element_bytes in ((trellis, 2), (suh, 2), (svh, 2), (marker, 4)):
        if item.nbytes != math.prod(item.shape) * element_bytes:
            raise ValidationError("packed header byte count disagrees with its physical shape")
        size = (size + 15) // 16 * 16 + item.nbytes
    return PackedFootprint(inputs, outputs, (size + 255) // 256 * 256)


@dataclass(frozen=True)
class RuntimeMemoryPlan:
    role: Role
    gpu: dict[str, int]
    host: dict[str, int]
    expert_slots: int
    expert_slot_bytes: int
    native_tensor_count: int
    packed_matrix_count: int
    ram_tier_bytes: int

    def as_dict(self):
        return {"stage": "ABC"[int(self.role)], "cache_dtype": "bfloat16", "layer_stage_wire": "fp32",
                "gpu_bytes": self.gpu, "host_bytes": self.host, "expert_slots": self.expert_slots,
                "expert_slot_bytes": self.expert_slot_bytes, "native_tensor_count": self.native_tensor_count,
                "packed_matrix_count": self.packed_matrix_count, "ram_tier_bytes": self.ram_tier_bytes,
                "qualification": "prospective admission; not physical residency or measured peak proof"}


def plan_runtime_memory(entries: dict[str, TensorInfo], plan: StagePlan, role: Role, *,
                        available_gpu_bytes: int, available_host_bytes: int,
                        requested_ram_tier_bytes: int = 8 * GIB):
    """Include all owned tensors and C's one explicit DSpark embedding replica.

    Available bytes must come from current OS/device budget queries for actual
    admission. Metadata-only callers may use this as a prospective calculation.
    Source payload verification, runtime dtype checks, eager allocation and
    subsequent budget monitoring remain independent mandatory steps.
    """
    integer(available_gpu_bytes, "available device bytes", 1, 1 << 40)
    integer(available_host_bytes, "available physical host bytes", 1, 1 << 40)
    integer(requested_ram_tier_bytes, "requested RAM tier", 0, 1 << 40)
    if not isinstance(role, Role) or not entries or len(entries) > 200000:
        raise ValidationError("runtime admission needs an explicit stage and complete owned tensor metadata")
    for name, info in entries.items():
        if (not isinstance(info, TensorInfo) or name != info.name or
                ownership(name, plan)[0] != role and not (role == Role.TERMINAL and name == "embed.weight")):
            raise ValidationError("runtime header plan contains an unowned tensor")
    if role == Role.TERMINAL and "embed.weight" not in entries:
        raise ValidationError("complete stage C admission must include its explicit DSpark embedding replica")
    prefixes = {name.removesuffix(".trellis") for name in entries if name.endswith(".trellis")}
    footprints = {prefix: packed_footprint(prefix, entries) for prefix in prefixes}
    dense = stage_bindings(plan, role, prefixes)
    permanent = {prefix for binding in dense.values() for prefix in binding.prefixes}
    routed = {}
    for prefix, value in footprints.items():
        if prefix in permanent:
            continue
        match = _EXPERT.fullmatch(prefix)
        if match is None:
            raise ValidationError("runtime admission has an unknown routed projection")
        routed.setdefault(match.groups()[:3], {})[match.group(4)] = value
    if not routed:
        raise ValidationError("stage has no complete routed expert catalogue")
    first_layer, last_layer = plan.bounds(role)
    expected = {("layers", str(layer), str(expert)) for layer in range(first_layer, last_layer)
                for expert in range(384)}
    if role == Role.TERMINAL:
        expected.update(("mtp", str(layer), str(expert)) for layer in range(3) for expert in range(128))
    if set(routed) != expected:
        raise ValidationError("routed expert admission does not cover every owned model and draft expert")
    slot_bytes = 0
    for matrices in routed.values():
        if set(matrices) != {"w1", "w2", "w3"}:
            raise ValidationError("expert admission is missing a gate/up/down projection")
        gate, down, up = (matrices[key] for key in ("w1", "w2", "w3"))
        if (gate.inputs != plan.geometry.hidden or gate.outputs != 2304 or
                (gate.inputs, gate.outputs) != (up.inputs, up.outputs) or
                (down.inputs, down.outputs) != (gate.outputs, gate.inputs)):
            raise ValidationError("expert admission geometry does not form a complete model bundle")
        slot_bytes = max(slot_bytes, sum(item.nbytes for item in matrices.values()))
    packed_names = {f"{prefix}.{suffix}" for prefix in prefixes for suffix in ("trellis", "suh", "svh", "mul1")}
    native = [info for name, info in entries.items() if name not in packed_names]
    if any(info.dtype not in {"BF16", "F16", "F32"} for info in native):
        raise ValidationError("an unsupported native tensor cannot enter the stage admission")
    native_bytes = sum(math.prod(info.shape) * native_element_bytes(info.name) for info in native)
    begin, end = plan.bounds(role)
    windows = end - begin + (3 if role == Role.TERMINAL else 0)
    sources = {source for source in plan.geometry.kv_sources if plan.owner(source) == role}
    sources.update(d.source_layer for d in plan.dependencies() if d.recipient == role and d.field == "kv")
    owned_sources = {source for source in sources if plan.owner(source) == role}
    payload, pool_other, host_maps = 0, 0, 0
    for source in sources:
        slots = (plan.geometry.context + plan.geometry.compress_ratios[source] - 1) // plan.geometry.compress_ratios[source]
        for width in (512, 128):
            payload += slots * width * 2
            pool_other += slots * 8 + 64 * (width * 2 + 256) + 64 * (width * 2 + 32)
            host_maps += slots * 16
    pool_other += windows * (128 * 512 * 2 + 128 * 8 + 64 * (512 * 2 + 256) + 64 * (512 * 2 + 32))
    host_maps += windows * 128 * 16
    compression = sum(2 * 64 * 512 * 2 + 4096 + ((2 * 2 + 64) * 1024 * 4
                      if plan.geometry.compress_ratios[source] == 2 else 0) for source in owned_sources)
    rope_types = {plan.geometry.compress_ratios[layer] > 0 for layer in range(begin, end)}
    if role == Role.TERMINAL:
        rope_types.add(False)
    gpu = {
        "BF16_source_index_shadow_payload": payload,
        "SWA_generations_undo_and_pool_scratch": pool_other,
        "FP32_compression_state_and_undo": compression,
        "FP32_RoPE_tables": len(rope_types) * plan.geometry.context * 64 * 4,
        "permanent_native_weights": native_bytes,
        "permanent_packed_weights": sum(footprints[p].nbytes for p in permanent),
        "prefix_and_request_checkpoints": 2 * windows * 128 * 512 * 2 + 4 * 1024 ** 2,
        "shared_EXL3_scratch_reservation": 64 * 1024 ** 2,
        "bounded_execution_workspaces": (2 if role == Role.HOST else 1) * GIB,
        "device_budget_headroom": GIB,
    }
    fixed = sum(gpu.values())
    slots = min(len(routed), 65536, max(0, (available_gpu_bytes - fixed) // slot_bytes))
    if not slots:
        raise ValidationError("device budget cannot hold eager BF16 pools, owned weights, workspaces and one exact expert")
    gpu["bounded_expert_arena"] = slots * slot_bytes
    host = {
        "locked_owned_weight_store": ((sum(info.nbytes for info in entries.values()) + 4095) // 4096 * 4096
                                      if role == Role.MIDDLE else 0),
        "fixed_pool_logical_generation_maps": host_maps,
        "token_image_Engram_history_reservation": plan.geometry.context * (20 if role == Role.HOST else 12),
        "bounded_transfer_reservation": 64 * 1024 ** 2,
        "pinned_and_boot_weight_staging": MAX_READ if role == Role.MIDDLE else 6 * MAX_READ,
        "image_and_Engram_row_caches": 384 * 1024 ** 2 if role == Role.HOST else 0,
        "runtime_metadata_and_libraries_reservation": 2 * GIB,
        "physical_host_headroom": GIB,
    }
    remaining = available_host_bytes - sum(host.values())
    tier = 0 if role == Role.MIDDLE else min(requested_ram_tier_bytes, max(0, remaining))
    if remaining < 0 or role != Role.MIDDLE and tier < 2 * slot_bytes:
        raise ValidationError("physical host budget cannot hold the mandatory resident state and bounded expert transfer tier")
    host["bounded_RAM_expert_tier"] = tier
    return RuntimeMemoryPlan(role, gpu, host, slots, slot_bytes, len(native), len(footprints), tier)
