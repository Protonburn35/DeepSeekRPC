"""Permanent, bounded packed storage for the owned non-routed projections."""
from __future__ import annotations

from dataclasses import dataclass
import threading

from .inventory import MAX_READ
from .jsonutil import ValidationError, integer
from .packed import DevicePackedLinear, PackedLinearLayout
from .planner import Role


@dataclass(frozen=True)
class DenseBinding:
    prefixes: tuple[str, ...]
    mode: str = "concat"
    output_fp32: bool = False

    def __post_init__(self):
        if (not isinstance(self.prefixes, tuple) or not 1 <= len(self.prefixes) <= 8 or
                len(set(self.prefixes)) != len(self.prefixes) or
                any(not isinstance(p, str) or not p or len(p) > 512 for p in self.prefixes) or
                self.mode not in {"concat", "grouped"} or type(self.output_fp32) is not bool):
            raise ValidationError("invalid permanent packed projection binding")


def layer_bindings(prefixes, *, source_layer: str, module_layer: str):
    """Map the pinned physical projections to a real vLLM decoder's modules."""
    local = {prefix.removeprefix(source_layer + ".") for prefix in prefixes
             if prefix.startswith(source_layer + ".") and ".ffn.experts." not in prefix}
    required = {"attn.wq_a", "attn.wkv", "attn.wq_b", "attn.wo_b",
                "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"}
    required.update(f"attn.wo_a.slice.{group}" for group in range(8))
    if not required <= local:
        raise ValidationError("owned decoder packed projection inventory is incomplete")
    bindings = {}

    def bind(target, components, *, mode="concat", output_fp32=False):
        if not set(components) <= local:
            raise ValidationError("fused packed projection is missing an original matrix")
        bindings[f"{module_layer}.{target}"] = DenseBinding(
            tuple(f"{source_layer}.{name}" for name in components), mode, output_fp32)
        local.difference_update(components)

    bind("attn.fused_wqa_wkv", ("attn.wq_a", "attn.wkv"))
    bind("attn.wo_a", tuple(f"attn.wo_a.slice.{group}" for group in range(8)), mode="grouped")
    bind("ffn.shared_experts.gate_up_proj", ("ffn.shared_experts.w1", "ffn.shared_experts.w3"))
    bind("ffn.shared_experts.down_proj", ("ffn.shared_experts.w2",))
    compressor = tuple(name for name in ("attn.compressor.wkv", "attn.compressor.wgate") if name in local)
    if compressor:
        if compressor[0] != "attn.compressor.wkv":
            raise ValidationError("compressor gate has no value projection")
        bind("attn.compressor.fused_wkv_wgate", compressor, output_fp32=True)
    # Remaining source linears have the same submodule suffix: wq_b, wo_b,
    # indexer q/k, Engram.wkv, and DSpark's explicit main projection.
    known = {"attn.wq_b", "attn.wo_b", "attn.indexer.wq_b", "attn.indexer.wk", "engram.wkv", "main_proj"}
    if not local <= known:
        raise ValidationError("unrecognized October packed decoder projection")
    for name in sorted(local):
        bindings[f"{module_layer}.{name}"] = DenseBinding((f"{source_layer}.{name}",))
    return bindings


def stage_bindings(plan, role: Role, prefixes, *, include_draft=True):
    """Bind constructor prefixes, including absolute draft layer IDs 40..42."""
    prefixes = set(prefixes)
    result = {}
    begin, end = plan.bounds(role)
    for layer in range(begin, end):
        result.update(layer_bindings(prefixes, source_layer=f"layers.{layer}", module_layer=f"model.layers.{layer}"))
    if role == Role.TERMINAL:
        if "head" not in prefixes:
            raise ValidationError("stage C is missing its packed vocabulary head")
        result["lm_head"] = DenseBinding(("head",), output_fp32=True)
        if include_draft:
            for draft in range(3):
                result.update(layer_bindings(prefixes, source_layer=f"mtp.{draft}",
                                             module_layer=f"model.layers.{plan.geometry.layers + draft}"))
            # The shared context combiner is a member of the first draft
            # block, while the decoder constructor uses absolute layer 40.
            result["model.mtp.0.main_proj"] = result.pop("model.layers.40.main_proj")
    consumed = {name for binding in result.values() for name in binding.prefixes}
    expected = {name for name in prefixes if ".ffn.experts." not in name}
    if role == Role.TERMINAL and not include_draft:
        expected = {name for name in expected if not name.startswith("mtp.")}
    if consumed != expected:
        raise ValidationError("owned permanent packed inventory contains an unmapped constructor projection")
    return result


class PackedBootUploader:
    """Bounded startup-only copies; B always reads its locked resident store."""
    def __init__(self, source):
        import torch
        from .resident_weights import LockedOwnedWeights
        self.source = source
        self.resident = isinstance(source, LockedOwnedWeights)
        if self.resident:
            source.register_cuda()
            self.staging_bytes = 0
        else:
            if source.role == Role.MIDDLE:
                raise ValidationError("B's packed bootstrap requires its completed locked resident store")
            self.staging_bytes = 3 * MAX_READ
            self._scratch = bytearray(MAX_READ)
            self._scratch_view = torch.frombuffer(self._scratch, dtype=torch.uint8)
            self._pinned = torch.empty(MAX_READ, dtype=torch.uint8, device="cpu", pin_memory=True)

    def upload(self, layout, destination):
        for part in layout.parts:
            for offset in range(0, part.info.nbytes, MAX_READ):
                count = min(MAX_READ, part.info.nbytes - offset)
                target = destination[part.offset + offset:part.offset + offset + count]
                self.copy_range(part.info.name, offset, target)

    def copy_range(self, name, offset, destination):
        import torch
        count = destination.numel()
        if (destination.device.type != "cuda" or destination.dtype != torch.uint8 or
                not destination.is_contiguous() or destination.ndim != 1 or not 1 <= count <= MAX_READ):
            raise ValidationError("boot copy requires one bounded contiguous CUDA byte range")
        if self.resident:
            self.source.copy_to_cuda(name, offset, destination)
        else:
            chunk = self.source.read_range(name, offset, count)
            self._scratch[:count] = chunk
            self._pinned[:count].copy_(self._scratch_view[:count])
            destination.copy_(self._pinned[:count], non_blocking=False)


class FixedPackedWeights:
    """One admitted immutable GPU allocation; actual BC_LinearEXL3 projections.

    This excludes routed experts: they use DeviceExpertCache's bounded slots.
    The owner retains this allocation until all work completes and closes it
    explicitly. Single-row ExLlama bindings share upstream scratch, so this
    component enforces one compute stream rather than auxiliary GEMM overlap.
    """
    def __init__(self, source, prefixes, *, admitted_bytes: int, device="cuda:0"):
        import torch
        integer(admitted_bytes, "permanent packed device admission", 256, 1 << 40)
        prefixes = tuple(prefixes)
        if (not prefixes or len(prefixes) > 4096 or len(set(prefixes)) != len(prefixes) or
                any(".ffn.experts." in prefix for prefix in prefixes)):
            raise ValidationError("permanent packed inventory must contain only owned non-routed projections")
        self.layouts = {prefix: PackedLinearLayout.from_source(source, prefix) for prefix in sorted(prefixes)}
        self.allocation_bytes = sum(layout.nbytes for layout in self.layouts.values())
        if self.allocation_bytes > admitted_bytes:
            raise ValidationError("permanent packed projections exceed admitted GPU bytes")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("permanent packed projections require the actual CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._mutex, self.closed = threading.RLock(), False
        self.matrices = {}
        with torch.cuda.device(self.device):
            self.compute_stream = torch.cuda.current_stream(self.device)
            self._arena = torch.zeros(self.allocation_bytes, dtype=torch.uint8, device=self.device)
            try:
                uploader = PackedBootUploader(source)
                self.boot_staging_bytes = uploader.staging_bytes
                offset = 0
                for prefix, layout in self.layouts.items():
                    storage = self._arena.narrow(0, offset, layout.nbytes)
                    uploader.upload(layout, storage)
                    self.matrices[prefix] = DevicePackedLinear(layout, storage)
                    offset += layout.nbytes
                self.compute_stream.synchronize()
            except BaseException:
                self.close()
                raise

    def binding_geometry(self, binding: DenseBinding):
        if not isinstance(binding, DenseBinding) or any(p not in self.layouts for p in binding.prefixes):
            raise ValidationError("packed projection binding refers to an unowned or missing matrix")
        layouts = [self.layouts[p] for p in binding.prefixes]
        if len({value.in_features for value in layouts}) != 1:
            raise ValidationError("packed projection binding inputs differ")
        if binding.mode == "grouped" and len({value.out_features for value in layouts}) != 1:
            raise ValidationError("grouped packed projection outputs differ")
        return layouts[0].in_features, sum(value.out_features for value in layouts)

    def forward(self, binding: DenseBinding, inputs):
        import torch
        with self._mutex, torch.cuda.device(self.device):
            width, output_width = self.binding_geometry(binding)
            if self.closed or inputs.ndim != (3 if binding.mode == "grouped" else 2):
                raise ValidationError("permanent packed projection is closed or has incompatible input rank")
            expected = ((len(inputs), len(binding.prefixes), width) if binding.mode == "grouped"
                        else (len(inputs), width))
            if (self.closed or inputs.device != self.device or torch.cuda.current_stream(self.device) != self.compute_stream or
                    inputs.shape != expected or not 1 <= len(inputs) <= 64 or
                    inputs.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
                raise ValidationError("permanent packed projection input or compute stream mismatch")
            output = torch.empty((len(inputs), output_width), dtype=torch.float32, device=self.device)
            offset = 0
            for group, prefix in enumerate(binding.prefixes):
                value = inputs[:, group] if binding.mode == "grouped" else inputs
                value = value.to(torch.float16).contiguous()
                if not bool(torch.isfinite(value).all()):
                    raise ValidationError("permanent packed input conversion overflowed")
                result = self.matrices[prefix].forward(value)
                output[:, offset:offset + result.shape[1]].copy_(result)
                offset += result.shape[1]
            converted = output if binding.output_fp32 else output.to(inputs.dtype)
            if not bool(torch.isfinite(converted).all()):
                raise ValidationError("permanent packed projection output is nonfinite")
            return converted

    def close(self):
        import torch
        with self._mutex, torch.cuda.device(self.device):
            if not self.closed:
                self.compute_stream.synchronize()
                for matrix in self.matrices.values():
                    matrix.close()
                self.matrices.clear()
                self._arena = None
                self.closed = True
