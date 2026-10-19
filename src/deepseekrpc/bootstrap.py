"""Actual native Windows owned-stage loading with admission before allocation.

Loading returns a LOADED stage. The distributed worker must still complete its
peer identity/phase barrier and graph checks before advertising READY. This
module never downloads weights or constructs the stock whole-model loader.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
from importlib import resources
from pathlib import Path
import threading

from .inventory import MAX_READ
from .jsonutil import ValidationError, fingerprint, integer, loads
from .planner import Geometry, Role, StagePlan
from .protocol import hex_id
from .weights import SelectiveWeights, ShardPin


def deployment_manifest():
    data = resources.files("deepseekrpc._vendor.vllm_windows").joinpath("DEPLOYMENT.json").read_bytes()
    return loads(data, limit=128 * 1024)


def _verified_metadata(root: Path, record):
    path = (root / record["name"]).resolve(strict=True)
    if path.parent != root or path.stat().st_size != record["bytes"] or record["bytes"] > 32 * 1024 * 1024:
        raise ValidationError("deployment metadata path/length differs from the pinned local file")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValidationError("deployment metadata differs from its publisher-pinned fingerprint")
    return data


def local_deployment(model_root: Path, *, split=(18, 5, 17)):
    """Verify the small local metadata without opening any model payload."""
    root = model_root.resolve(strict=True)
    if not root.is_dir():
        raise ValidationError("model root must be an existing local directory")
    manifest = deployment_manifest()
    metadata = manifest["pack"]["metadata"]
    config = loads(_verified_metadata(root, metadata["config.json"]), limit=64 * 1024)
    _verified_metadata(root, metadata["model.safetensors.index.json"])
    _verified_metadata(root, metadata["tokenizer.json"])
    plan = StagePlan(Geometry.from_config(config), tuple(split))
    # These constants bind concrete native kernels and the actual graph, not
    # an interchangeable model family with superficially compatible fields.
    g = plan.geometry
    if (g.layers, g.hidden, g.streams, g.context, g.window, g.kv_width, g.index_width) != (
            40, 5120, 4, 1048576, 128, 512, 128):
        raise ValidationError("this runtime requires the pinned complete October DeepSeek geometry")
    if any(plan.owner(layer) != Role.TERMINAL for layer in (37, 38, 39)):
        raise ValidationError("all three DSpark target feature layers must belong to stage C")
    identity = fingerprint({"pack": manifest["pack"], "plan": plan.as_dict(),
                            "token_map": manifest["token_map"]["canonical_lookup_sha256"]})
    return manifest, config, plan, identity


def _pool_specs(plan, role):
    begin, end = plan.bounds(role)
    layers = list(range(begin, end)) + (list(range(40, 43)) if role == Role.TERMINAL else [])
    for layer in layers:
        yield "windows", layer, 128, 512
    sources = {s for s in plan.geometry.kv_sources if plan.owner(s) == role}
    sources.update(d.source_layer for d in plan.dependencies() if d.field == "kv" and d.recipient == role)
    for source in sorted(sources):
        slots = (plan.geometry.context + plan.geometry.compress_ratios[source] - 1) // plan.geometry.compress_ratios[source]
        yield "sources", source, slots, 512
        yield "index_keys", source, slots, 128


@dataclass
class LoadedStage:
    role: Role
    plan: StagePlan
    identity: str
    graph: object
    attention: object
    weights: object
    experts: object
    packed: object
    memory: object
    guard: object
    stream: object
    cancellation: threading.Event
    tokenizer: object
    engram_history: object
    report: dict
    _resources: ExitStack
    closed: bool = False

    def close(self):
        if not self.closed:
            self.cancellation.set()
            self.stream.synchronize()
            self.graph = self.attention = None
            self._resources.close()
            self.weights = self.experts = self.packed = self.engram_history = self.tokenizer = None
            self.closed = True


def load_stage(model_root: Path, *, role: Role, run_id: str, rendezvous: Path,
               engram_root: Path | None = None, split=(18, 5, 17), device=0,
               requested_ram_tier_bytes=8 * 1024 ** 3):
    from .runtime_identity import require_windows_runtime
    # Establish exact EXL3/math policy before any model/backend import.
    runtime = require_windows_runtime(device)
    import torch
    from .attention_state import AttentionState
    from .cache_pool import DeviceRowPool
    from .cuda_cache import DesktopUploader, DeviceExpertCache, ResidentUploader
    from .dense_weights import FixedPackedWeights, stage_bindings
    from .engram_history import EngramHistory
    from .engram_storage import EngramRows
    from .gpu_selftest import run_cuda_checks
    from .memory_budget import query_device_budget, RuntimeBudgetGuard
    from .native_windows import physical_host_memory
    from .resident_weights import LockedOwnedWeights
    from .routed_experts import BoundedRoutedExperts, ExpertCatalog
    from .runtime_config import make_vllm_config
    from .runtime_memory import plan_runtime_memory
    from .stage_graph import initialize_local_groups, OwnedStageGraph
    from .storage import RAMCache
    from .tokenizer import TokenizerAdapter
    from .vllm_experts import OwnedExl3Config

    if not isinstance(role, Role):
        raise ValidationError("stage loading needs an explicit A/B/C role")
    hex_id(run_id, 32, "worker run")
    integer(device, "CUDA device index", 0, 31)
    manifest, config, plan, identity = local_deployment(model_root, split=split)
    root = model_root.resolve(strict=True)
    if (role == Role.HOST) != (engram_root is not None):
        raise ValidationError("exactly stage A must supply the original local Engram directory")
    vllm_config = make_vllm_config(root)
    initialize_local_groups(vllm_config, rendezvous)
    cancellation = threading.Event()
    with ExitStack() as cleanup:
        cleanup.callback(torch.distributed.destroy_process_group)
        pins = ShardPin.from_lock(manifest["shards"])
        index = manifest["pack"]["metadata"]["model.safetensors.index.json"]
        source = SelectiveWeights(root, root / index["name"], index_sha256=index["sha256"],
                                  pins=pins, plan=plan, role=role,
                                  replicate_draft_embedding=role == Role.TERMINAL)
        cleanup.callback(source.close)
        # info() validates each physical shard and its complete publisher SHA
        # with bounded reads. No model allocation occurs before admission.
        entries = {name: source.info(name) for name in sorted(source.names)}
        baseline, host = query_device_budget(device), physical_host_memory()
        memory = plan_runtime_memory(entries, plan, role, available_gpu_bytes=baseline.available,
                                     available_host_bytes=host["available_bytes"],
                                     requested_ram_tier_bytes=requested_ram_tier_bytes)
        if role == Role.MIDDLE:
            source = LockedOwnedWeights(source, admitted_bytes=memory.host["locked_owned_weight_store"])
            cleanup.callback(source.close)
        cuda_device = torch.device("cuda", device)
        stream = torch.cuda.Stream(device=cuda_device)
        with torch.cuda.device(cuda_device), torch.cuda.stream(stream), torch.inference_mode():
            pools = {"windows": {}, "sources": {}, "index_keys": {}}
            pool_device = pool_host = 0
            for kind, layer, slots, width in _pool_specs(plan, role):
                salt = hashlib.sha256(f"{run_id}/{int(role)}/{kind}/{layer}".encode()).digest()
                # Different source/shadow maps deliberately cannot be reused
                # as wire offsets. Peers transfer logical IDs and exact rows.
                offset = int.from_bytes(salt[:8], "little") % slots
                allocation = slots * (width * 2 + 8) + 64 * (width * 2 + 256) + 64 * (width * 2 + 32)
                metadata_bytes = slots * 16
                pool = DeviceRowPool(slots, width, context=1048576, offset=offset,
                                     admitted_bytes=allocation + metadata_bytes, device=cuda_device)
                cleanup.callback(pool.close)
                pools[kind][layer] = pool
                pool_device += pool.allocation_bytes
                pool_host += pool.map.metadata_bytes
            if (pool_device != memory.gpu["BF16_source_index_shadow_payload"] + memory.gpu["SWA_generations_undo_and_pool_scratch"] or
                    pool_host != memory.host["fixed_pool_logical_generation_maps"]):
                raise ValidationError("actual eager pool admission disagrees with its complete memory plan")
            attention = AttentionState(plan, role, **pools, include_draft=role == Role.TERMINAL)
            prefixes = {name.removesuffix(".trellis") for name in source.names if name.endswith(".trellis")}
            bindings = stage_bindings(plan, role, prefixes)
            permanent = {p for binding in bindings.values() for p in binding.prefixes}
            packed = FixedPackedWeights(source, sorted(permanent),
                                        admitted_bytes=memory.gpu["permanent_packed_weights"], device=cuda_device)
            cleanup.callback(packed.close)
            if role == Role.MIDDLE:
                uploader = ResidentUploader(source)
            else:
                # Reserve one extra bundle for bytearray -> immutable bytes
                # construction; both belong to the declared RAM-tier budget.
                tier = memory.ram_tier_bytes
                ram = RAMCache(tier - memory.expert_slot_bytes, memory.expert_slot_bytes,
                               max_entries=65536, max_inflight=1)
                uploader = DesktopUploader(source, ram)
            experts = DeviceExpertCache(admitted_bytes=memory.gpu["bounded_expert_arena"],
                                        slot_bytes=memory.expert_slot_bytes, uploader=uploader, device=cuda_device)
            cleanup.callback(experts.close)
            begin, end = plan.bounds(role)
            layers = [(layer, f"layers.{layer}", 384, 6) for layer in range(begin, end)]
            if role == Role.TERMINAL:
                layers += [(40 + draft, f"mtp.{draft}", 128, 3) for draft in range(3)]
            executors = {}
            for layer, prefix, count, topk in layers:
                catalog = ExpertCatalog.from_source(source, model=identity, quantization=identity,
                                                    layer=layer, experts=count, prefix=f"{prefix}.ffn.experts")
                executors[f"model.layers.{layer}.ffn.experts"] = BoundedRoutedExperts(
                    catalog, experts, topk=topk, cancelled=cancellation.is_set)
            vllm_config.quant_config = OwnedExl3Config.bind(vllm_config.quant_config, executors,
                                                           packed_weights=packed, dense_bindings=bindings)
            stores, tokenizer, history = {}, None, None
            if role == Role.HOST:
                original = engram_root.resolve(strict=True)
                original_pins = ShardPin.from_lock(manifest["shards"], model="official")
                for layer, filename in ((1, "model-00047-of-00048.safetensors"),
                                        (14, "model-00048-of-00048.safetensors")):
                    path = (original / filename).resolve(strict=True)
                    if path.parent != original:
                        raise ValidationError("original Engram shard escapes its admitted directory")
                    stores[layer] = EngramRows(path, layer=layer, cache_bytes=128 * 1024 ** 2,
                                               pin=original_pins[filename])
                    cleanup.callback(stores[layer].close)
                tokenizer = TokenizerAdapter(root / "tokenizer.json",
                                             expected_sha256=manifest["token_map"]["tokenizer_sha256"])
                history = EngramHistory(config, tokenizer,
                                         expected_map_sha256=manifest["token_map"]["canonical_lookup_sha256"])
            graph = OwnedStageGraph(vllm_config, attention, row_stores=stores)
            native = graph.load_owned_native(source, admitted_bytes=memory.gpu["permanent_native_weights"] +
                                            MAX_READ + memory.host["pinned_and_boot_weight_staging"])
            if sum(record["target_bytes"] for record in native) != memory.gpu["permanent_native_weights"]:
                raise ValidationError("actual native parameters disagree with complete destination-dtype admission")
            checks = run_cuda_checks()
            stream.synchronize()
        guard = RuntimeBudgetGuard(memory, baseline=baseline, device=device)
        report = {"status": "LOADED", "stage": "ABC"[int(role)], "identity": identity,
                  "runtime": runtime, "memory": memory.as_dict(), "native_parameters": native,
                  "small_CUDA_checks": checks, "current_budgets": guard.check(),
                  "distributed_graph_qualification": "pending peer/phase barrier and real graph checks"}
        return LoadedStage(role, plan, identity, graph, attention, source, experts, packed, memory,
                           guard, stream, cancellation, tokenizer, history, report, cleanup.pop_all())
