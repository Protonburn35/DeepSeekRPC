"""Typed source/shadow bindings and window-scoped sparse selection state."""
from __future__ import annotations

from dataclasses import dataclass
import threading

from .cache_pool import DeviceRowPool
from .compression_state import DeviceCompressionState
from .index_selection import IndexSelection, TiledIndexer
from .jsonutil import ValidationError, integer
from .planner import Role, StagePlan
from .protocol import hex_id
from .state import Operation


@dataclass(frozen=True)
class AttentionWindow:
    operation: Operation
    history_digest: str
    phase: str
    encoder_tokens: int

    def __post_init__(self):
        hex_id(self.history_digest, 64, "immutable token/image history")
        if self.phase not in {"encoder", "target", "decode", "draft"} or self.operation.rows > 64:
            raise ValidationError("invalid bounded attention execution phase")
        integer(self.encoder_tokens, "encoder token extent", 1, 1048576)
        if self.operation.start + self.operation.rows > self.encoder_tokens:
            raise ValidationError("attention window exceeds its bound encoder extent")

    def visible(self, source: int, ratio: int):
        integer(ratio, "source compression ratio", 1, 2)
        # Encoder sources remain causal. Decoder source 20 sees the entire
        # encoded prompt only during the complete exact target replay.
        if source == 20 and self.phase == "target":
            return (self.encoder_tokens // ratio,) * self.operation.rows
        return tuple((position + 1) // ratio for position in range(
            self.operation.start, self.operation.start + self.operation.rows))


class AttentionState:
    """Only admitted owned pools and explicit replicas can enter the graph.

    Pool allocation is a separate eager admission step. This registry contains
    no cache fallback, implicit source allocation or process-global model hook.
    The worker must hold its execution lock through begin/compute/finish and
    coordinate all three participants before publishing a committed result.
    """
    def __init__(self, plan: StagePlan, role: Role, *, windows: dict[int, DeviceRowPool],
                 sources: dict[int, DeviceRowPool], index_keys: dict[int, DeviceRowPool],
                 include_draft: bool = False, tile_rows: int = 8192):
        if not isinstance(role, Role):
            raise ValidationError("attention registry requires an explicit stage")
        begin, end = plan.bounds(role)
        layers = set(range(begin, end))
        if include_draft:
            if role != Role.TERMINAL:
                raise ValidationError("draft attention belongs to stage C")
            layers.update(range(plan.geometry.layers, plan.geometry.layers + 3))
        needed = {s for s in plan.geometry.kv_sources if plan.owner(s) == role}
        needed.update(d.source_layer for d in plan.dependencies()
                      if d.recipient == role and d.field == "kv")
        if set(windows) != layers or set(sources) != needed or set(index_keys) != needed:
            raise ValidationError("attention pool inventory differs from owned layers and exact source replicas")
        all_pools = list(windows.values()) + list(sources.values()) + list(index_keys.values())
        if (any(not isinstance(p, DeviceRowPool) or p.closed for p in all_pools) or
                len({id(p) for p in all_pools}) != len(all_pools) or
                len({p.device for p in all_pools}) != 1):
            raise ValidationError("attention pools must be distinct live allocations on one CUDA device")
        g = plan.geometry
        for pool in windows.values():
            if pool.width != 512 or pool.map.slots != g.window or pool.map.context != g.context:
                raise ValidationError("SWA pool does not match the model geometry")
        for source in needed:
            slots = (g.context + g.compress_ratios[source] - 1) // g.compress_ratios[source]
            for width, pool in ((512, sources[source]), (128, index_keys[source])):
                if pool.width != width or pool.map.slots != slots or pool.map.context != g.context:
                    raise ValidationError("BF16 source/shadow allocation geometry mismatch")
        self.plan, self.role, self.device = plan, role, all_pools[0].device
        self.windows, self.sources, self.index_keys = dict(windows), dict(sources), dict(index_keys)
        self._pools, self._compressors = all_pools, {}
        self.indexer = TiledIndexer(tile_rows=tile_rows)
        self._mutex = threading.RLock()
        self._window, self._selections, self._begun_compressors = None, {}, []
        self._failed = False
        self._cancelled = None

    def attach_compressor(self, layer: int, state: DeviceCompressionState):
        if (layer in self._compressors or layer not in self.plan.geometry.kv_sources or
                self.plan.owner(layer) != self.role or not isinstance(state, DeviceCompressionState) or
                state.device != self.device or state.ratio != self.plan.geometry.compress_ratios[layer] or
                self._window is not None):
            raise ValidationError("compressor binding is duplicate, unowned or incompatible")
        self._compressors[layer] = state

    def reset(self, session: str, epoch: int):
        with self._mutex:
            if self._window is not None:
                raise ValidationError("attention reset requires no active window")
            for resource in [*self._pools, *self._compressors.values()]:
                resource.reset(session, epoch)
            self._selections.clear()
            self._failed = False

    def begin(self, window: AttentionWindow, *, cancelled=None):
        with self._mutex:
            if self._window is not None or self._failed or cancelled is not None and not callable(cancelled):
                raise ValidationError("attention state is active or needs coherent session recovery")
            begun = []
            try:
                for pool in self._pools:
                    pool.begin(window.operation)
                    begun.append(pool)
            except BaseException:
                for pool in reversed(begun):
                    pool.finish(window.operation, 0)
                raise
            self._window = window
            self._cancelled = cancelled
            self._selections.clear()
            self._begun_compressors.clear()

    @property
    def current(self) -> AttentionWindow:
        if self._window is None or self._failed:
            raise ValidationError("attention graph has no valid current window")
        return self._window

    def cancelled(self):
        return self._cancelled is not None and bool(self._cancelled())

    def compress(self, layer: int, scores):
        window = self.current
        if layer not in self._compressors or layer in self._begun_compressors:
            raise ValidationError("attention attempted an unbound or repeated source compressor")
        state = self._compressors[layer]
        state.begin(window.operation)
        self._begun_compressors.append(layer)
        return state.forward(window.operation, scores)

    def publish_source(self, layer: int, rows, kv, index):
        import torch
        window = self.current
        if (layer not in self._compressors or rows.operation != window.operation or
                kv.device != self.device or index.device != self.device or
                kv.dtype != torch.bfloat16 or index.dtype != torch.bfloat16 or
                kv.shape != (len(rows.logical_positions), 512) or
                index.shape != (len(rows.logical_positions), 128)):
            raise ValidationError("source publication is unowned or differs from its compressor window")
        if not rows.logical_positions:
            return
        for pool, values in ((self.sources[layer], kv), (self.index_keys[layer], index)):
            if window.phase == "target":
                # The first pass already produced this immutable encoder
                # state. Replay must reproduce it; do not overwrite global N.
                with pool.read_window(window.operation):
                    slots = torch.tensor(pool.physical(rows.logical_positions), dtype=torch.int64, device=self.device)
                    if not torch.equal(pool.data.index_select(0, slots).view(torch.int16), values.view(torch.int16)):
                        raise ValidationError("exact target replay changed a first-pass BF16 source row")
            else:
                pool.write(rows.logical_positions, rows.token_positions, values.contiguous())

    def publish_selection(self, layer: int, selection: IndexSelection):
        import torch
        window, g = self.current, self.plan.geometry
        if layer not in g.index_sources or layer in self._selections or not isinstance(selection, IndexSelection):
            raise ValidationError("selection source is unknown, duplicate or stale")
        kv_source = max(s for s in g.kv_sources if s <= layer)
        visible = window.visible(kv_source, g.compress_ratios[kv_source])
        if (selection.topk.device != self.device or selection.topk.dtype != torch.int32 or
                selection.topk.shape != (window.operation.rows, 512)):
            raise ValidationError("published top-k tensor has the wrong geometry or integer type")
        for ids, count in zip(selection.topk.tolist(), visible):
            previous, padded = -1, False
            for logical in ids:
                if logical == -1:
                    padded = True
                elif padded or not previous < logical < count:
                    raise ValidationError("published top-k is not sorted, unique and within current visibility")
                else:
                    previous = logical
        candidates = selection.candidate_blocks
        if (layer == g.candidate_source) != (candidates is not None):
            raise ValidationError("candidate source publication is missing or belongs to another layer")
        if candidates is not None:
            if (candidates.device != self.device or candidates.dtype != torch.int32 or
                    candidates.shape != (window.operation.rows, 2048)):
                raise ValidationError("published candidate blocks have incompatible geometry")
            for blocks, count in zip(candidates.tolist(), visible):
                previous, padded = -1, False
                for block in blocks:
                    if block == -1:
                        padded = True
                    elif padded or not previous < block < (count + 7) // 8:
                        raise ValidationError("candidate blocks are not sorted, unique and currently visible")
                    else:
                        previous = block
        self._selections[layer] = selection

    def selection(self, layer: int) -> IndexSelection:
        self.current
        if layer not in self._selections:
            raise ValidationError("required local or transferred source selection has not been published")
        return self._selections[layer]

    def select(self, layer: int, query, weights, *, cancelled=None):
        g, window = self.plan.geometry, self.current
        if layer not in g.index_sources or self.plan.owner(layer) != self.role:
            raise ValidationError("only an owned index source can score keys")
        source = max(s for s in g.kv_sources if s <= layer)
        candidates = self.selection(g.candidate_source).candidate_blocks if layer > g.candidate_source else None
        result = self.indexer.select(self.index_keys[source], window.operation, query, weights,
                                     window.visible(source, g.compress_ratios[source]),
                                     produce_candidates=layer == g.candidate_source,
                                     candidate_blocks=candidates, cancelled=cancelled)
        self.publish_selection(layer, result)
        return result

    def fail(self):
        self._failed = True

    def finish(self, operation: Operation, accepted: int):
        with self._mutex:
            if self._window is None or self._window.operation != operation or self._failed and accepted:
                raise ValidationError("attention finish targets a stale or failed window")
            try:
                for layer in self._begun_compressors:
                    self._compressors[layer].finish(operation, accepted)
                for pool in self._pools:
                    pool.finish(operation, accepted)
            except BaseException:
                # A partially failed cross-resource finish is a session fault;
                # the coordinator must never publish it as a coherent commit.
                self._failed = True
                raise
            self._window = None
            self._cancelled = None
            self._selections.clear()
            self._begun_compressors.clear()
            self._failed = False

    def rewind_for_exact_target(self, history_digest: str, encoder_tokens: int):
        """Preserve full encoder sources; clear every local SWA and open pair.

        The coordinator must establish the matching immutable full token/image
        history and all-stage encoder barrier before invoking this operation.
        Local source and shadow validity is checked in bounded metadata tiles.
        """
        hex_id(history_digest, 64, "exact replay history")
        integer(encoder_tokens, "exact replay encoder extent", 1, self.plan.geometry.context)
        with self._mutex:
            if self._window is not None or self._failed:
                raise ValidationError("exact target rewind requires a coherent quiescent encoder barrier")
            for source, pool in self.sources.items():
                count = encoder_tokens // self.plan.geometry.compress_ratios[source]
                for keys in (pool, self.index_keys[source]):
                    for start in range(0, count, 8192):
                        keys.physical(range(start, min(start + 8192, count)))
            for pool in self.windows.values():
                pool.rewind_ring_for_replay(pool.map.session, pool.map.epoch)
            for state in self._compressors.values():
                state.rewind_for_replay(state.session, state.epoch)
