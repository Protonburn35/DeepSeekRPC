"""Owned step dependency transfers over the actual three-stage data mesh."""
from dataclasses import asdict, dataclass

from .execution import StageExecution, source_closures
from .index_selection import IndexSelection
from .jsonutil import ValidationError, fingerprint
from .planner import Role
from .protocol import Frame, Kind, TensorDescriptor
from .tensor_wire import CudaTensorWire


@dataclass(frozen=True)
class TransferSpec:
    semantic: str
    source: int
    dtype: str
    shape: tuple[int, ...]
    logical_row: int

    def descriptor(self, window):
        tensor_id = fingerprint({"window": asdict(window), "field": asdict(self)})[:32]
        return TensorDescriptor(self.semantic, self.source, self.dtype, self.shape,
                                self.logical_row, tensor_id, 0, 0)

    def frame(self, run, sender, receiver, window):
        op = window.operation
        kind = Kind.STAGE_INPUT if self.semantic in {"hidden_states", "pre_mix"} else Kind.SOURCE_DELTA
        return Frame(kind, run, sender, receiver, op.session, op.request, op.epoch,
                     op.number, 0, op.transaction, op.start, op.rows, tensor=self.descriptor(window))


def transfers(plan, sender, receiver, window):
    """Derive local expected geometry; never allocate from a peer descriptor."""
    if not isinstance(sender, Role) or not isinstance(receiver, Role) or sender >= receiver:
        raise ValidationError("stage data follows explicit upstream-to-downstream owners")
    if window.phase == "draft":
        return ()
    dependencies = [d for d in plan.dependencies() if d.owner == sender and d.recipient == receiver]
    result, rows, start = [], window.operation.rows, window.operation.start
    if window.phase in {"encoder", "decode"}:
        for source in sorted({d.source_layer for d in dependencies if d.field == "kv"}):
            logical, _ = source_closures(window, plan.geometry.compress_ratios[source])
            if logical:
                result += [TransferSpec("kv", source, "bf16", (len(logical), 512), logical[0]),
                           TransferSpec("index_k", source, "bf16", (len(logical), 128), logical[0])]
    selections = {d.source_layer for d in dependencies if d.field in {"topk", "candidates"}}
    for source in sorted(selections):
        if window.phase == "encoder" and source >= 20:
            continue
        result.append(TransferSpec("topk", source, "i32", (rows, 512), start))
        if source == plan.geometry.candidate_source:
            result.append(TransferSpec("candidates", source, "i32", (rows, 2048), start))
    boundary = receiver == sender + 1 and (window.phase != "encoder" or plan.bounds(receiver)[0] <= 20)
    if boundary:
        result += [TransferSpec("hidden_states", -1, "fp32", (rows, 4, 5120), start),
                   TransferSpec("pre_mix", -1, "fp32", (rows, 4), start)]
    return tuple(result)


class StageDataMesh:
    """Called only by the serialized model actor, before/after its forward."""
    def __init__(self, execution: StageExecution, connections):
        if not isinstance(execution, StageExecution):
            raise ValidationError("stage data mesh requires its actual CUDA execution owner")
        stage = execution.stage
        if set(connections) != set(Role) - {stage.role}:
            raise ValidationError("data mesh must contain exactly the other two authenticated peers")
        if (len({id(c) for c in connections.values()}) != 2 or
                len({c.run for c in connections.values()}) != 1):
            raise ValidationError("data mesh channels are duplicated or belong to different runs")
        for peer, connection in connections.items():
            if (connection.local, connection.peer, connection.plane) != (stage.role, peer, "data"):
                raise ValidationError("mesh data channel differs from its certificate role/plane")
        self.execution, self.connections = execution, dict(connections)
        self.role, self.run = stage.role, next(iter(connections.values())).run
        self.wires = {peer: CudaTensorWire(connection, device=stage.attention.device, stream=stage.stream,
                                           cancelled=stage.cancellation.is_set)
                      for peer, connection in connections.items()}

    def _frame(self, peer, window, spec, *, incoming):
        sender, receiver = (peer, self.role) if incoming else (self.role, peer)
        return spec.frame(self.run, sender, receiver, window)

    def receive_dependencies(self, window):
        import torch
        from .stage_graph import StageBoundary
        stage = self.execution.stage
        values = {}
        with torch.cuda.device(stage.attention.device), torch.cuda.stream(stage.stream), torch.inference_mode():
            for peer in sorted(self.connections):
                if peer >= self.role:
                    continue
                specs = transfers(stage.plan, peer, self.role, window)
                for spec in specs:
                    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "i32": torch.int32}[spec.dtype]
                    destination = torch.empty(spec.shape, dtype=dtype, device=stage.attention.device)
                    self.wires[peer].receive(self._frame(peer, window, spec, incoming=True),
                                              spec.descriptor(window), destination)
                    values[spec.semantic, spec.source] = destination
                for source in sorted({s.source for s in specs if s.semantic == "kv"}):
                    self.execution.import_source(source, values.pop(("kv", source)), values.pop(("index_k", source)))
                for source in sorted({s.source for s in specs if s.semantic == "topk"}):
                    selection = IndexSelection(values.pop(("topk", source)), values.pop(("candidates", source), None))
                    self.execution.import_selection(source, selection)
            hidden, mix = values.pop(("hidden_states", -1), None), values.pop(("pre_mix", -1), None)
            if values or (hidden is None) != (mix is None):
                raise ValidationError("received stage fields do not form the exact admitted boundary")
            return None if hidden is None else StageBoundary(window, hidden, mix)

    def send_outputs(self, window, result):
        for peer in sorted(self.connections):
            if peer <= self.role:
                continue
            specs = transfers(self.execution.stage.plan, self.role, peer, window)
            values = {}
            for source in sorted({s.source for s in specs if s.semantic == "kv"}):
                exported = self.execution.export_source(source)
                if exported is None:
                    raise ValidationError("owner failed to publish its current exact source closures")
                _, _, kv, index = exported
                values["kv", source], values["index_k", source] = kv, index
            for source in sorted({s.source for s in specs if s.semantic == "topk"}):
                selection = self.execution.export_selection(source)
                values["topk", source] = selection.topk
                if source == self.execution.stage.plan.geometry.candidate_source:
                    values["candidates", source] = selection.candidate_blocks
            if any(spec.semantic == "hidden_states" for spec in specs):
                if result is None or result.boundary is None or result.boundary.window != window:
                    raise ValidationError("actual stage did not produce its required FP32 boundary")
                values["hidden_states", -1] = result.boundary.hidden
                values["pre_mix", -1] = result.boundary.pre_mix
            for spec in specs:
                self.wires[peer].send(self._frame(peer, window, spec, incoming=False), spec.descriptor(window),
                                      values[spec.semantic, spec.source])

    def close(self):
        for connection in self.connections.values():
            connection.close()
