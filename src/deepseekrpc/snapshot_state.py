"""Exact state inventory and bounded readers for real owned CUDA stages."""
from contextlib import ExitStack
from dataclasses import asdict
import hashlib

import numpy as np

from .history import ImageSpan, TokenHistory
from .jsonutil import ValidationError, fingerprint, integer
from .planner import Role
from .prefix import prefix_offer
from .protocol import MAX_PAYLOAD, hex_id
from .request_settings import RequestSettings
from .snapshot_archive import Record, archive_chunks, encode_header, file_chunks, locked_read
from .vision_input import ImageInfo

TIMELINE = {"session", "request", "epoch", "history", "encoder_tokens", "processed", "pending_token", "stop_reason", "next_operation"}


def timeline(service):
    ledger = service.execution.ledger
    return {"session": ledger.session, "request": ledger.request, "epoch": ledger.epoch,
            "history": ledger.history_digest, "encoder_tokens": ledger.encoder_tokens,
            "processed": ledger.processed, "pending_token": ledger.pending_token,
            "stop_reason": ledger.stop_reason, "next_operation": ledger._next}


def validate_timeline(value):
    if not isinstance(value, dict) or set(value) != TIMELINE:
        raise ValidationError("snapshot timeline has unknown or missing fields")
    for key in ("session", "request"):
        hex_id(value[key], 32, "snapshot session/request")
    hex_id(value["history"], 64, "snapshot history")
    integer(value["epoch"], "snapshot epoch", 0, (1 << 63) - 2)
    integer(value["encoder_tokens"], "snapshot original encoder extent", 1, 1048575)
    integer(value["processed"], "snapshot processed target rows", value["encoder_tokens"], 1048576)
    integer(value["next_operation"], "snapshot next operation", 1, (1 << 63) - 1)
    if value["pending_token"] is not None:
        integer(value["pending_token"], "snapshot pending token", 0, 129279)
        if value["processed"] == 1048576:
            raise ValidationError("snapshot pending token exceeds the full context")
    if (value["stop_reason"] is not None and not isinstance(value["stop_reason"], str) or
            value["stop_reason"] not in {None, "eos", "length", "stop"} or
            value["pending_token"] is None and value["stop_reason"] is None):
        raise ValidationError("snapshot has no defined next input or committed stopping reason")
    return value["processed"] + (value["pending_token"] is not None)


def pool_ranges(plan, role, processed):
    begin, end = plan.bounds(role)
    for layer in list(range(begin, end)) + (list(range(40, 43)) if role == Role.TERMINAL else []):
        yield "windows", layer, max(0, processed - 128), min(128, processed), 512
    sources = {source for source in plan.geometry.kv_sources if plan.owner(source) == role}
    sources.update(d.source_layer for d in plan.dependencies() if d.field == "kv" and d.recipient == role)
    for source in sorted(sources):
        count = processed // plan.geometry.compress_ratios[source]
        yield "sources", source, 0, count, 512
        yield "index_keys", source, 0, count, 128


def state_inventory(plan, role, metadata, *, identity, run):
    """Derive all shapes locally before allocating or mutating live state."""
    required = {"abi", "stage", "identity", "run", "timeline", "settings", "images", "assets", "prefix", "rng", "stops"}
    if (not isinstance(metadata, dict) or set(metadata) != required or type(metadata["abi"]) is not int or metadata["abi"] != 1 or
            type(metadata["stage"]) is not int or metadata["stage"] != int(role) or
            metadata["identity"] != identity or metadata["run"] != run):
        raise ValidationError("snapshot stage/runtime/precision/plan identity differs")
    length = validate_timeline(metadata["timeline"])
    settings = RequestSettings.from_wire(metadata["settings"])
    if length - metadata["timeline"]["encoder_tokens"] > settings.max_tokens:
        raise ValidationError("snapshot generated history exceeds its saved output limit")
    if not isinstance(metadata["images"], list) or len(metadata["images"]) > 4096:
        raise ValidationError("snapshot image span inventory exceeds its bound")
    images, previous = [], 0
    for value in metadata["images"]:
        if not isinstance(value, dict) or set(value) != set(ImageSpan.__dataclass_fields__):
            raise ValidationError("unknown snapshot image span fields")
        span = ImageSpan(**value)
        if span.start < previous or span.start + span.length > metadata["timeline"]["encoder_tokens"]:
            raise ValidationError("snapshot image spans overlap or extend into generated output")
        images.append(span)
        previous = span.start + span.length
    records = [Record("history.tokens", "i32", (length,)), Record("history.mask", "u8", (length,))]
    for kind, layer, start, count, width in pool_ranges(plan, role, metadata["timeline"]["processed"]):
        records.append(Record(f"{kind}.{layer}", "bf16", (count, width)))
    pairs = [source for source in plan.geometry.kv_sources if plan.owner(source) == role and plan.geometry.compress_ratios[source] == 2]
    for source in pairs:
        records.append(Record(f"pair.{source}", "fp32", (2, 1024)))
    checkpoint = metadata["prefix"]
    if checkpoint is not None:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"offer", "windows", "pairs"}:
            raise ValidationError("unknown snapshot encoder checkpoint fields")
        offer = checkpoint["offer"]
        if not isinstance(offer, dict) or set(offer) != {"abi", "identity", "end", "history", "checkpoint"}:
            raise ValidationError("unknown snapshot encoder checkpoint identity")
        end = integer(offer["end"], "snapshot encoder checkpoint", 64, metadata["timeline"]["encoder_tokens"] - 1)
        if end % 64 or offer["identity"] != identity or type(offer["abi"]) is not int or offer["abi"] != 1:
            raise ValidationError("snapshot encoder checkpoint changes the original chunk/precision identity")
        hex_id(offer["history"], 64, "snapshot encoder prefix")
        expected = {key: offer[key] for key in ("abi", "identity", "end", "history")}
        if fingerprint(expected) != offer["checkpoint"]:
            raise ValidationError("snapshot encoder checkpoint fingerprint differs")
        begin, finish = plan.bounds(role)
        windows = {str(layer) for layer in range(begin, min(finish, 21))}
        if not isinstance(checkpoint["windows"], dict) or set(checkpoint["windows"]) != windows or checkpoint["pairs"] != pairs:
            raise ValidationError("snapshot encoder checkpoint state inventory differs from owned layers")
        for layer in sorted(windows, key=int):
            logical = checkpoint["windows"][layer]
            expected = list(range(max(0, end - 128), end)) if int(layer) < 20 else []
            if logical != expected or any(type(v) is not int for v in logical):
                raise ValidationError("snapshot encoder SWA logical validity is incomplete")
            records.append(Record(f"prefix.windows.{layer}", "bf16", (len(logical), 512)))
        for source in pairs:
            records.append(Record(f"prefix.pair.{source}", "fp32", (2, 1024)))
    if not isinstance(metadata["assets"], list) or len(metadata["assets"]) > 4096:
        raise ValidationError("snapshot original image asset inventory exceeds its bound")
    if role == Role.HOST:
        keys = {span.key for span in images}
        observed = set()
        for value in metadata["assets"]:
            if not isinstance(value, dict) or set(value) != {"info", "feature_sha256"}:
                raise ValidationError("snapshot image asset metadata is incomplete")
            info = ImageInfo.from_wire(value["info"])
            hex_id(value["feature_sha256"], 64, "snapshot exact BF16 vision features")
            if info.identity != identity or info.key not in keys or info.key in observed:
                raise ValidationError("snapshot image assets are duplicated, foreign or unrelated to the history")
            observed.add(info.key)
            if any(span.length != info.span_tokens for span in images if span.key == info.key):
                raise ValidationError("snapshot original image span geometry changed")
            records.extend([Record(f"image.{info.key}.original", "u8", (info.original_bytes,)),
                            Record(f"image.{info.key}.features", "bf16", (info.span_tokens, 5120))])
        if observed != keys:
            raise ValidationError("snapshot omits an original image or its exact vision features")
    elif metadata["assets"]:
        raise ValidationError("only A stores original images and exact vision features")
    if role != Role.TERMINAL and (metadata["rng"] is not None or metadata["stops"] is not None):
        raise ValidationError("only C owns terminal sampling and stop-prefix state")
    if role == Role.TERMINAL:
        from .sampling import RandomStreams
        from .detokenize import StopTracker
        RandomStreams(settings.seed).restore(metadata["rng"])
        StopTracker.validate_snapshot(metadata["stops"], settings.stop)
    return tuple(records), settings, tuple(images)


class StageSnapshot:
    """A sealed real stage state; serialization runs on its model actor."""
    def __init__(self, service, identifier):
        from .stage_service import StageService
        if not isinstance(service, StageService):
            raise ValidationError("state snapshots require the actual persistent native stage service")
        execution, stage = service.execution, service.execution.stage
        ledger = execution.ledger
        if (ledger.phase != "decode" or not ledger.quiescent or service._sampling is not None or
                execution._window is not None or execution.history.digest != ledger.history_digest or
                service._settings is None):
            raise ValidationError("snapshot save requires a globally committed idle target boundary")
        self.service, self.id = service, hex_id(identifier, 32, "state snapshot")
        self._resources = ExitStack()
        self.closed = False
        try:
            checkpoint = execution.prefix
            prefix = None
            if checkpoint.offer is not None:
                prefix = {"offer": checkpoint.offer, "windows": {
                    str(layer): sorted(int(v) for v in logical if v >= 0) for layer, (logical, _) in checkpoint._windows.items()},
                    "pairs": sorted(checkpoint._pairs)}
            self.metadata = {"abi": 1, "stage": int(stage.role), "identity": stage.identity, "run": service.data.run,
                "timeline": timeline(service), "settings": service._settings.as_dict(),
                "images": [asdict(span) for span in execution.history.images], "assets": [], "prefix": prefix,
                "rng": service._sampler.random.snapshot() if stage.role == Role.TERMINAL else None,
                "stops": service._stops.snapshot() if stage.role == Role.TERMINAL else None}
            self.assets = service.image_rows.assets if stage.role == Role.HOST and service.image_rows is not None else None
            if execution.history.images:
                if stage.role == Role.HOST:
                    if self.assets is None:
                        raise ValidationError("snapshot cannot omit the request's original images")
                    keys = sorted({span.key for span in execution.history.images})
                    self._resources.enter_context(self.assets.pin(keys))
                    for key in keys:
                        info, digest, size = self.assets._row(key)
                        if digest is None or size != info.span_tokens * 10240:
                            raise ValidationError("snapshot requires all exact vision features before saving")
                        self.metadata["assets"].append({"info": asdict(info), "feature_sha256": digest})
            self.records, _, _ = state_inventory(stage.plan, stage.role, self.metadata, identity=stage.identity, run=service.data.run)
            self.header = encode_header(self.metadata, self.records)
            self.total = len(self.header) + sum(record.nbytes for record in self.records)
            self.header_sha256 = hashlib.sha256(self.header).hexdigest()
            stage.stream.synchronize()
        except BaseException:
            self._resources.close()
            raise

    def offer(self):
        if self.closed:
            raise ValidationError("snapshot state seal is closed")
        return {"id": self.id, "bytes": self.total, "header_sha256": self.header_sha256,
                "timeline": self.metadata["timeline"], "settings_digest": self.service._settings.digest}

    def _array(self, array):
        view = memoryview(np.ascontiguousarray(array)).cast("B")
        for start in range(0, len(view), MAX_PAYLOAD):
            yield bytes(view[start:start + MAX_PAYLOAD])

    def _gpu(self, data, logical, *, offset, slots):
        import torch
        width = data.shape[-1]
        step = min(1024, MAX_PAYLOAD // (width * data.element_size()))
        for start in range(0, len(logical), step):
            rows = logical[start:start + step]
            indices = torch.tensor([(int(v) + offset) % slots for v in rows], dtype=torch.int64, device=data.device)
            selected = data.index_select(0, indices).contiguous()
            if not bool(torch.isfinite(selected).all()):
                raise ValidationError("snapshot encountered nonfinite live state")
            dtype = torch.uint16 if data.dtype == torch.bfloat16 else torch.float32
            yield selected.view(dtype).cpu().numpy().tobytes()

    def _file(self, path, count, digest):
        hasher = hashlib.sha256()
        with locked_read(path) as stream:
            for data in file_chunks(stream, count):
                hasher.update(data)
                yield data
            if stream.read(1) or hasher.hexdigest() != digest:
                raise ValidationError("snapshot original image/feature source checksum differs")

    def chunks(self):
        import torch
        if self.closed:
            raise ValidationError("snapshot state is no longer sealed")
        execution, state = self.service.execution, self.service.execution.stage.attention
        readers = {"history.tokens": lambda: self._array(execution.history._tokens[:execution.history.length]),
                   "history.mask": lambda: self._array(execution.history._text[:execution.history.length].view(np.uint8))}
        for kind, layer, first, count, width in pool_ranges(state.plan, state.role, execution.ledger.processed):
            pool = getattr(state, kind)[layer]
            logical = range(first, first + count)
            for start in range(0, count, 8192):
                pool.physical(logical[start:start + 8192])
            readers[f"{kind}.{layer}"] = lambda p=pool, rows=logical: self._gpu(p.data, rows, offset=p.map.offset, slots=p.map.slots)
        for layer, compressor in state._compressors.items():
            if compressor.position != execution.ledger.processed or compressor.active is not None:
                raise ValidationError("snapshot compressor position differs from the processed cursor")
            if compressor._ring is not None:
                readers[f"pair.{layer}"] = lambda c=compressor: self._array(c._ring.cpu().numpy())
        checkpoint = execution.prefix
        if self.metadata["prefix"] is not None:
            for layer, (logical, data) in checkpoint._windows.items():
                rows = self.metadata["prefix"]["windows"][str(layer)]
                pool = state.windows[layer]
                readers[f"prefix.windows.{layer}"] = lambda d=data, r=rows, p=pool: self._gpu(d, r, offset=p.map.offset, slots=p.map.slots)
            for layer, data in checkpoint._pairs.items():
                readers[f"prefix.pair.{layer}"] = lambda d=data: self._array(d.cpu().numpy())
        for value in self.metadata["assets"]:
            info = ImageInfo.from_wire(value["info"])
            readers[f"image.{info.key}.original"] = lambda i=info: self._file(self.assets._path(i.key, "img"), i.original_bytes, i.original_sha256)
            readers[f"image.{info.key}.features"] = lambda i=info, digest=value["feature_sha256"]: self._file(
                self.assets._path(i.key, "bf16"), i.span_tokens * 10240, digest)
        yield from archive_chunks(self.header, self.records, readers)

    def close(self):
        if not self.closed:
            self.closed = True
            self._resources.close()
