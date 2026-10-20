"""Two-pass bounded restore: validate all streams, then mutate owned GPU state."""
from contextlib import ExitStack
import hashlib

import numpy as np

from .detokenize import StopTracker, TokenBytes
from .history import TokenHistory
from .jsonutil import ValidationError, integer
from .planner import Role
from .prefix import prefix_offer
from .protocol import MAX_PAYLOAD, hex_id
from .publication import PublicationLedger
from .snapshot_archive import MAX_ARCHIVE, file_chunks, read_header
from .snapshot_state import pool_ranges, state_inventory
from .snapshot_wire import TileReader
from .terminal_sampling import TerminalSampler
from .vision_input import ImageInfo


class StageRestore:
    def __init__(self, service, identifier, *, total, sha256, header_sha256):
        from .stage_service import StageService
        if not isinstance(service, StageService):
            raise ValidationError("restore requires the actual persistent native stage service")
        self.service, self.id = service, hex_id(identifier, 32, "snapshot restore")
        self.total = integer(total, "restored state stream length", 1, MAX_ARCHIVE)
        self.sha256 = hex_id(sha256, 64, "restored state complete checksum")
        self.header_sha256 = hex_id(header_sha256, 64, "restored state inventory checksum")
        self._resources, self.closed = ExitStack(), False
        self.validated = self.applied = False
        self.metadata = self.records = self.header = self.history = self.settings = self.sampler = self.stops = None

    def _check_cancel(self):
        if self.closed or self.service.execution.stage.cancellation.is_set():
            raise InterruptedError("snapshot restore cancelled")

    def _checked(self, chunks):
        digest, count = hashlib.sha256(), 0
        for data in chunks:
            self._check_cancel()
            if not isinstance(data, bytes) or not 0 < len(data) <= MAX_PAYLOAD:
                raise ValidationError("restore source exceeded its bounded tile size")
            count += len(data)
            if count > self.total:
                raise ValidationError("restore source exceeds its admitted archive length")
            digest.update(data)
            yield data
        if count != self.total or digest.hexdigest() != self.sha256:
            raise ValidationError("restore source differs from the complete prevalidated archive checksum")

    @staticmethod
    def _finite(record, data):
        if record.dtype == "bf16":
            bits = np.frombuffer(data, dtype="<u2")
            if np.any((bits & 0x7f80) == 0x7f80):
                raise ValidationError("snapshot contains nonfinite BF16 cache/features")
        elif record.dtype == "fp32" and not np.isfinite(np.frombuffer(data, dtype="<f4")).all():
            raise ValidationError("snapshot contains nonfinite FP32 compressor state")

    def validate(self, chunks):
        if self.validated:
            raise ValidationError("restore validation is already sealed")
        service, stage = self.service, self.service.execution.stage
        reader = TileReader(self._checked(chunks))
        metadata, records, header, _ = read_header(reader, expected_bytes=self.total)
        if hashlib.sha256(header).hexdigest() != self.header_sha256:
            raise ValidationError("restore header changed after host validation")
        expected, settings, images = state_inventory(stage.plan, stage.role, metadata, identity=stage.identity, run=service.data.run)
        if records != expected:
            raise ValidationError("restore record shapes/types differ from the locally derived state inventory")
        if stage.role == Role.TERMINAL:
            self.sampler = TerminalSampler(settings.seed)
            self.sampler.random.restore(metadata["rng"])
            self.stops = StopTracker(TokenBytes(stage.tokenizer), settings.stop)
            self.stops.restore(metadata["stops"])
        length = records[0].shape[0]
        tokens, mask = np.empty(length, dtype=np.int32), np.empty(length, dtype=np.uint8)
        history = TokenHistory(stage.identity)
        assets = service.image_rows.assets if stage.role == Role.HOST and service.image_rows is not None else None
        image_metadata = {value["info"]["key"]: value for value in metadata["assets"]}
        for record in records:
            self._check_cancel()
            offset, image_data = 0, None
            if record.name.startswith("image."):
                image_data = bytearray()
            for data in file_chunks(reader, record.nbytes):
                self._check_cancel()
                self._finite(record, data)
                if record.name == "history.tokens":
                    values = np.frombuffer(data, dtype="<i4")
                    tokens[offset // 4:(offset + len(data)) // 4] = values
                elif record.name == "history.mask":
                    values = np.frombuffer(data, dtype=np.uint8)
                    if np.any(values > 1):
                        raise ValidationError("snapshot image mask is not lossless boolean state")
                    mask[offset:offset + len(data)] = values
                elif image_data is not None:
                    image_data.extend(data)  # One <=32 MiB image or <=10 MiB feature asset.
                offset += len(data)
            if record.name == "history.mask":
                value = metadata["timeline"]
                if history.set_prompt(tokens, mask.view(np.bool_), images) != value["history"]:
                    raise ValidationError("snapshot exact token/image history checksum differs")
                if value["pending_token"] is not None and tokens[-1] != value["pending_token"]:
                    raise ValidationError("snapshot pending input differs from its published last token")
                generated = tokens[value["encoder_tokens"]:]
                history._generated_counts[:] = np.bincount(generated, minlength=129280).astype(np.int32)
                if metadata["prefix"] is not None:
                    offer = metadata["prefix"]["offer"]
                    if prefix_offer(stage.identity, history, offer["end"]) != offer:
                        raise ValidationError("snapshot encoder checkpoint differs from exact original token/image history")
            if image_data is not None:
                if assets is None:
                    raise ValidationError("A cannot restore images without its real owned image store/resolver")
                _, key, kind = record.name.split(".")
                info = ImageInfo.from_wire(image_metadata[key]["info"])
                if kind == "original":
                    if hashlib.sha256(image_data).hexdigest() != info.original_sha256 or assets.ingest(bytes(image_data)) != info:
                        raise ValidationError("snapshot original image bytes/processor/grid changed")
                    self._resources.enter_context(assets.pin([key]))
                else:
                    if hashlib.sha256(image_data).hexdigest() != image_metadata[key]["feature_sha256"]:
                        raise ValidationError("snapshot exact vision features failed their original checksum")
                    assets.store_features(key, bytes(image_data))
        reader.finish()  # Exhausts the checksum/deadline validator before sealing.
        self.metadata, self.records, self.header, self.history, self.settings = metadata, records, header, history, settings
        self.validated = True
        return {"validated": True, "id": self.id, "sha256": self.sha256, "timeline": metadata["timeline"]}

    def apply(self, chunks, *, session, request, epoch):
        import torch
        if not self.validated or self.applied or self.closed:
            raise ValidationError("restore apply requires a complete sealed first validation pass")
        service, execution = self.service, self.service.execution
        stage, state = execution.stage, execution.stage.attention
        hex_id(session, 32, "restored new session")
        hex_id(request, 32, "restored new request")
        integer(epoch, "restored new epoch", max(execution.ledger.epoch + 1, state.next_reset_epoch,
                self.metadata["timeline"]["epoch"] + 1), (1 << 63) - 1)
        reader = TileReader(self._checked(chunks))
        metadata, records, header, _ = read_header(reader, expected_bytes=self.total)
        if header != self.header or records != self.records:
            raise ValidationError("restore apply differs from its sealed first-pass inventory")
        stage.stream.synchronize()
        execution.prefix.invalidate()
        if state._failed:
            state.discard_session(session, epoch)
        else:
            state.reset(session, epoch)
        pools = {f"{kind}.{layer}": (getattr(state, kind)[layer], first, count, width)
                 for kind, layer, first, count, width in pool_ranges(stage.plan, stage.role, metadata["timeline"]["processed"])}
        checkpoint = execution.prefix
        for record in records:
            self._check_cancel()
            if record.name in pools:
                pool, first, count, width = pools[record.name]
                for start in range(0, count, 1024):
                    self._check_cancel()
                    rows = min(1024, count - start)
                    data = reader.read(rows * width * 2)
                    if len(data) != rows * width * 2:
                        raise ValidationError("restore cache record is truncated")
                    values = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(rows, width).to(state.device)
                    pool.restore_rows(list(range(first + start, first + start + rows)), values)
            elif record.name.startswith("pair.") or record.name.startswith("prefix.pair."):
                data = reader.read(record.nbytes)
                if len(data) != record.nbytes:
                    raise ValidationError("restore FP32 compressor record is truncated")
                values = torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(1, 2, 1024).to(state.device)
                layer = int(record.name.rsplit(".", 1)[1])
                if record.name.startswith("prefix."):
                    if layer not in checkpoint._pairs:
                        checkpoint._pairs[layer] = values.clone()
                    else:
                        checkpoint._pairs[layer].copy_(values)
                else:
                    state._compressors[layer]._ring.copy_(values)
            elif record.name.startswith("prefix.windows."):
                layer = int(record.name.rsplit(".", 1)[1])
                logical = metadata["prefix"]["windows"][str(layer)]
                if layer not in checkpoint._windows:
                    checkpoint._windows[layer] = (np.full(128, -1, dtype=np.int64),
                        torch.zeros((128, 512), dtype=torch.bfloat16, device=state.device))
                mapping, target = checkpoint._windows[layer]
                mapping.fill(-1)
                target.zero_()
                data = reader.read(record.nbytes)
                if len(data) != record.nbytes:
                    raise ValidationError("restore encoder checkpoint is truncated")
                if logical:
                    slots = [(value + state.windows[layer].map.offset) % 128 for value in logical]
                    mapping[slots] = logical
                    values = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(len(logical), 512).to(state.device)
                    target.index_copy_(0, torch.tensor(slots, dtype=torch.int64, device=state.device), values)
            else:
                # Tokens/masks and images were validated/imported on the first
                # pass. Consume their identical bytes under the complete hash.
                for data in file_chunks(reader, record.nbytes):
                    self._check_cancel()
        reader.finish()
        value = metadata["timeline"]
        for compressor in state._compressors.values():
            compressor.position = value["processed"]
        if stage.engram_history is not None:
            stage.engram_history.reset()
            for start in range(0, value["processed"], 64):
                self._check_cancel()
                tokens, mask = self.history.rows(start, min(64, value["processed"] - start))
                stage.engram_history.begin(list(tokens), list(mask))
                stage.engram_history.commit(len(tokens))
        execution.ledger.restore_committed(value, session=session, request=request, epoch=epoch, history_digest=self.history.digest)
        execution.history = self.history
        execution._window = execution._inputs = execution._result = execution._prepared = execution._hashes = execution._last_logits = None
        execution._hash_pending = execution._forwarded = False
        execution._imported_sources.clear()
        execution._imported_selections.clear()
        service._settings, service._sampler, service._stops, service._sampling = self.settings, self.sampler, self.stops, None
        service._outputs = PublicationLedger(execution.ledger, execution.history,
            **({"validate_sampling": service._validate_sampling, "finalize_sampling": service._finalize_sampling}
               if stage.role == Role.TERMINAL else {}))
        if metadata["prefix"] is not None:
            checkpoint._offer = dict(metadata["prefix"]["offer"])
            checkpoint._session, checkpoint._epoch = session, epoch
        if stage.role == Role.MIDDLE:
            stage.weights.revalidate_residency()
        stage.stream.synchronize()
        stage.guard.check()
        service._ready = False
        self.applied = True
        return {"applied": True, "id": self.id, "session": session, "request": request, "epoch": epoch,
                "history": self.history.digest, "processed": value["processed"], "pending_token": value["pending_token"],
                "settings_digest": self.settings.digest, "next_operation": value["next_operation"]}

    def close(self):
        if not self.closed:
            self.closed = True
            self._resources.close()
