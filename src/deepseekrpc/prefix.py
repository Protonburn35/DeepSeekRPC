"""Exact CED encoder checkpoints; full target replay always remains required."""
import numpy as np

from .history import TokenHistory
from .jsonutil import ValidationError, fingerprint


def checkpoint_anchor(history: TokenHistory):
    """Keep a complete 64-row boundary before the final prompt chunk/image."""
    if not isinstance(history, TokenHistory) or not history.length:
        raise ValidationError("encoder checkpoint requires the complete immutable prompt")
    anchor = ((history.length - 1) // 64) * 64
    for image in reversed(history.images):
        if image.start < anchor < image.start + image.length:
            anchor = (image.start // 64) * 64
    return anchor


def prefix_offer(identity, history, end):
    if end <= 0 or end % 64 or end >= history.length or identity != history.identity:
        raise ValidationError("encoder prefix must leave at least one original prompt chunk to recompute")
    value = {"abi": 1, "identity": identity, "end": end, "history": history.prefix_digest(end)}
    return {**value, "checkpoint": fingerprint(value)}


def select_prefix(offers, prompt, *, features_available):
    """Unanimous exact identity; a miss never changes one rank in isolation."""
    if not isinstance(offers, list) or len(offers) != 3:
        raise ValidationError("prefix reuse requires all three actual loaded-stage offers")
    offer = offers[0]
    if offer is None or any(other != offer for other in offers):
        return None, "no unanimous encoder checkpoint"
    if not isinstance(offer, dict) or set(offer) != {"abi", "identity", "end", "history", "checkpoint"}:
        raise ValidationError("unknown encoder checkpoint metadata")
    end = offer["end"]
    if type(end) is not int or not 0 < end < prompt.length or end % 64:
        return None, "checkpoint does not precede the new prompt tail"
    try:
        expected = prefix_offer(prompt.identity, prompt, end)
    except ValidationError:
        return None, "checkpoint falls inside a changed image span"
    if offer != expected:
        return None, "token/image/component prefix changed"
    if not features_available(end):
        return None, "original exact image features are not retained"
    return dict(offer), "exact encoder checkpoint; full target replay required"


class EncoderCheckpoint:
    """Only SWA/pair before-images are copied; large source pools stay owned.

    Encoder compressed rows are immutable across the later target replay, and
    decode only appends after them. A fault invalidates this proof. Retention
    remaps valid prefix generations under a newer session epoch without a
    second multi-GiB allocation.
    """
    def __init__(self, execution):
        from .execution import StageExecution
        if not isinstance(execution, StageExecution):
            raise ValidationError("encoder checkpoint requires the actual stage execution owner")
        self.execution = execution
        self._windows, self._pairs = {}, {}
        self._offer, self._session, self._epoch = None, None, -1

    @property
    def offer(self):
        return None if self._offer is None else dict(self._offer)

    def invalidate(self):
        self._offer = None
        self._session, self._epoch = None, -1

    def capture(self):
        stage, ledger, history = self.execution.stage, self.execution.ledger, self.execution.history
        state, end = stage.attention, ledger.encoder_cursor
        if (ledger.phase != "encoder" or end != checkpoint_anchor(history) or not end or
                ledger._active is not None or ledger._awaiting is not None or state.active_operation is not None):
            raise ValidationError("encoder checkpoint requires its sealed complete-boundary phase")
        offer = prefix_offer(stage.identity, history, end)
        for layer, pool in state.windows.items():
            if layer > 20:
                continue
            if layer not in self._windows:
                self._windows[layer] = (np.empty(128, dtype=np.int64), pool.data.clone())
            logical, data = self._windows[layer]
            logical[:] = pool.map.logical
            data.copy_(pool.data)
        for layer, compressor in state._compressors.items():
            if compressor.active is not None or compressor.position != end:
                raise ValidationError("encoder checkpoint has an incomplete compressor phase")
            if compressor._ring is not None:
                if layer not in self._pairs:
                    self._pairs[layer] = compressor._ring.clone()
                else:
                    self._pairs[layer].copy_(compressor._ring)
        allocated = sum(data.numel() * data.element_size() for _, data in self._windows.values())
        allocated += sum(data.numel() * data.element_size() for data in self._pairs.values())
        if allocated > stage.memory.gpu["prefix_and_request_checkpoints"]:
            self.invalidate()
            raise ValidationError("bounded encoder checkpoint exceeded its admitted memory")
        stage.stream.synchronize()
        self._offer, self._session, self._epoch = offer, ledger.session, ledger.epoch
        return self.offer

    def validate(self, offer, incoming):
        state = self.execution.stage.attention
        if (self._offer is None or offer != self._offer or
                prefix_offer(incoming.identity, incoming, offer["end"]) != offer or
                state._failed or state.active_operation is not None):
            raise ValidationError("encoder reuse differs from the retained exact checkpoint")
        for pool in state._pools:
            if pool.map.session != self._session or pool.map.epoch != self._epoch or pool.map.active is not None:
                raise ValidationError("encoder checkpoint source generations were invalidated")
        for source, pool in state.sources.items():
            count = offer["end"] // state.plan.geometry.compress_ratios[source]
            pool.map.validate_prefix(count)
            state.index_keys[source].map.validate_prefix(count)

    def restore(self, offer, incoming, *, session, epoch):
        self.validate(offer, incoming)
        stage, state, end = self.execution.stage, self.execution.stage.attention, offer["end"]
        for source, pool in state.sources.items():
            count = end // state.plan.geometry.compress_ratios[source]
            pool.retain_prefix(session, epoch, count)
            state.index_keys[source].retain_prefix(session, epoch, count)
        for layer, pool in state.windows.items():
            if layer in self._windows:
                logical, data = self._windows[layer]
                pool.restore_ring(session, epoch, logical, data)
            else:
                pool.reset(session, epoch)
        for layer, compressor in state._compressors.items():
            compressor.reset(session, epoch)
            if compressor._ring is not None:
                compressor._ring.copy_(self._pairs[layer])
            compressor.position = end
        state._selections.clear()
        if stage.engram_history is not None:
            stage.engram_history.reset()
            for start in range(0, end, 64):
                if stage.cancellation.is_set():
                    raise InterruptedError("encoder prefix hash reconstruction cancelled")
                tokens, mask = incoming.rows(start, min(64, end - start))
                stage.engram_history.begin(list(tokens), list(mask))
                stage.engram_history.commit(len(tokens))
        stage.stream.synchronize()
        self._session, self._epoch = session, epoch
        return end
