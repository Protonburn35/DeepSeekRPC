"""Exact official October n-gram hashing with bounded transactional validity.

Hashing is CPU work on A. This does not execute Engram's learned projection/gate
or replace the whole model's distributed transaction/coordinator.
"""
from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace

import numpy as np

from .jsonutil import ValidationError, integer
from .protocol import hex_id
from .tokenizer import TokenizerAdapter


class EngramHistory:
    def __init__(self, config: dict, tokenizer: TokenizerAdapter, *, expected_map_sha256: str,
                 context: int = 1048576, max_rows: int = 64):
        import torch
        from ._vendor.deepseek_1017.engram import (
            EngramLayout, NgramHashState, build_compressed_token_map,
        )
        c = config.get("text_config", config)
        self.capacity = integer(context, "Engram context", 1, c["max_position_embeddings"])
        self.max_rows = integer(max_rows, "Engram chunk rows", 1, 4096)
        self.position = self.generation = 0
        self._pending = None
        self._mutex = threading.RLock()
        hex_id(expected_map_sha256, 64, "compressed token map")
        mapping, count = build_compressed_token_map(tokenizer)
        digest = hashlib.sha256(np.asarray(mapping, dtype="<i8").tobytes()).hexdigest()
        if digest != expected_map_sha256 or count != c["engram_compressed_vocab_size"]:
            raise ValidationError("compressed token map fingerprint or vocabulary mismatch")
        self.map_sha256, self.vocab_size = digest, len(tokenizer)
        fields = {name: c[name] for name in (
            "engram_layer_ids", "engram_num_embeddings", "engram_max_ngram_size",
            "engram_vocab_size", "engram_n_heads", "engram_head_dim", "engram_compressed_vocab_size",
        )}
        args = SimpleNamespace(**fields, engram_pad_id=c["engram_pad_token_id"],
                               max_batch_size=1, max_seq_len=context)
        # Do not inherit a caller's default CUDA device for CPU hash history.
        with torch.device("cpu"):
            layout = EngramLayout.from_args(args)
            self._state = NgramHashState(args, layout, tokenizer)
            self._state.cache.zero_()
        self.persistent_history_bytes = self._state.cache.numel() * self._state.cache.element_size()

    def begin(self, tokens: list[int], text_mask: list[bool] | None = None) -> np.ndarray:
        import torch
        with self._mutex:
            if self._pending is not None:
                raise ValidationError("Engram hash transaction is already pending")
            integer(len(tokens), "Engram token rows", 1, self.max_rows)
            if self.position + len(tokens) > self.capacity:
                raise ValidationError("Engram context exceeded; no history was truncated")
            for token in tokens:
                integer(token, "token ID", 0, self.vocab_size - 1)
            if text_mask is not None and (len(text_mask) != len(tokens) or any(type(v) is not bool for v in text_mask)):
                raise ValidationError("Engram text/image mask must match token rows")
            ids = torch.tensor([tokens], dtype=torch.int64, device="cpu")
            mask = None if text_mask is None else torch.tensor([text_mask], dtype=torch.bool, device="cpu")
            result = self._state(ids, self.position, mask).squeeze(0).numpy()
            self._pending = len(tokens)
            result.setflags(write=False)
            return result

    def commit(self, accepted: int):
        with self._mutex:
            if self._pending is None:
                raise ValidationError("no Engram hash transaction is pending")
            integer(accepted, "accepted hash rows", 0, self._pending)
            # Tentative suffix cache bytes stay invalid. Future begin overwrites
            # them before lookback; no committed prefix row was overwritten.
            self.position += accepted
            self._pending = None
            return self.position

    def replay(self, tokens: list[int], *, start: int, text_mask: list[bool] | None = None) -> np.ndarray:
        """Recompute committed hash rows for the second exact encoder pass.

        The phase scheduler must separately bind the immutable full token/image
        identity. Compressed token-map equality is only an Engram-state check;
        different token IDs can share that compressed map entry.
        """
        import torch
        with self._mutex:
            if self._pending is not None:
                raise ValidationError("Engram replay requires no tentative hash transaction")
            integer(start, "Engram replay position", 0, self.position)
            integer(len(tokens), "Engram replay rows", 1, self.max_rows)
            if start + len(tokens) > self.position:
                raise ValidationError("Engram replay exceeds committed history")
            for token in tokens:
                integer(token, "token ID", 0, self.vocab_size - 1)
            if text_mask is not None and (len(text_mask) != len(tokens) or any(type(v) is not bool for v in text_mask)):
                raise ValidationError("Engram replay text/image mask must match token rows")
            ids = torch.tensor([tokens], dtype=torch.int64, device="cpu")
            mask = None if text_mask is None else torch.tensor([text_mask], dtype=torch.bool, device="cpu")
            compressed = self._state.token_map[ids]
            if mask is not None:
                compressed = torch.where(mask, compressed, self._state.DEAD)
            if not torch.equal(compressed, self._state.cache[:, start:start + len(tokens)]):
                raise ValidationError("Engram replay would change committed hash history")
            # The unchanged original operation rewrites exactly these same
            # compressed values, then reads the original absolute lookback.
            result = self._state(ids, start, mask).squeeze(0).numpy()
            result.setflags(write=False)
            return result

    def abort(self):
        return self.commit(0)

    def reset(self):
        with self._mutex:
            self.position = 0
            self._pending = None
            self.generation += 1
            # Validity is empty; subsequent begin never reads uninitialized tail.
