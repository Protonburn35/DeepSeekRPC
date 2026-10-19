"""Bounded exact encoded-row access to the original October Engram tables.

This provides storage, not the complete Engram hash/project/gate operation. It
never allocates a full table and never substitutes a missing row. Decoding below
is a separately tested FP32 diagnostic baseline; the model adapter must preserve
the selected source computation's explicit BF16/FP16 conversion policy.
"""
from collections import OrderedDict
from pathlib import Path
import threading

import numpy as np

from .inventory import TensorFile
from .jsonutil import ValidationError, integer

ROW_WIDTH = 256
SCALE_WIDTH = 8
ENCODED_BYTES = ROW_WIDTH + SCALE_WIDTH
MAX_BATCH = 1024


class EngramRows:
    def __init__(self, path: Path, *, layer: int, cache_bytes: int):
        if layer not in (1, 14):
            raise ValidationError("only the pinned October Engram layers are supported")
        integer(cache_bytes, "Engram cache budget", 0, 1 << 40)
        self._file = TensorFile(path)
        self.weight = f"layers.{layer}.engram.embed.weight"
        self.scale = f"layers.{layer}.engram.embed.scale"
        try:
            w, s = self._file.tensors[self.weight], self._file.tensors[self.scale]
            if (w.dtype != "F8_E4M3" or s.dtype != "F8_E8M0" or len(w.shape) != 2 or
                    w.shape[1] != ROW_WIDTH or s.shape != (w.shape[0], SCALE_WIDTH)):
                raise ValidationError("Engram table format differs from the pinned FP8/UE8M0 layout")
            self.rows = w.shape[0]
        except BaseException:
            self._file.close()
            raise
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        # Explicit payload + conservative Python-entry reservation; no table mirror.
        self.max_cached_rows = cache_bytes // (ENCODED_BYTES + 256)
        self._mutex = threading.Lock()
        self.closed = False
        self.disk_row_reads = self.cache_hits = 0

    def gather_encoded(self, ids) -> np.ndarray:
        ids = np.asarray(ids)
        if ids.ndim != 1 or len(ids) > MAX_BATCH or ids.dtype not in (np.dtype("int32"), np.dtype("int64")):
            raise ValidationError("Engram row requests require a bounded 1D integer array")
        if np.any(ids < 0) or np.any(ids >= self.rows):
            raise ValidationError("Engram row index outside table")
        result = np.empty((len(ids), ENCODED_BYTES), dtype=np.uint8)
        with self._mutex:
            if self.closed:
                raise ValidationError("Engram row store is closed")
            for index, requested in enumerate(ids):
                row = int(requested)
                data = self._cache.get(row)
                if data is None:
                    data = (self._file.read_range(self.weight, row * ROW_WIDTH, ROW_WIDTH)
                            + self._file.read_range(self.scale, row * SCALE_WIDTH, SCALE_WIDTH))
                    self.disk_row_reads += 1
                    if self.max_cached_rows:
                        while len(self._cache) >= self.max_cached_rows:
                            self._cache.popitem(last=False)
                        self._cache[row] = data
                else:
                    self._cache.move_to_end(row)
                    self.cache_hits += 1
                result[index] = np.frombuffer(data, dtype=np.uint8)
        return result

    def close(self):
        with self._mutex:
            if not self.closed:
                self._file.close()
                self._cache.clear()
                self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def decode_fp8_rows(encoded: np.ndarray) -> np.ndarray:
    if encoded.dtype != np.uint8 or encoded.ndim != 2 or encoded.shape[1] != ENCODED_BYTES:
        raise ValidationError("encoded Engram row layout mismatch")
    values = encoded[:, :ROW_WIDTH]
    exponent = ((values >> 3) & 15).astype(np.int32)
    mantissa = (values & 7).astype(np.float32)
    magnitude = np.where(exponent == 0, np.ldexp(mantissa, -9),
                         np.ldexp(1 + mantissa / 8, exponent - 7)).astype(np.float32)
    magnitude = np.where((values & 128) != 0, -magnitude, magnitude)
    scales = encoded[:, ROW_WIDTH:]
    if np.any((exponent == 15) & (mantissa == 7)) or np.any(scales == 255):
        raise ValidationError("Engram row contains NaN encoding")
    scale_values = np.ldexp(np.ones(scales.shape, dtype=np.float32), scales.astype(np.int32) - 127)
    output = magnitude.reshape(-1, SCALE_WIDTH, 32) * scale_values[..., None]
    if not np.all(np.isfinite(output)):
        raise ValidationError("Engram dequantization overflow; no implicit precision fallback")
    return output.reshape(-1, ROW_WIDTH)
