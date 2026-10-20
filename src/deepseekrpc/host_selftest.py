"""Bounded native file/Engram/locked-page diagnostics without model payloads."""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from .engram_storage import EngramRows, decode_fp8_rows
from .inventory import TensorFile
from .native_windows import LockedBuffer, mark_sparse


def windows_cpu_features():
    if os.name != "nt":
        return {"status": "not run", "reason": "native Windows feature API is required"}
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    query = dll.IsProcessorFeaturePresent
    query.argtypes, query.restype = [wintypes.DWORD], wintypes.BOOL
    # Windows reports OS-enabled processor features; do not infer usable AVX
    # merely from a CPUID hardware bit or execute unguarded AVX instructions.
    features = {name: bool(query(index)) for name, index in {
        "xsave_enabled": 17, "sse3": 13, "sse4_1": 37,
        "sse4_2": 38, "avx": 39, "avx2": 40, "avx512f": 41}.items()}
    return {"status": "passed", "scope": "Windows IsProcessorFeaturePresent; OS-enabled features",
            "features": features, "engram_dispatch": "exact scalar native CUDA decode; no host AVX512 requirement"}


def run_host_checks():
    report = {"scope": "bounded synthetic files and host memory; no full model or CUDA execution",
              "platform": "Windows" if os.name == "nt" else "portable CPU diagnostic", "checks": [],
              "cpu": windows_cpu_features()}
    with tempfile.TemporaryDirectory(prefix="deepseekrpc-selftest-") as temporary:
        root = Path(temporary)
        # A sparse >4-GiB address exercises real 64-bit seek/read without
        # allocating, reading or downloading a 4-GiB payload.
        payload_size = (1 << 32) + 512
        header = json.dumps({"synthetic.bytes": {"dtype": "U8", "shape": [payload_size],
            "data_offsets": [0, payload_size]}}, separators=(",", ":")).encode()
        path = root / "sparse.safetensors"
        expected = bytes(range(64))
        with path.open("w+b", buffering=0) as stream:
            mark_sparse(stream)
            stream.write(struct.pack("<Q", len(header)) + header)
            stream.seek(8 + len(header) + payload_size - len(expected))
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        with TensorFile(path) as source:
            if source.read_range("synthetic.bytes", payload_size - 64, 64) != expected:
                raise AssertionError("native 64-bit file offset read changed synthetic bytes")
        report["checks"].append({"name": "native file read above 4 GiB", "status": "passed",
            "read_bytes": 64, "scope": "synchronous bounded file API; no asynchronous IO claim"})

        # Deliberately tiny independent encoded rows: +1/-1/0 with unit scale.
        weights = np.array([[0x38] * 256, [0xb8] * 256, [0] * 256], np.uint8)
        scales = np.full((3, 8), 127, np.uint8)
        payload = weights.tobytes() + scales.tobytes()
        header = json.dumps({"layers.1.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [3, 256],
            "data_offsets": [0, weights.nbytes]}, "layers.1.engram.embed.scale": {
            "dtype": "F8_E8M0", "shape": [3, 8], "data_offsets": [weights.nbytes, len(payload)]}}).encode()
        path = root / "engram.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
        with EngramRows(path, layer=1, cache_bytes=2 * (264 + 256)) as rows:
            actual = rows.gather_encoded(np.array([1, 0, 1], np.int64))
            expected = np.concatenate((weights[[1, 0, 1]], scales[[1, 0, 1]]), axis=1)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(decode_fp8_rows(actual), np.array([-1., 1., -1.], np.float32)[:, None].repeat(256, axis=1))
            if rows.disk_row_reads != 2 or rows.cache_hits != 1:
                raise AssertionError("synthetic Engram row cache counters differ")
        report["checks"].append({"name": "Engram original-byte reads, bounded cache and independent +/-1 decode",
            "status": "passed", "row_requests": 3, "storage_row_reads": 2, "cache_hits": 1,
            "encoded_sha256": hashlib.sha256(actual.tobytes()).hexdigest()})
    if os.name == "nt":
        with LockedBuffer(65536, admitted_bytes=65536) as buffer:
            buffer.write(0, bytes(range(256)))
            if buffer.read(0, 256) != bytes(range(256)):
                raise AssertionError("locked private host buffer changed bytes")
            report["locked_memory"] = buffer.verify_locked_pages()
        report["checks"].append({"name": "Win32 private committed host pages: lock, all-page residency and release", "status": "passed"})
    else:
        report["checks"].append({"name": "Win32 locked private host pages", "status": "not run"})
    return report
