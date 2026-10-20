"""Bounded content-free operational counters, timings and opt-in tracing.

Counter names/events are closed inventories. No caller can put prompt text,
paths, token IDs, usernames, image data or arbitrary messages in these logs.
File read counters measure application I/O requests, not physical SSD traffic.
"""
from bisect import bisect_left
from contextlib import contextmanager
from pathlib import Path
import os
import threading
import time

from .jsonutil import ValidationError, canonical, integer

COMPONENTS = frozenset({"rpc", "experts", "cache", "io", "kv", "cuda", "vision", "spec"})
COUNTERS = frozenset({"gpu_bundle_requests", "gpu_bundle_hits", "gpu_bundle_loads", "gpu_evictions",
    "ram_bundle_requests", "ram_bundle_hits", "ram_bundle_loads", "ram_evictions", "routed_token_assignments",
    "routed_unique_bundles", "model_file_read_bytes", "model_file_reads", "integrity_read_bytes",
    "engram_row_requests", "engram_row_hits", "engram_storage_rows", "engram_file_reads", "engram_file_read_bytes",
    "pcie_upload_bytes", "rpc_send_bytes", "rpc_receive_bytes", "rpc_send_frames", "rpc_receive_frames",
    "encoder_new_tokens", "encoder_reused_tokens", "target_replay_tokens", "published_tokens",
    "draft_proposed_tokens", "draft_accepted_tokens", "target_verified_inputs", "rollback_inputs", "target_passes",
    "session_faults", "cancelled_requests", "completed_requests", "paused_requests", "failed_requests", "image_encodes"})
TIMINGS = frozenset({"model_file_read", "integrity_read", "engram_gather", "pcie_upload",
    "rpc_send", "rpc_receive", "rpc_encode", "rpc_decode", "stage_host_forward", "stage_cuda_forward",
    "prefill", "time_to_first_token", "request_wall", "inter_output_batch", "image_encode", "load"})
EVENTS = {"loaded": ("cuda", "info"), "mesh_rebound": ("rpc", "info"),
    "session_fault": ("kv", "error"), "window": ("cuda", "debug"),
    "expert_load": ("experts", "debug"), "file_read": ("io", "debug"),
    "request_complete": ("spec", "info"), "request_failed": ("spec", "warn"),
    "runtime_failed": ("cuda", "error"), "image_encoded": ("vision", "debug"),
    "frame": ("rpc", "debug"), "cache_hit": ("cache", "debug"), "rollback": ("kv", "debug")}
FIELDS = frozenset({"rows", "bytes", "tokens", "new_tokens", "reused_tokens", "replay_tokens",
    "elapsed_ns", "proposed", "accepted", "rolled_back", "operation", "phase_code", "failure_code", "peer", "kind",
    "ttft_ns", "prefill_ns", "decode_ns", "output_batches"})
BUCKETS = (1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000,
           1_000_000_000, 10_000_000_000, 60_000_000_000)
LEVELS = {"trace": 0, "debug": 1, "info": 2, "warn": 3, "error": 4}
PHASES = frozenset({"boot", "diagnostic", "inference", "administration"})


class Metrics:
    def __init__(self, *, root=None, stage=None, log_level="info", traces=(), log_bytes=4 * 1024 ** 2):
        if log_level not in LEVELS or not set(traces) <= COMPONENTS or stage not in {None, "A", "B", "C"}:
            raise ValidationError("invalid bounded diagnostic configuration")
        self._lock = threading.RLock()
        self._start, self._phase = time.perf_counter_ns(), "boot"
        self.stage, self.log_level, self.traces = stage, log_level, frozenset(traces)
        self.log_bytes = integer(log_bytes, "bounded log file bytes", 1024, 4 * 1024 ** 2)
        self.root = Path(root) if root is not None else None
        self._counters, self._timings, self._latest = {}, {}, {}
        self._write_failure = False
        self._stop, self._writer = threading.Event(), None
        if self.root is not None:
            if self.root.is_symlink():
                raise ValidationError("diagnostics directory cannot be redirected")
            self.root.mkdir(exist_ok=True)
            known = {"metrics.json", "metrics.json.partial", "events.jsonl", "events.1.jsonl", "events.2.jsonl"}
            if any(path.name not in known or path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                   for path in self.root.iterdir()):
                raise ValidationError("diagnostics directory contains unrelated or redirected files")

    def set_phase(self, phase):
        if phase not in PHASES:
            raise ValidationError("unknown metrics lifetime phase")
        with self._lock:
            self._phase = phase

    def start(self):
        if self._writer is not None:
            raise ValidationError("metrics writer is already started")
        def write():
            while not self._stop.wait(5):
                self.flush()
        self._writer = threading.Thread(target=write, name="deepseekrpc-metrics", daemon=True)
        self._writer.start()
        return self

    def close(self):
        self._stop.set()
        if self._writer is not None:
            self._writer.join(5)
        self.flush()

    def add(self, name, value=1):
        if name not in COUNTERS:
            raise ValidationError("unknown operational counter")
        integer(value, "counter increment", 0, (1 << 63) - 1)
        with self._lock:
            key = self._phase + "." + name
            self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name, nanoseconds):
        if name not in TIMINGS:
            raise ValidationError("unknown operational timing")
        integer(nanoseconds, "monotonic elapsed nanoseconds", 0, (1 << 63) - 1)
        with self._lock:
            key = self._phase + "." + name
            entry = self._timings.setdefault(key, {"count": 0, "sum_ns": 0, "max_ns": 0, "buckets": [0] * (len(BUCKETS) + 1)})
            entry["count"] += 1
            entry["sum_ns"] += nanoseconds
            entry["max_ns"] = max(entry["max_ns"], nanoseconds)
            entry["buckets"][bisect_left(BUCKETS, nanoseconds)] += 1

    @contextmanager
    def measure(self, name):
        first = time.perf_counter_ns()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter_ns() - first)

    @staticmethod
    def _fields(fields):
        if set(fields) - FIELDS:
            raise ValidationError("diagnostics refuse arbitrary payload fields")
        for value in fields.values():
            integer(value, "content-free diagnostic value", 0, (1 << 63) - 1)

    def latest_request(self, **fields):
        self._fields(fields)
        with self._lock:
            self._latest = dict(fields)

    def event(self, name, **fields):
        if name not in EVENTS:
            raise ValidationError("unknown content-free diagnostic event")
        self._fields(fields)
        component, level = EVENTS[name]
        if LEVELS[level] < LEVELS[self.log_level] and component not in self.traces:
            return
        with self._lock:
            if self.root is None or self._write_failure:
                return
            encoded = canonical({"elapsed_ns": time.perf_counter_ns() - self._start, "stage": self.stage,
                "phase": self._phase, "component": component, "level": level, "event": name, **fields}) + b"\n"
            if len(encoded) > 2048:
                raise ValidationError("bounded event exceeds its fixed numeric field inventory")
            try:
                current = self.root / "events.jsonl"
                if current.is_symlink() or current.exists() and current.stat().st_nlink != 1:
                    raise OSError("redirected diagnostic file")
                if current.exists() and current.stat().st_size + len(encoded) > self.log_bytes:
                    oldest = self.root / "events.2.jsonl"
                    if oldest.exists():
                        oldest.unlink()
                    previous = self.root / "events.1.jsonl"
                    if previous.exists():
                        os.replace(previous, oldest)
                    os.replace(current, previous)
                with current.open("ab", buffering=0) as stream:
                    stream.write(encoded)
            except OSError:
                self._write_failure = True

    def snapshot(self):
        with self._lock:
            ratios = {}
            for phase in PHASES:
                def ratio(hits, requests):
                    denominator = self._counters.get(phase + "." + requests, 0)
                    return self._counters.get(phase + "." + hits, 0) / denominator if denominator else None
                ratios[phase] = {"gpu_bundle_hit_fraction": ratio("gpu_bundle_hits", "gpu_bundle_requests"),
                    "ram_bundle_hit_fraction_on_gpu_miss": ratio("ram_bundle_hits", "ram_bundle_requests"),
                    "engram_row_hit_fraction": ratio("engram_row_hits", "engram_row_requests")}
            return {"format": 1, "stage": self.stage, "elapsed_ns": time.perf_counter_ns() - self._start,
                "phase": self._phase, "counters": dict(self._counters),
                "timings": {key: {**entry, "buckets": list(entry["buckets"])} for key, entry in self._timings.items()},
                "timing_bucket_upper_ns": [*BUCKETS, None], "fractions": ratios,
                "latest_request": dict(self._latest), "log_write_failed": self._write_failure,
                "not_measured": {"physical_ssd_bytes": None, "hard_page_faults": None, "pagefile_io_bytes": None,
                    "thermal_power": None, "read_ahead": "not implemented; no speculative read-ahead is issued"},
                "counter_scope": "application requests; OS file-cache hits and physical SSD reads are not distinguishable"}

    def compact(self):
        """Current local lifetime only; sized for the 16-KiB control budget."""
        with self._lock:
            prefix = self._phase + "."
            return {"phase": self._phase,
                "counters": {key[len(prefix):]: value for key, value in self._counters.items() if key.startswith(prefix)},
                "timings": {key[len(prefix):]: {name: value for name, value in entry.items() if name != "buckets"}
                    for key, entry in self._timings.items() if key.startswith(prefix)},
                "log_write_failed": self._write_failure}

    def flush(self):
        with self._lock:
            if self.root is None or self._write_failure:
                return
            raw = canonical(self.snapshot()) + b"\n"
            if len(raw) > 256 * 1024:
                raise ValidationError("fixed metrics inventory unexpectedly exceeded its bound")
            try:
                partial = self.root / "metrics.json.partial"
                if partial.exists():
                    if partial.is_symlink() or partial.stat().st_nlink != 1:
                        raise OSError("redirected diagnostic temporary file")
                    partial.unlink()
                with partial.open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(partial, self.root / "metrics.json")
            except OSError:
                self._write_failure = True


current = Metrics()


def configure(root, *, stage, log_level="info", traces=()):
    global current
    current = Metrics(root=root, stage=stage, log_level=log_level, traces=traces)
    return current


class RequestMeasurement:
    """Committed outputs and inclusive scheduler wall time, including backpressure."""
    def __init__(self):
        self.metrics, self.started = current, time.perf_counter_ns()
        self.first = self.previous = None
        self.tokens = self.batches = 0
        self.prefill_ns = self.new = self.reused = self.replay = 0

    def prefilled(self, started, inventory):
        self.prefill_ns = time.perf_counter_ns() - started
        self.new, self.reused, self.replay = (inventory[key] for key in
            ("new_encoder_tokens", "reused_encoder_tokens", "target_replay_tokens"))
        self.metrics.add("encoder_new_tokens", self.new)
        self.metrics.add("encoder_reused_tokens", self.reused)
        self.metrics.add("target_replay_tokens", self.replay)
        self.metrics.observe("prefill", self.prefill_ns)

    def batch(self, count):
        now = time.perf_counter_ns()
        if self.first is None:
            self.first = now
            self.metrics.observe("time_to_first_token", now - self.started)
        else:
            self.metrics.observe("inter_output_batch", now - self.previous)
        self.previous = now
        self.tokens += count
        self.batches += 1
        self.metrics.add("published_tokens", count)

    def finish(self, outcome):
        if outcome not in {"completed", "paused", "cancelled", "failed"}:
            raise ValidationError("unknown measured request outcome")
        now = time.perf_counter_ns()
        self.metrics.observe("request_wall", now - self.started)
        self.metrics.add(outcome + "_requests")
        self.metrics.latest_request(tokens=self.tokens, output_batches=self.batches,
            new_tokens=self.new, reused_tokens=self.reused, replay_tokens=self.replay,
            elapsed_ns=now - self.started, ttft_ns=0 if self.first is None else self.first - self.started,
            prefill_ns=self.prefill_ns, decode_ns=0 if self.first is None else now - self.first)
        self.metrics.event("request_complete" if outcome in {"completed", "paused"} else "request_failed",
            tokens=self.tokens, elapsed_ns=now - self.started)
