from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.metrics import COUNTERS, TIMINGS, Metrics


def test_content_free_counters_are_threadsafe_have_explicit_denominators_and_atomic_files(tmp_path):
    metrics = Metrics(root=tmp_path / "diagnostics", stage="B")
    metrics.set_phase("inference")
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda _: metrics.add("gpu_bundle_requests"), range(1000)))
    metrics.add("gpu_bundle_hits", 250)
    metrics.observe("rpc_send", 1_000_000)
    metrics.observe("rpc_send", 1_000_001)
    metrics.flush()
    report = json.loads((tmp_path / "diagnostics/metrics.json").read_bytes())
    assert report["counters"]["inference.gpu_bundle_requests"] == 1000
    assert report["fractions"]["inference"]["gpu_bundle_hit_fraction"] == .25
    assert report["fractions"]["inference"]["ram_bundle_hit_fraction_on_gpu_miss"] is None
    assert report["timings"]["inference.rpc_send"]["buckets"][3:5] == [1, 1]
    assert report["not_measured"]["physical_ssd_bytes"] is None
    assert not (tmp_path / "diagnostics/metrics.json.partial").exists()


def test_trace_logs_rotate_with_fixed_bounds_and_refuse_content(tmp_path):
    metrics = Metrics(root=tmp_path / "diagnostics", stage="A", traces=["io"], log_bytes=1024)
    for _ in range(100):
        metrics.event("file_read", bytes=1024)
    logs = list((tmp_path / "diagnostics").glob("*.jsonl"))
    assert 1 <= len(logs) <= 3 and all(path.stat().st_size <= 1024 for path in logs)
    assert sum(len(path.read_bytes().splitlines()) for path in logs) < 100
    with pytest.raises(ValidationError):
        metrics.event("file_read", prompt="private text")
    with pytest.raises(ValidationError):
        metrics.latest_request(tokens="private text")
    with pytest.raises(ValidationError):
        metrics.add("a username is not a counter")
    metrics.flush()
    assert b"private" not in (tmp_path / "diagnostics/metrics.json").read_bytes()


def test_complete_compact_inventory_leaves_room_for_real_status_in_control_frame():
    metrics = Metrics()
    for name in COUNTERS:
        metrics.add(name, (1 << 63) - 1)
    for name in TIMINGS:
        metrics.observe(name, (1 << 63) - 1)
    from deepseekrpc.jsonutil import canonical
    assert len(canonical(metrics.compact())) < 8 * 1024
