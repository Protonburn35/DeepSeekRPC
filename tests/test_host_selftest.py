import os

from deepseekrpc.host_selftest import run_host_checks


def test_shipped_synthetic_host_diagnostics_use_actual_files_and_native_memory():
    report = run_host_checks()
    assert report["checks"][0]["status"] == "passed"
    assert report["checks"][1]["storage_row_reads"] == 2
    assert report["checks"][1]["cache_hits"] == 1
    assert report["checks"][-1]["status"] == ("passed" if os.name == "nt" else "not run")
    if os.name == "nt":
        assert report["locked_memory"]["locked_bytes"] == 65536
