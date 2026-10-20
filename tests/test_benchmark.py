from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import pytest

from deepseekrpc.admin_client import AdminClient
from deepseekrpc.benchmark import measure_stream


@pytest.mark.parametrize("frozen_clock", [False, True])
def test_benchmark_client_uses_real_http_sse_usage_and_omits_content(monkeypatch, frozen_clock):
    # Client-protocol fixture only. This server never claims to load/run a model.
    if frozen_clock:
        monkeypatch.setattr("deepseekrpc.benchmark.time", SimpleNamespace(perf_counter_ns=lambda: 1000))
    observed = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            observed.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for value in [{"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": "synthetic client fixture"}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "length"}]},
                    {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}}]:
                self.wfile.write(b"data: " + json.dumps(value).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = measure_stream(AdminClient(f"http://127.0.0.1:{server.server_port}/v1"),
            {"model": "client-fixture", "seed": 41017, "messages": [{"role": "user", "content": "private"}]})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
    assert report["usage"]["total_tokens"] == 24
    if frozen_clock:
        assert report["wall_seconds"] == report["headers_seconds"] == report["first_visible_delta_seconds"] == 0
        assert report["end_to_end_output_tokens_per_second"] is None
    else:
        assert 0 < report["headers_seconds"] <= report["first_visible_delta_seconds"] <= report["wall_seconds"]
        assert report["end_to_end_output_tokens_per_second"] > 0
    assert observed[0]["stream_options"] == {"include_usage": True}
    assert "synthetic" not in json.dumps(report) and "private" not in json.dumps(report)


def test_matched_benchmark_refuses_implicit_random_seed():
    with pytest.raises(ValueError, match="seed"):
        measure_stream(None, {"model": "fixture"})
