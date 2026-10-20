"""Matched real private-API measurements; generated content is never retained."""
import time

from .jsonutil import ValidationError, integer, loads


def _visible(delta):
    if not isinstance(delta, dict) or not isinstance(delta.get("tool_calls", []), list):
        raise ValidationError("benchmark stream contains an invalid delta")
    if any(isinstance(delta.get(key), str) and delta[key] for key in ("content", "reasoning_content")):
        return True
    return any(any(isinstance(call.get("function", {}).get(key), str) and call["function"][key] for key in ("name", "arguments"))
               for call in delta.get("tool_calls", []))


def measure_stream(client, body):
    if not isinstance(body, dict) or type(body.get("seed")) is not int:
        raise ValidationError("matched benchmarks need an explicit request JSON object with a fixed integer seed")
    request = {**body, "stream": True, "stream_options": {"include_usage": True}}
    # On CPython 3.12 Windows, monotonic_ns uses GetTickCount64 (about 15 ms
    # resolution). QueryPerformanceCounter also remains monotonic and resolves
    # local requests below a tick. Do not invent throughput if no time elapsed.
    started, first, usage, reason, done = time.perf_counter_ns(), None, None, None, False
    events = total_bytes = 0
    with client.open("POST", "/chat/completions", body=request) as response:
        headers_ns = time.perf_counter_ns() - started
        while line := response.readline(1024 * 1024 + 1):
            total_bytes += len(line)
            if len(line) > 1024 * 1024 or total_bytes > 1024 ** 3 or events > 2 * 1048576:
                raise ValidationError("benchmark stream exceeds the bounded protocol inventory")
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            value = loads(data, limit=1024 * 1024)
            if not isinstance(value, dict) or "error" in value:
                raise RuntimeError("actual benchmark generation returned a streaming error")
            events += 1
            if value.get("usage") is not None:
                usage = value["usage"]
            for choice in value.get("choices", []):
                if first is None and _visible(choice.get("delta", {})):
                    first = time.perf_counter_ns() - started
                if choice.get("finish_reason") is not None:
                    reason = choice["finish_reason"]
    elapsed = time.perf_counter_ns() - started
    if not done or usage is None or reason not in {"stop", "length", "tool_calls"}:
        raise RuntimeError("benchmark stream did not complete with actual usage and a successful finish reason")
    if not isinstance(usage, dict) or not {"prompt_tokens", "completion_tokens", "total_tokens"} <= set(usage):
        raise ValidationError("benchmark response omitted authoritative context/output usage")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        integer(usage[key], "benchmark actual token count", 0, 1048576)
    if usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]:
        raise ValidationError("actual benchmark usage exceeds or disagrees with true context accounting")
    usage = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return {"wall_seconds": elapsed / 1e9, "headers_seconds": headers_ns / 1e9,
        "first_visible_delta_seconds": None if first is None else first / 1e9,
        "usage": usage, "finish_reason": reason, "stream_events": events, "wire_bytes": total_bytes,
        "end_to_end_output_tokens_per_second": usage["completion_tokens"] / (elapsed / 1e9) if elapsed > 0 else None}


def _counter(snapshot, suffix):
    return sum(value for key, value in snapshot["metrics"]["counters"].items() if key.endswith("." + suffix))


def run_benchmark(client, body, *, repeat=3):
    integer(repeat, "benchmark repetitions", 1, 20)
    report = {"format": 1, "scope": "actual already-loaded private HTTP application; no GPU/model emulation",
        "workload": "same caller-supplied request and seed; streaming/usage enabled; content/paths are omitted",
        "runs": [], "throughput_targets": {"new_prefill_tokens_per_second": 500, "generation_tokens_per_second": 40},
        "targets_status": "stretch goals; no fidelity waiver or predicted achievement"}
    for _ in range(repeat):
        before = client.request("GET", "/diagnostics")
        if not before["runtime"]["ready"] or before["runtime"]["active_request"]:
            raise ValidationError("benchmark requires the already-loaded ready cluster without another active request")
        measured = measure_stream(client, body)
        deadline = time.monotonic() + 15
        while True:
            after = client.request("GET", "/diagnostics")
            if not after["runtime"]["active_request"]:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("benchmark request has not released its actual model reservation")
            time.sleep(.05)
        valid = (_counter(after, "completed_requests") - _counter(before, "completed_requests") == 1 and
            all(_counter(after, key) == _counter(before, key) for key in ("paused_requests", "failed_requests", "cancelled_requests")) and
            _counter(after, "published_tokens") - _counter(before, "published_tokens") == measured["usage"]["completion_tokens"])
        measured["server_counter_attribution"] = "single completed request" if valid else "unavailable; another operation or mismatched counters were observed"
        if valid:
            latest = after["metrics"]["latest_request"]
            measured["scheduler"] = latest
            measured["new_encoder_tokens_per_second"] = latest["new_tokens"] / (latest["prefill_ns"] / 1e9) if latest["prefill_ns"] else None
            # Prefill includes both encoder work and required target replay.
            # Reused encoder tokens never inflate the numerator.
            measured["steady_published_tokens_per_second"] = max(0, latest["tokens"] - 1) / (latest["decode_ns"] / 1e9) if latest["decode_ns"] else None
            measured["host_application_file_read_bytes"] = sum(_counter(after, key) - _counter(before, key)
                for key in ("model_file_read_bytes", "engram_file_read_bytes"))
        measured["peer_metrics"] = (after.get("worker") or {}).get("peer_diagnostics")
        measured["peer_sample_age_ns"] = (after.get("worker") or {}).get("peer_heartbeat_sample_age_ns")
        report["runs"].append(measured)
    return report
