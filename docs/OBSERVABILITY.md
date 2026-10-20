# Logs, diagnostics and matched benchmarks

Each persistent worker writes `state_root/diagnostics/metrics.json` atomically
every five seconds, including while weights load. Default event logs contain
fixed event names and numeric fields. Three rotated files of at most 4 MiB each
bound retained log volume. Prompts, generated text, tool arguments, image bytes,
paths, environment values and credentials are absent from those logs.

Use `serve --log-level error|warn|info|debug|trace` and repeated component flags,
for example `--trace rpc --trace io --trace kv`. Allowed components are rpc,
experts, cache, io, kv, cuda, vision and spec. Tracing is opt-in and can affect
performance. Fixed event/field inventories refuse arbitrary string payloads.
An inability to write logs disables that writer and sets `log_write_failed`;
it does not authorize a different model or storage fallback.

Counters record completed GPU bundle hits/promotions/evictions, the RAM tier on
GPU misses, routed token assignments and unique per-window bundles, model and
Engram file-read requests, Engram row hits, blocking PCIe upload bytes, plaintext
framed RPC bytes, new/reused encoder work, required target replay and committed
output/draft/rollback work. GPU hit fraction uses all valid GPU bundle requests;
RAM fraction uses RAM bundle requests made on GPU misses; Engram fraction uses
row requests including duplicates. Failed requests can leave requests greater
than completed hits plus loads. A bundle request is not a token/expert route.

File-read bytes are application requests. The synchronous unbuffered Python
reader does not bypass the Windows system file cache. Its file miss count is
not evidence of a physical SSD miss. Physical SSD bytes, hard page faults,
pagefile I/O and thermal/power data remain explicitly unavailable. No
speculative read-ahead is issued, so the implementation does not manufacture
prefetch/wasted-read numbers. Boot integrity reads have separate counters.
Lifetime phases are local; B/C inference counters also include installed-model
diagnostic probes until another local lifecycle phase is selected.

Nanosecond timing histograms have fixed, noncumulative buckets plus count,
sum and maximum. File/transfer/RPC timings are inclusive host wall times.
`stage_cuda_forward` is actual start/end CUDA-event elapsed time on the owned
stream, including joins and host submission gaps; it is not active-kernel-only
utilization. The host-forward timer also includes admission and synchronization.
Current compact peer summaries arrive through existing heartbeats. Diagnostics
include their sample age, not a claim that cached peer counters are live.

The scheduler measures first committed output, prefill, batch intervals and
request wall time including output backpressure. The first-token timer begins
at coordinator admission, after HTTP prompt preparation. Fresh/reused encoder
tokens are separate from full target replay. Rejected proposals do not count as
published output. Live `/v1/diagnostics` exposes these public counters, component
identities, allocation ledgers, last budget checks and qualification status;
normal chat responses do not carry tracing internals.

Create a bounded live host archive without loading another model:

```powershell
.\runtime\Scripts\python.exe -I -m deepseekrpc.cli diagnostics --output .\host-diagnostics.zip
```

For saved local A/B/C metrics and the three known numeric log files, add
`--config .\config\B.json` on the corresponding PC instead of a live API URL.
The collector reads only explicit files from an owned runtime directory; it
does not walk credentials, home, model, image or snapshot directories. Offline
results show sample age and do not assert that the process is currently ready.
Archives and benchmark outputs must be new files. API keys, when needed, are
read with `--api-key-file`; they are not written to reports.

After reviews and local/cluster qualification, measure the actual already-loaded
service using a request file with an explicit seed:

```powershell
.\runtime\Scripts\python.exe -I -m deepseekrpc.cli benchmark --request-file .\config\benchmark.example.json --repeat 3 --output .\benchmark.json
```

Run matched benchmarks without other clients. The same request and seed are
used each time; streaming and authoritative usage reporting are enabled.
Reports retain counts and timings, not request paths or generated content.
They distinguish response headers, first visible content/reasoning/tool delta,
end-to-end output rate and the scheduler's committed-token timers. A role-only
initial SSE event is not counted as first visible output. Server counter
attribution requires exactly one completed request, matching published usage
and no intervening pause/cancel/failure. Otherwise those derived figures are
unavailable. Peer summaries retain their independent heartbeat sample age.

New-encoder rate uses only newly processed encoder tokens in its numerator and
includes the required target replay in measured prefill time. Warm prefix hits
do not inflate that numerator. Compare fresh and warm cases separately; no
unmeasured throughput is populated. The 500 prompt-token/s and 40 generated-
token/s figures remain stretch goals. Installed-model measurements have not
run in the CPU-only development environment.
