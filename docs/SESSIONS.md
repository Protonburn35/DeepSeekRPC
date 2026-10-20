# Prefixes, pause and manual snapshots

This describes implemented source behavior. GPU prefix reuse, complete
three-stage snapshots and same-seed continuation have not executed on hardware.

Every ordinary chat request supplies its complete messages, tool results and
original images. A reusable encoder checkpoint binds exact tokens, image bytes,
positions, model/runtime identity and precision. All stages must offer the same
checkpoint. A hit retains its compressed source rows and restores the small
SWA/FP32 compressor checkpoint. It recomputes the old prompt's last partial
chunk, then runs the complete target replay for the new global encoder extent.
A changed prefix or missing exact vision features causes full encoder replay.
The coordinator reports reused/new encoder tokens and target replay tokens
separately; it never claims to skip the required target pass.

Ctrl+C, client disconnect and ordinary Stop cancel the current request while
healthy weights remain loaded. Fresh authenticated connections and a coherent
new epoch recover interrupted state. The explicit administrative pause finishes
the current committed output batch and leaves a pending input plus committed
RNG state. A pause during prefill waits for that prefill to finish.

Use the installed CLI on A, choosing 8082 explicitly for coexistence:

```powershell
deepseekrpc session pause --base-url http://127.0.0.1:8082/v1
deepseekrpc session status --base-url http://127.0.0.1:8082/v1
deepseekrpc snapshots save --base-url http://127.0.0.1:8082/v1
deepseekrpc snapshots list --base-url http://127.0.0.1:8082/v1
deepseekrpc snapshots restore --id SNAPSHOT_ID --base-url http://127.0.0.1:8082/v1
deepseekrpc session resume --id RESTORED_SESSION_ID --base-url http://127.0.0.1:8082/v1
```

Replace the displayed opaque IDs with the command results. Append
`--api-key-file .\credentials\api.key` when configured. The client reads the
token from that file and keeps it out of command-line arguments. It connects
directly to an explicit private endpoint and refuses redirects.

The pause response requests a pause. Wait for the active generation stream to
finish before saving. A paused OpenAI completion ends with `length` and includes
`deepseekrpc_session`. Administrative continuation returns NDJSON token batches
and raw text, including original reasoning/tool markers. It continues the same
pending input, sampling settings, penalties, stop/UTF-8 suffix and three RNG
domains without reseeding. EOS/stop/output-ended requests remain ended.
An ordinary subsequent OpenCode chat request still supplies its complete client
history and uses exact prefix negotiation.

Snapshots contain inference state. Retain the client's conversation, tool
results and original images independently. All source/shadow/index rows are
unquantized BF16, open compressor pairs retain FP32, and draft context remains
target-derived. Sparse selections and proposals are window-local and are
regenerated for the next admitted window. Weights and expert caches are excluded.

Only idle committed state may be saved. B/C stream tiles of at most 1 MiB to A;
they do not write snapshot files. A saves its state and original/feature assets
in its dedicated runtime snapshot directory. The exact byte total and 2 GiB
free-space reserve are checked before writing. Near 1M, source/index/shadow
payload alone is 5 GiB for the default split; SWA, metadata and images add bytes.
The default quota is 16 snapshots. Remove one explicitly when needed:

```powershell
deepseekrpc snapshots delete --id SNAPSHOT_ID --base-url http://127.0.0.1:8082/v1
```

A snapshot appears only after all three streams, checksums and acknowledgments
are complete and its manifest is published atomically. Restore holds Windows
file handles that deny write/delete sharing, verifies every complete file, then
validates every stage stream before applying GPU state. It uses another bounded
streaming pass for application, independent local slot maps and a fresh epoch.
B's locked weight pages are checked again before readiness. No second complete
multi-GiB cache copy is allocated. This validation policy adds disk/network time.

Restores require the exact application/runtime, model, stage plan, cache/wire
precision and configured run identity. An interrupted apply invalidates the
affected session on all ranks while retaining healthy weights for reset/replay.
An incomplete `.pending-...` directory is never listed as restorable. A device
failure requires explicit process/device recovery.
