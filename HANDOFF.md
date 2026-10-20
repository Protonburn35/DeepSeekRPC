# DeepSeekRPC implementation checkpoint

The goal is the complete native Windows three-PC October 1017 EXL3 2.9
application described by docs/IMPLEMENTATION_SPEC.md, including CUDA. This is
an incomplete engineering checkpoint, not an Astra review candidate or a
hardware-test release. Continue implementing; the user wants the first whole
Astra handoff after all mandatory application features are present.

All three deployment PCs run Windows 11: A RTX 4090, B RTX 4060 laptop, C RTX
5070 Ti + Ryzen 7 9800X3D. Current user instructions require unquantized BF16
KV/index storage and FP32 stage activation/pre_mix wire. FP16 inside existing
EXL3 weight kernels is separate. Optional FP16 cache mode is excluded.

Reuse Protonburn35/DeepSeekRPC on main. Pushes, Windows-only CI, direct commits
and merges are authorized. Use the neutral automation author and keep personal
identity out of source, artifacts and logs. Do not modify working StrataRPC,
create another project repository, download full model/Engram payloads into
cloud, rent GPUs or rely on access to the user's personal cloud VM.

## Source and environment

Checkout: /workspace/DeepSeekRPC. Read-only upstream references, bounded model
metadata/headers and source archives: /workspace/deepseekrpc-references.
Writable build copies: /workspace/deepseekrpc-build. Python tools:
/workspace/deepseekrpc-tools/.venv. Logs/artifacts:
/workspace/deepseekrpc-evidence. All admitted revisions/hashes are under
provenance/. Source precedence and later user overrides are in docs/DECISIONS.md.

The complete original Astra prompt is retained unchanged in
IMPLEMENTATION_SPEC.md; the older handoff is background research. User
instructions supersede both, including current push/CI authorization, all-Windows
hardware, BF16-only caches, FP32 wire and complete implementation before review.

## Implemented components

- Complete physical pack inventory and selective publisher-verified local reads;
  B boot-only VirtualLocked canonical weights with all-page residency checks,
  permanently closed file source and one shared MiB pinned transfer tile.
- Actual EXL3 plugin linears, all owned non-routed bindings, bounded expert arenas
  and leases, all routed choices, official pre-w2 weighting and shared experts.
- Native CUDA BF16 sparse attention/index/scatter and original Engram decode;
  eager independent source/shadow/SWA pools, undo, exact visibility, FP32 pair
  compression, actual learned Engram operations and delayed mHC backend.
- Actual owned vLLM decoder/vision/draft construction and complete native loading;
  C input embedding replica; original target feature/Markov/confidence heads.
- Native Windows bootstrap and whole-stage budget admission. LOADED is distinct
  from distributed READY. Peer identities bind model, tokenizer, precision,
  installed source/components and native binaries.
- Exact sampled rejection correction, independent transactional RNG domains,
  bounded Stage C Markov sampling and separate processed/pending token decisions.
- Original token/image history identities and generated-token counts; phase
  ledger and actual CUDA execution boundary with dependencies, full CED traversal,
  unanimous commit receipts, reset and faulted-session invalidation.
- Mutual TLS, bounded lossless fragments and CUDA transfer tiles; one-owner
  control socket pumps, correlated RPC, cancellation/heartbeat during queued work,
  deadline invalidation and certificate-derived mesh connection acceptance.
- Persistent bounded stage actors/services, locally derived dependency mesh,
  full-prompt data transfer, full two-pass CED and speculative coordinator source.
  Output publication waits for all actual GPU receipts, all-stage history commits
  and Stage C RNG finalization. The new GPU/service integration remains unrun.
- Unchanged original image processing, exact token-span expansion, bounded
  original/feature storage and A's actual tower/aligner window resolver source.
  Exact incremental ByteLevel/UTF-8 and transactional stops trim speculative
  accepted prefixes before GPU finish and retain partial stop text for streaming.

These components have not executed a complete model. OpenAI JSON/SSE request
ownership, the original streaming parser, persistent Windows startup and
authenticated interrupted-mesh replacement now have integrated source. Early
control/API diagnostics run before model loading; Ctrl+C cancels a request and
Ctrl+Break explicitly shuts down. Native worker/model execution is unrun.
Exact prefix checkpoints, bounded all-stage state save/restore, administrative
pause and exact pending-input/RNG continuation now have integrated source and
HTTP/CLI entry points. Their actual GPU/distributed execution remains unrun.
The complete Windows dependency/install bundle and broader GPU diagnostics are
being qualified. Optional grammar/response-format and named tool constraints
are explicitly rejected; the specification allows clear unsupported controls.
They are not required features being silently deferred.
Bounded numeric logs, machine-readable cache/I/O/transfer/request metrics, live
and offline diagnostic archives and matched real-API benchmark modes now have
integrated source. The separate OpenCode profile and its actual pinned SDK wire
adapter have CPU protocol evidence. Full Docker/GPU performance is unrun.

## Verification

Project native ABI 2 passed Windows compile/link/install/import, SM89/SM120
inspection and SDK-free import again in run 39340258236. Full pinned EXLlama
passed run 39336142655, including both architectures, SDK-free native import,
Python package import and dependency consistency. Full vLLM native run
39336142640 reached 107/203 compile steps, then failed in optional FlashKDA and
Qutlass extensions outside the selected EXL3 path. A fifth exact-source profile
patch omits those extensions and preserves core/MoE targets. The next native
build must qualify it. Build reports distinguish native-only qualification
from complete Python serving dependencies. No GPU kernel/model has executed.

Windows CPU run 39344050108 and project native run 39344050177 passed fdf1247. The expanded local suite
passed 265 tests with four actual-Win32-only skips after image and stop additions.
The preceding stop suite reproduced the open CPU issue; the real ATen observer
narrows it to sqrt (up to 3e-4), while matrix multiply/softplus stay near FP32 error.
The next passing suite does not resolve it. Focused tests include actual thread invalidation/queue bounds,
history/RNG publication and a real TLS million-token prompt transfer. The full follow-up
caught the intermittent CPU Torch router discrepancy again: the FP64 diagnostic
puts NumPy error below 7e-8 and Torch up to 4.6e-5 on the first four rows. Keep
tolerance unchanged; diagnose the real CPU operation path. A passing rerun is
not a repair. Tests use frozen source methods and independently pinned CPU math
packages; CPU/reference, native builds and hardware qualification remain distinct.

The application wheel builds and packaged deployment metadata/history/source
identity were exercised after installation outside the checkout. See
BUILD_REVIEW_LEDGER.md for precise per-checkpoint scope and docs/FINDINGS.md for
open source fidelity findings. Update evidence when builds finish.

Actual SM89/SM120 execution, full model/vision/1M context behavior, complete B
residency and performance remain unrun/unmeasured. Independent Astra review
remains pending and user-mediated. Do not ask for an intermediate review or an
early hardware trial.

## Development checks

Use the existing Python environment, PYTHONUTF8=1, OMP_NUM_THREADS=2 and
OPENBLAS_NUM_THREADS=2. Run appropriate focused tests and the complete suite
before a tested checkpoint. tools/ci_check.py builds/installs the application
wheel outside the checkout. tools/audit_installed_integration.py exercises the
real CPU-only empty vLLM package and plugin; it does not qualify CUDA constructors.
The reusable cloud configuration is saved as a draft; publishing is a separate
UI action and is not a reason to stop implementation.

The 2026-10-20 API/worker suite passed 304 tests with four Win32-only skips and
one failure in the still-open Linux CPU sqrt finding. Immediate captured sqrt
bytes reproduce the error, have no later mutation, and three repeated calls on
the identical captured input are accurate. This is evidence for a transient
real CPU library path, not a resolved defect. Keep all tolerances unchanged.

The later prefix/archive suite passed 323 tests/four skips; the integrated
snapshot/session suite passed 332/four skips. Neither pass repairs the prior
CPU failure. The latest private API/archive/inventory/prefix focused checks
passed 36 tests. Windows CPU run 39350431569 and project native run 39350431617
passed checkpoint 8100979. Full vLLM/EXLlama runs 39345608248/39345608258 remain
in progress. See docs/SESSIONS.md for implemented source semantics and limits.
