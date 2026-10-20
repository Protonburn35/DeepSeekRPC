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

The user added official deepseek-ai/DeepInfer as an eighth source-only reference.
It is pinned at 8b16149741a4935e1dec8cea27853f273b374c5b. Read
docs/DEEPINFER_REVIEW.md: preserve full target replay, exact rejection correction
and confidence-independent fixed proposal verification. Do not run upstream
examples/tests that acquire model payloads. Independent tiny joint-law fixtures
pass; no upstream model/GPU execution has occurred.

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

## Latest 2026-10-20 Windows/source checkpoint

The eighth DeepInfer review and exact conditional sampling fixtures are complete
source work. Preserve BF16 draft-context/proposal keys too: DeepInfer explicitly
quantizes those to FP8, which does not supersede the user's requirement. Its
metadata confirms 128 prior target keys plus five bidirectional draft keys.
No DeepInfer import/build/example/test or model download has run here.

Windows CPU 39411095812 passed 27e7de91c857e482d53b79bc1b21f06ff717475f:
374 passed/one intentional non-Windows refusal skip. Actual private standalone
Python/TinyCC and offline Triton SM89/SM120 compilation passed 39411095629.
Blackwell ELF ABI 8's shifted SM field is fixed using actual captured bytes and
a pinned auxiliary LLVM header. A neutral annotated remote rollback tag is
checkpoint-27e7de9-windows-cpu-compiler. It is not GPU/model/review qualified.
Native ABI 2 run 39410211028 at 6f0f086 passed compile/link/import/SM inspection
and the actual Windows Torch 2.11+cu130 CPU suite (366 passes/one skip).

Long full vLLM run 39408672207 at 441b344 remains in progress with the unchanged
five-patch profile. The earlier 39400420994 was accidentally cancelled by the
DeepInfer source-lock trigger at 133/189, without a compiler error. Native vLLM
concurrency now preserves running builds; duplicate pending run 39410149104 was
cancelled. EXLlama duplicate 39408672326 was cancelled; use successful complete
39345608258. GitHub API's temporary HTTP 401 recovered without user action.
Do not ask the user to reconnect or provide a token.

The complete runtime workflow's YAML error is fixed and all workflows are
parsed in CPU CI. It now automatically follows successful native vLLM builds,
validates exact pins/profile/native source, installs the five-wheel private
runtime and performs actual dependency/native/compiler/API/parser qualification.
It can also be manually dispatched with those three successful run IDs. Fix any
actual failure, then preserve/download the successful artifact. The source
packet builder requires matching clean-commit Windows CPU and sealed complete
runtime evidence; it does not label pending work or hardware as passed.

New docs/REVIEW_GUIDE.md navigates all integrated features and fidelity hazards.
Do not ask for an intermediate Astra review. Next work is complete runtime
installation qualification, whole-candidate self-review and the exact-source
review packet. No optional grammar expansion or platform/model substitution is
required. All actual CUDA/model/hardware tests remain unrun, speculation remains
gated off until actual cluster qualification, and DSR-AUDIT-011 remains open.

Windows CPU 39415738104 exposed FileStore's non-ASCII Windows filename failure
(379 tests passed, one failed, one intentional skip). DSR-AUDIT-020 replaces
the single local rank's rendezvous with actual Torch HashStore/Gloo, shared by
startup and runtime qualification. No filesystem rendezvous or TCP rendezvous
listener is needed. Local full suite: 378 passed/four actual-Windows-only skips.
The next Windows CPU run must confirm this fix; preserve long vLLM 39408672207.

The HashStore follow-up failed Windows run 39416751099: that native binding is
intentionally omitted on Windows. Current source uses LocalRendezvousStore,
a real bounded Python subclass of Torch Store through its supported C++
trampoline (auxiliary Torch 2.11 source contract is pinned separately). It
retains the Python owner until actual vLLM model/world teardown; otherwise a
real later subgroup lost its Python callback methods. Qualification now forces
GC, creates a real additional Gloo group, performs its collective and repeats
cleanup. Check the subsequent Windows run, not either preceding failed attempt.
