# Append-only build/review ledger

## 2026-10-19 — initial M0/M1 engineering checkpoint

- User authorized implementing Astra's prompt in the existing DeepSeekRPC repo;
  older personal Google Cloud VM and draft-hold instructions are superseded.
- All seven reference commits resolved; source archives acquired and hashed.
  Official and EXL3 model revisions pinned; only metadata and bounded headers
  were acquired, never full weights/Engram.
- Source compatibility checker exit 0; preserved NOTICE and documentation BAD
  lines. Complete header-derived inventory reconciles 192495 pack tensors plus
  four external Engram table/scale tensors and actual physical K variants.
- Initial local source checks: 129 passed, one native-Windows-only skip. Linux
  wheel build/install and installed CLI outside checkout passed. This first run
  explicitly recorded a dirty working tree; exact-commit evidence follows in
  the external evidence directory after the source checkpoint is committed.
- True three-process TLS/TCP test uses affine fixture math, persistent processes,
  fragments, commits and cancel/resume. It does not execute DeepSeek or CUDA.
- Two missing upstream encoding example fixtures caused initial test failures;
  fetched original examples from the same pinned revision. Final encoding suite
  passes without replacing goldens or disabling assertions.
- CPU Torch initial uv extraction failed; pip installed the same hash-required
  CPU wheel successfully with TLS verification. No affected unchecked artifact
  was used, and no expected checksum changed.
- Native Windows CI workflow prepared with pinned action commits. No remote
  publication or CI execution yet. Independent Astra review pending.
- Source implementation remains incomplete; Windows/CUDA, real GPU/model/vision/
  sessions/long context are unrun; speed is unmeasured. See requirements table.

## 2026-10-19 — native Windows evidence and partial M2 follow-up

- User authorized the initial push/draft PR/CI and subsequently granted ongoing
  repository authorization. PR #1 is a draft. Future CI builds are Windows-only.
- The first Windows 2025 run (39309316416, job 118254514086) passed 129 tests,
  one non-Windows-only skip, wheel build and install. The final CLI check failed
  before launch: base Python puts console scripts in Scripts/, not alongside
  python.exe. Corrected discovery to sysconfig.get_path("scripts"); repeat pending.
- The first checkpoint inherited a personal local Git author setting. Replaced
  its author/committer metadata and tag with the neutral project identity, and
  checked tracked files for the inherited name/email (zero matches). Repository
  Git configuration now uses that neutral identity for future commits.
- Built and installed the pinned upstream empty CPU vLLM and no-CUDA plugin
  wheels; real import, plugin discovery and entry-point registration passed.
  This is not native extension or CUDA qualification. Resolved the actual common
  dependency stack, corrected NumPy to 2.3.5 and setuptools to 80.10.2; pip check
  passes. The cloud-only audit lock is under tools/cloud/.
- Added the unchanged official Engram hash source, metadata-only tokenizer
  fixture, canonical compressed map SHA and independently generated upstream
  hash fixture. Focused hash/storage suite: 11 passed. Hash history allocates
  its full 1M metadata eagerly and supports partial acceptance/cancellation;
  this is not a full-model 1M allocation or learned Engram execution claim.
- DSR-AUDIT-001 records a scale-byte-zero discrepancy between official Torch
  conversion and a vLLM kernel shortcut. The decoder retains official semantics;
  named CPU regression passes. Actual GPU adapter and Astra review remain open.
- Expanded local source suite: 135 passed, one Win32-only skip; wheel build,
  install and CLI outside the checkout passed. Exact committed-source evidence
  follows the checkpoint. Build tools are now included in the hash-pinned dev
  lock and wheel builds use those installed tools without build isolation.

## 2026-10-19 — owned loading and resident boot admission

- SelectiveWeights validates the pinned index, header and full publisher shard
  SHA-256 with bounded local reads, then exposes owned tensor ranges only.
  Corrupt payload, mutation, unsafe shard names, unowned access and absent
  non-owned files have focused regression coverage. No weights were downloaded.
- LockedOwnedWeights implements B's boot-only locked host allocation and closes
  every model file handle permanently. Its real Win32 fixture deletes the local
  synthetic model files after preload; native CI execution pending. End-to-end
  laptop/device/staging residency remains unqualified.
- DSR-AUDIT-002 documents conflicting cached Python/CUDA INT8 defaults and the
  explicit pre-import policy. It is not yet wired into a full model executor.
- Windows run 39310450571: 134 passed, one failed, one skipped. The untrusted-CA
  negative case was rejected, but Windows exposed the peer TLS close as an
  expected connection reset. Narrowed the assertion to accept SSLError or that
  reset on the server; it still requires handshake failure.
- Reusable cloud installer validation uncovered Hugging Face Hub 2.x selected by
  the CPU tools, conflicting with the installed source-audit Transformers <2
  constraint. Pinned Hub 1.33.0 in all tool/application locks; pip check passes.
  Refreshed installer passed 141 tests, two Win32-only skips and native C++ check.
- User authorized merging and direct main commits after checks, and explicitly
  deferred Astra handoff until the complete CUDA/application implementation.

## 2026-10-19 — first fully passing native Windows foundation checkpoint

- Windows run 39311940687, job 118263493278, passed 142 tests and one explicitly
  non-Windows-only skip, wheel build/install, installed CLI outside the checkout.
  This includes real Win32 locked preload after closing/deleting the synthetic
  model files. Source tested by GitHub was merge tree 745f8d24286aa8de273762a1f94159f87e4a51fd;
  its implementation head is d2be9102b206a0ff6dcd6c32bc181d7a2541d341.
- Integrated the tested head into main by fast-forward, preserving neutral commit
  metadata. Closed draft PR #1 because its source is now completely included.
  Tagged checkpoint/m2-d2be910 and retained a source archive plus SHA256SUMS.
- Added actual CUDA source for FP16 sparse attention/index scores, original
  Engram decode and generation-checked scatter, with an explicit extension ABI
  and no CPU inference fallback. Three new independent CPU equation fixtures
  pass. CUDA build qualification is pending the newly prepared Windows workflow;
  no GPU execution or full distributed graph completion is inferred.
- Windows CUDA build dependencies and compiler SDK packages are hash-pinned.
  The Windows SDK helper refuses installation on the Linux cloud host. Future
  native evidence verifies both sm_89 and sm_120 cubins plus no-GPU import.

## 2026-10-19 — BF16 cache requirement and first CUDA compiler repair

- User clarified the actual requirement: unquantized KV storage. BF16-only
  KV/index caches replace the original FP16 default and its optional precision
  modes. No FP8/FP4/INT8 cache fallback. Planner, typed wire and native operators
  enforce this. The 5 GiB source/shadow arithmetic is unchanged; other state and
  working buffers remain additional. Original FP16 EXL3 kernels are separate.
- Added exact selected mul1 layout validation for physical K2/3/4/5/6/8,
  rotations, shapes and signed multiplier bytes. DevicePackedLinear binds the
  actual plugin/ExLlama class and forces packed execution without reconstruction;
  CUDA execution remains unrun. Focused loader/layout/Win32 CPU checks passed
  15 tests, two native Win32-only skips. CUDA host registration/DMA on B has an
  explicit bounded direct-copy path without another pageable host mirror.
- Native Windows CUDA run 39313690245 verified SDK downloads and actual Torch
  2.11.0+cu130, compiled the C++ bindings, then hit a Torch Python-header/NVCC
  generated-code ambiguity. Replaced torch/extension.h in the CUDA translation
  unit with ATen/ATen.h; Python bindings remain in bindings.cpp. No math assertion
  was disabled. Also replaced the Windows infinity macro with CUDA's constant.
- Planner/protocol/layout focused suite after BF16 policy: 55 passed locally.
  Inventory regenerated with BF16 plan metadata; tensor bytes and ownership are
  unchanged. New native build remains pending; actual GPU execution is unrun.

## 2026-10-19 — native CUDA compilation passes

- Run 39314752138 passed on native Windows for clean source 21fe6b7. Actual
  CUDA 13.2 compilation, link, install, no-GPU extension ABI import and cuobjdump
  verification passed; the module contains both sm_89 and sm_120 cubins. Wheel
  hash and exact checks are retained in evidence/native-first-build.json.
- The same run passed 160 CPU tests with actual Windows Torch 2.11.0+cu130,
  one explicitly non-Windows-only skip. GPU kernels were not executed.
- Added the fixed-allocation CUDA expert tier over bounded desktop RAM or B's
  locked host store, stream/event protected slot leases and stale-handle refusal.
  Its actual GPU operation and WDDM residency remain unrun. Original ExLlama
  native extension and full vLLM graph/build integration are still required.
- User reaffirmed FP32 stage activation/pre_mix transfers. Planner/inspection
  CLI now expose the enforced --layer-stage-wire fp32 setting explicitly.
- Native Windows host-lock admission now reserves process working-set quota
  before VirtualLock, preserving the current resident runtime. Added a real
  64 MiB lock/nested allocation/quota release regression; Windows run pending.

## 2026-10-19 — full pinned EXLlama Windows build qualification started

Added a separate source-verified, hash-pinned native Windows EXLlama build job.
It requires precompiled SM89/SM120 cubins and imports the installed package;
there is no model download or GPU execution. The previous project CUDA run
39316435141 passed compile/link/import/cubin checks and CPU stack tests. The
new full EXLlama build is pending and does not complete native vLLM or the graph.

## 2026-10-19 — expand the verified SDK for the upstream runtime

Full EXLlama run 39317323604 verified source/dependencies and reached native
compilation (206 translation units), then stopped at missing cusparse.h in
Torch's CUDAContextLight.h. The earlier minimal SDK sufficed for this project's
four kernels but not upstream Torch CUDA headers. Added NVIDIA-published hashes
and byte bounds for cuSPARSE, cuSOLVER, cuRAND, cuFFT, NVRTC and NVTX components;
source is unchanged. Also constrained llguidance below 1.8 to match vLLM common
requirements. Repeat full upstream Windows build pending.

## 2026-10-19 — bounded routed executor and native vLLM build source

Implemented the owned expert catalogue, all-route partitioning, bounded CUDA
expert execution and a real FusedMoEMethodBase adapter. The adapter prevents
per-layer/all-expert CUDA Parameter allocation; the complete graph/owned loader
integration is still pending. Preserved official routing-weight placement before
the down projection's FP16 conversion (DSR-AUDIT-003). Added a real-GPU operation
self-test command, not executed on this CPU host.

Local expanded suite: 166 passed, three actual-Win32-only skips. An earlier
full run showed a router CPU-reference mismatch (24/42 weights, maximum error
4.6253204e-05, exact selected IDs unchanged). The isolated test, preceding math
suite and subsequent complete run passed at the original tolerance. No tolerance
was widened; retain this intermittent observation for further diagnosis.

Added source/dep-pinned native vLLM wheel build qualification on Windows. Full
serving dependency consistency, GPU execution and full model remain unqualified.

## 2026-10-19 — eager BF16 cache pools and bounded CUDA undo

Implemented actual eager BF16 device row pools, independently mapped source /
shadow slots, generation-checked scatter, metadata reservations and bounded
GPU undo rows. Backups finish before the mutation becomes abortable. Rejected
suffix writes restore payload/generations in reverse token order, including
multiple overwrites of a small ring; failed tentative writes block consumption.

New slot-map tests plus existing state/protocol checks: 26 passed locally.
Actual GPU rollback regression is in selftest-cuda and remains unrun. Full-model
pool registration, global transactions and whole-process/WDDM admission remain
pending; this does not establish 1M model execution or physical residency.

## 2026-10-19 — bounded BF16 index selection and vLLM long-path repair

Added tiled actual-CUDA index scoring over independently mapped BF16 key pools,
streaming top-512, source-20 candidate blocks and candidate-only downstream
scoring. Ranking diagnostics pass across odd tile sizes and equal-score cases;
GPU index/rollback self-tests exist but remain unrun. Source projection/RoPE and
the complete graph/phase scheduler are still pending. Cache read leases retain
the window until GPU completion; cancellation checks occur between key tiles.

Native vLLM run 39318861172 completed SDK/Torch/CMake setup and fetched all ten
locked dependency commits. Configuration stopped when Git checked out very
long ROCm submodule paths inside the FlashAttention source. The Windows job now
sets core.longpaths for its own Git subprocesses without changing global Git
configuration. Full compilation/import remains pending.

## 2026-10-19 — real learned disk-Engram adapter source

Added a subclass of the actual vLLM Engram graph that preserves its learned
projection/gate and replaces whole-table allocation with bounded original row
reads, pinned upload and explicit BF16 decode. Production construction requires
stage A, TP=1 and verified publisher header/full-file SHA. Extended the installed
integration audit to check actual upstream adapter bases; no namespace stubs.
Engram storage/history checks: 12 passed locally. CUDA graph construction and
learned output have not run. The model-stage factory still needs to inject it.

Found DSR-AUDIT-004: selected CUDA mHC imports demand TileLang even though the
Windows requirement set excludes it for a TVM-FFI conflict. Native wheel builds
do not resolve this graph issue; an explicit compatible backend remains needed.

## 2026-10-19 — explicit Windows CUDA mHC backend source

Added bounded native-Torch CUDA mHC pre/post and shifted post/pre, preserving
the upstream delayed mix equations, BF16 internal streams and FP32 coefficients.
FP32 stage activation/pre_mix transport remains required. The exact-source patch
selects this backend and defers unused TileLang imports on Windows. Packaged the
upstream Apache-2.0 notice, file hash and adaptations. CUDA math policy disables
TF32/reduced-precision auxiliary reductions and detects later changes.

Coefficient diagnostics against independent FP64 equations, plus precision
policy mutation regressions: four passed on CPU. The source patch applied to a
verified writable copy and both patched Python modules compile. Updated actual
installed class/entrypoint audit passed. Actual patched Windows graph import,
CUDA differential execution and full stage integration remain pending.

## 2026-10-19 — full EXLlama Windows wheel qualification

Run 39318861176 at af15c3a passed the complete pinned EXLlama compile/link,
wheel install, native and Python imports, pip consistency and actual fatbin
inspection for SM89 and SM120. The report and listing are retained in
exllama-first-build.json and exllama-first-fatbin.log. The wheel SHA is
7ff7c8976e3553bcba34131408eccfd4408173ffcfd5f64508c8f13bf5f3e950.
No GPU operation executed. The first run used SDK PATH during import; a
separate no-SDK native import check is now required for subsequent builds.

## 2026-10-19 — concrete BF16 attention adapter and source patch repairs

Added a concrete subclass of the real vLLM attention, original fused Q/KV
norm and RoPE calls, explicit unquantized BF16 source/shadow pools, independent
packed output groups, FP32 pair state around the real compression kernel,
bounded native parameter bootstrap, and typed per-phase visibility/selection.
The exact decoder patch adds explicit attention/Engram factories; no namespace
replacement or CPU inference fallback is used. The real installed CPU audit
imports all three owned adapters. CUDA constructors, numerical execution,
complete stage bootstrap and distributed scheduling remain unrun/incomplete.

Native vLLM run 39320169086 reached compiler C2719 in unused SM12x FP8 GEMMs.
The explicit EXL3 profile omits unused SM12x FP8/NVFP4 matrix families and
reports capabilities by compiled dispatch, without changing CUDA descriptor
alignment. Runs 39326982042 and 39327599761 exposed Windows line conversion:
first the checked-out patch hash, then Git's patched output hash. Repository
attributes retain LF and each apply command explicitly disables autocrlf.
All three chained patch output hashes now pass with inherited autocrlf=true.
The next native Windows qualification remains pending.

Local expanded suite: 190 passed, three real-Win32-only skips. Before that
passing run the previously recorded CPU router discrepancy recurred with the
same 24/42 weights and maximum 4.6253204e-05 difference; IDs stayed identical.
Added FP64 diagnostic reporting on failure without changing the frozen oracle
or widening tolerance. Another 500 isolated comparisons passed. The root cause
of this intermittent host numerical discrepancy remains unresolved; a passing
rerun is not its repair.

## 2026-10-19 — owned stage construction and explicit draft graph source

Added actual decoder/vision/draft stage construction, complete native parameter
binding, exact FP32 boundary promotion/validation and local real Gloo-group setup.
Source 20 has a separate encoder-projection path. Target features are the mean
HC streams entering layers 37/38/39, preserving the official attention-input
convention rather than reading those layers' final outputs. Draft blocks use
bidirectional five-row attention over verified target context and never insert
proposal rows. Original Markov/confidence heads remain explicit operations.

Stage C's input embedding is an explicit publisher-verified 1.23 GiB replica,
not an implicit transfer or unrestricted ownership exception. The runtime source
identity manifest is packaged with the wheel. Complete application bootstrap,
distributed phases, sampling/acceptance, API/vision input handling and persistence
still require implementation. No GPU constructor/model operation has run.

Local suite: 192 passed, three actual-Win32-only skips. Updated constructor
prefix coverage uses the complete physical pack, including all draft matrices.
Installed CPU adapter imports passed; Windows CPU run 39329361879 also passed
the previous checkpoint. Native vLLM run 39329361817 remains in progress.

## 2026-10-19 — complete admission arithmetic and bounded resident transfers

Reconciled all 47,900 packed matrices and 896 native allocations (including C's
replica) against complete pinned headers. The plan counts native destination
dtypes, hot packed arenas, five-GiB source/index/shadow payload, generation/maps,
RoPE, FP32 pair state, checkpoints, workspaces and headroom before deriving
GPU expert slots. Missing experts and physical host/device shortfalls fail.
Metadata-only estimates are not measured peaks or residency proof.

Replaced B's whole-store CUDA registration with one shared 1 MiB pinned transfer
tile and a direct locked-to-pinned native copy. Original canonical pages remain
VirtualLocked; file handles stay closed. Added real QueryWorkingSetEx inspection
of every canonical page after preload and GlobalMemoryStatusEx host counters.
These Win32 API extensions await the next actual Windows CPU run.

Native ABI 2 adds CUDA-LUID-matched DXGI local/nonlocal budget counters. No
telemetry number is fabricated on a CPU host. The guard uses a pre-allocation
baseline and explicitly does not claim permanent GPU pinning. Windows native
compile/link/import and actual GPU budget queries remain pending. Local suite:
194 passed, four actual-Win32-only skips. The CUDA arithmetic kernels are
unchanged; the previous ABI 1 build does not qualify the added native query.

## 2026-10-19 — ABI 2 qualification and sampled verification arithmetic

Windows CPU run 39334269398 passed actual Win32 locked-page/direct-copy/host
memory checks. Native run 39334269334 passed ABI 2 compile/link, import,
SM89/SM120 inspection and import without the compiler SDK on PATH. Reports
are retained in native-abi2-build.json and native-abi2-fatbin.log. Actual GPU
counter queries, kernels and full B residency have not executed.

Added complete local terminal probability controls, independent transactional
target/draft/accept PCG64 domains, exact speculative rejection correction and
explicit processed-row/pending-output decisions. Fourteen focused tests passed,
including exact rational marginal recovery and every rejection prefix. Normal
rejection retains all draws consumed by the proposal algorithm; reusing the
rejection-conditioned proposal suffix could bias subsequent proposals. A full
uncommitted abort restores all three domains. Production integration is pending.

vLLM native run 39329361817 reached 80/203 units and failed on an upstream
Windows INFINITY macro expanding to an overflowing float cast. An additional
exact-source patch represents the same negative infinity via CUDA numeric_limits.
The next actual compiler run must validate it.

## 2026-10-19 — actual startup assembly and phase/terminal control source

Added local publisher-pinned deployment metadata, exact configuration of the
real vLLM classes, current Windows budget admission and owned stage bootstrap.
The bootstrap eagerly allocates every BF16 pool, preloads B's locked canonical
weights, binds complete packed/native/expert catalogues and original Engram,
loads the actual graph and invokes the small CUDA checks. It returns LOADED;
distributed readiness still needs the peer barrier and real graph qualification.
On this CPU host the real vLLM config reaches its explicit CUDA-only Engram
platform guard; no native GPU constructor is claimed as tested.

Stage C now applies actual graph Markov-conditioning calls around the local
sampler. Target distributions are lazy so an accepted EOS/length limit does
not request undefined grammar state afterward. The phase ledger enforces full
fixed-chunk encoder and target passes, all-stage receipts, separate processed
and pending positions, duplicate-commit identity and fault/reset rules. These
are production components, not yet a complete worker/API/coordinator service.

Added a CUDA tensor-wire implementation with one fixed MiB pinned tile and
bounded CPU staging per connection. Its metadata checks share the exercised
fragment cursor. Transport/state/three-process CPU regressions: 43 passed;
sampling/admission/phase checks: 30 passed. Actual GPU transfers remain unrun.

Full EXLlama run 39326981917 passed compile/link, install, Python/native imports,
both SM cubins and pip consistency. This run predates the separate SDK-free
native-import check; the newer run 39336142655 must qualify that requirement.

## 2026-10-19 — actual execution boundary, identity and persistent control I/O

The execution component now connects actual CUDA forwards, remote BF16
source/index rows, current top-k/candidates, independent slot maps and original
Engram begin/replay/commit to the production phase ledger. Source closures use
absolute parity and operation hashes bind the exact original tokens/masks.
Forced recovery quiesces CUDA and invalidates a faulted session under a newer
epoch without unloading its weight allocations. GPU recovery remains unrun.

Added fixed-capacity original token/image history and separate generated-token
counts. Repetition penalties use all token counts; presence/frequency penalties
use generated tokens. Terminal decisions bind those counts, controls, grammar
identity and RNG state. Prefix digests include original image/processor keys;
they cannot treat compressed-token-map equality as exact history equality.
Runtime peer identity now hashes installed application/component source and
native binaries, in addition to pinned versions, model metadata and precision.

Target prefill no longer computes the vocabulary head for every prompt row:
only the final target prompt position produces seed logits. Verification still
projects all bounded input rows, and every target row still updates attention
and draft context. This preserves full CED while avoiding unnecessary head work.

Added one-owner TLS control pumps, correlated bounded RPC queues, compact error
codes, deadline invalidation and certificate-derived mesh peer/plane acceptance
independent of connection arrival order. Real loopback TLS tests exercise replies
and cancellation while an execution-queue fixture waits. These tests do not
emulate or qualify model execution. Production service dispatch, coordinator,
API/vision, prefix checkpoint reuse and snapshots remain incomplete.

Expanded local suite before the latest control additions: 234 passed, four
actual-Win32-only skips. The application wheel built and its deployment metadata,
history operations and packaged source identity were inspected after installation
outside the checkout. Windows CPU run 39337612127 passed checkpoint 65631c7.

The complete follow-up then caught the known CPU router discrepancy: 238 passed,
one failed and four Win32-only skips. FP64 comparison identifies the Torch side
(up to 4.6e-5 on the first four rows), with NumPy below 7e-8. Added observation
of the real ATen intermediate results to the failure diagnostic without editing
the frozen Gate body or its comparison tolerance. The next full run and three
diagnostic follow-up suites each passed 239 tests/four skips; 3,000 isolated
observed Gate calls and an alignment-varied raw-linear check did not reproduce
the failure. The host numerical issue remains open; none of those passes is a
causal repair. Native Windows/runtime Torch qualification stays separate.

## 2026-10-19 — stage services, full-prompt transfer and cluster scheduling

Added the real stage dependency mesh, explicit stage-stream CUDA transfers,
bounded serialized model actors and production service dispatch. Control I/O
handles heartbeat/cancellation without waiting for the model mutex; cancellation
invalidates old queued work. Full prompts travel as bounded I32 tokens, U8 masks
and exact original image-reference bytes, with full validation before GPU reset.
A real loopback TLS test transfers 1,048,576 tokens, including an image span.

The cluster coordinator source now schedules both complete fixed-chunk CED
passes, the full encoder barrier/rewind, terminal seed sampling, actual five-row
draft proposals, all-stage target verification and accepted-prefix finishes.
Normal decode has explicit disabled/context-tail reasons. Output publication
requires all three actual GPU receipts, all three matching history commits and
Stage C RNG finalization before the API can receive a token batch. Metadata/RNG
tests reject partial or prepared-only acknowledgment proofs and conflicting
history. Actual graph/network coordination on GPUs remains unrun. Startup,
interrupted mesh re-pairing, API/vision, reuse and snapshots remain incomplete.

The complete expanded local suite passed 250 tests with four actual-Win32-only
skips. The intermittent Torch CPU reference issue above remains open; this pass
does not explain or repair it. Windows CPU run 39340172589 passed a24b96f.
Project ABI 2 native run 39340258236 passed compile/link/install/import,
SM89/SM120 inspection and SDK-free import. Full EXLlama run 39336142655 passed
the corresponding native checks, SDK-free native import, Python package import
and pip consistency; its report is retained as exllama-sdk-free-build.json.
Full vLLM native run 39336142640 remains in progress. None ran GPU kernels.

## 2026-10-19 — original images and transactional stop handling

Vendored the unchanged original October image processor with its source hash.
Bounded input inspection/expansion preserves the original image ID, pad-free
span/grid, BF16 patches, original bytes and processor/model identity. Dedicated
opaque storage has pinned assets, payload/entry/RAM budgets, atomic writes,
checksums and <=64-row reads. The actual A tower/aligner resolves a single image
at a time and reuses its exact stored BF16 features for both CED passes. Its
attention permits fused Torch SDPA only; quadratic math fallback is disabled.
GPU image execution remains unrun.

The tokenizer byte decoder matches the real Rust tokenizer across all 129,280
IDs and preserves split UTF-8. Transactional stops buffer partial stop strings,
bind the prefix state into C's sampling digest and trim accepted speculative
rows before GPU commit. Initial/replacement tokens remain distinct pending
inputs. A stopped request cannot admit another decode window. New CUDA checks
cover the maximum vision attention shape and odd/partial compressor rollback;
their actual GPU execution remains unrun.

Windows CPU run 39342333068 passed 5684afb. Original image/storage focused
checks passed 12 tests, and stop/sampling/metadata checks passed 30. The earlier
expanded image suite passed 257 tests/four Win32-only skips. The following full
stop suite caught the intermittent reference failure (264 passed, one failed,
four skipped), narrowing it to real ATen sqrt; see DSR-AUDIT-011. With added
failure-only diagnostics the next full suite passed 265 tests/four skips. This
does not resolve the host issue. No comparison tolerance was changed.

## 2026-10-20 — actual streaming parser and next Windows profile repair

The actual installed vLLM 1017 parser now uses a preserving adapter: original
DSML string attributes determine value types, literal arguments/input wrappers
remain intact, and reasoning/content whitespace is retained. The original
two-newline block delimiter is handled across arbitrary token chunks. Complete
responses must agree with the unchanged original parser before successful API
completion. The real installed tokenizer/parser passed 20 checks spanning
Unicode, whitespace, multiple calls, contradictory schemas, literal wrappers
and long arguments across 1/2/5/6-token chunks. This is Python protocol evidence,
not model execution. The API endpoint is still being integrated.

Windows CPU run 39344050108 and project native run 39344050177 passed fdf1247.
The installed image/stop wheel passed original-processor hash and package
identity checks outside the checkout. Full vLLM native run 39336142640 failed
at unused FlashKDA/Qutlass extensions after reaching 107/203 compile steps;
see DSR-AUDIT-012. Added a fifth exact-source patch for the explicit owned
profile, preserving core/MoE targets and common Python requirements. Fresh
five-patch application, final-patch idempotence and setup syntax passed. Native
compilation and the complete Windows runtime dependency bundle remain pending.

## 2026-10-20 — HTTP ownership and persistent mesh lifecycle source

Added bounded OpenAI JSON/SSE serving and original prompt/image preparation,
late-cancel request scoping, pre-load diagnostics, strict shared/local Windows
configuration and persistent worker launchers. Every plane is bound to one
authenticated mesh incarnation. Real three-role TLS tests cover loading-time
control, interruption/reconnect, stale incarnation and duplicate-plane refusal.
Actual background Uvicorn startup/loading readiness, port conflicts and real
exclusive process-directory ownership were exercised without a model.

The focused startup/API/mesh suite passed 19 tests; transport/mesh/actor passed
22. Frozen Expert/Block tests cover repaired BF16 and MoE accumulation boundaries
(DSR-AUDIT-013). The expanded full suite has 304 passes, four Win32 skips and one
failure at the existing Linux CPU sqrt discrepancy. The new bounded synthetic
capture confirms immediate bad output with no mutation and accurate subsequent
calls. No tolerance was changed or later pass substituted for this failure.
Windows runtime dependencies resolve to 155 hash-pinned packages; installed
Windows import/pip consistency is still pending. Full vLLM/EXLlama jobs at
fad5701 remain in progress. Persistent Windows CUDA model/API recovery is unrun.

## 2026-10-20 — exact prefix checkpoints and state persistence source

Added unanimous token/image/component prefix negotiation and the real encoder
SWA/FP32-pair checkpoint. Large source/index/shadow rows stay in their admitted
pools and receive a new local epoch; tail validity is cleared. Reuse keeps the
original fixed chunk boundary and still performs complete exact target replay.
Missing original features or changed image bytes prevent reuse. The focused
prefix/history/pool/phase suite passed 35 tests.

Implemented state-only snapshot inventory, bounded archive/TLS streaming,
host-owned atomic manifests and two-pass restore. Every complete stage archive
is checked before any GPU application. Windows file handles deny write/delete
sharing through both passes. State includes valid BF16 rows, FP32 open pairs,
exact history/images, encoder checkpoints, pending inputs and C's RNG/stop state.
Restore establishes fresh independent slot generations, rebuilds original
Engram hash history and rechecks B's locked weight pages. All three apply
receipts precede readiness. Actual GPU save/restore remains unrun.

Added explicit committed-boundary pause and continuation without reseeding or
re-forwarding an already processed input. Manual continuation exposes bounded
NDJSON token/text events; ordinary OpenAI completions retain their preserving
parser. Host HTTP and CLI entry points connect these source paths to the actual
coordinator. No model is emulated by the persistence or API tests.

The prefix/archive full suite passed 323 tests/four Win32-only skips. The later
integrated snapshot/session suite passed 332/four skips; focused client/API/
archive/inventory/prefix checks passed 36. These passes do not resolve the prior
captured Linux CPU sqrt failure (DSR-AUDIT-011). Windows CPU run 39350431569 and
project ABI-2 native run 39350431617 passed 8100979. Full pinned vLLM/EXLlama jobs
39345608248/39345608258 continue. No GPU kernel/model has executed here.

The application wheel built and installed outside the checkout. Snapshot,
restore, resume and private-client modules imported from that installed wheel;
application_source_identity covered 89 files with SHA-256
0a2f6d5ff91409dc09a7ccee969bb20362c4bbf6d0baaf60db9827691a238e97.
Installed snapshot CLI help exited successfully. This is packaging/import
evidence, without native model construction or full-model qualification.

## 2026-10-20 — isolated Windows bootstrap, runtime assembly and diagnostics

Windows CPU runs 39355257126 and 39355507560 passed the credential/bootstrap
source. Bootstrap run 39355507561 passed actual Windows PowerShell 5.1,
checksum-pinned uv 0.12.23 and CPython 3.12.14, two isolated installs, unchanged
user/machine PATH and refusal to mark an application installed after bootstrap
alone. The earlier Get-FileHash command failure was repaired using streaming
.NET SHA-256. Fresh P256 role credentials use neutral subjects and private
current-user/SYSTEM Windows ACLs; real mTLS and ACL checks passed.

Full pinned EXLlama run 39345608258 passed compile/link/install/import,
both architectures and dependency consistency. Its retained wheel SHA-256 is
980945a64b84da802ebae5529244d50b887861391e251d3a2fc65347a7226bf1.
Full vLLM run 39345608248 reached step 163/189, then hit its three-hour timeout
while building FlashAttention. The same complete profile was restarted as
39400420994 with a six-hour limit. It remains unqualified until completion.

Added complete five-wheel Windows bundle assembly, app-local MSVC runtime,
hash-locked external dependency installation, bare-system-PATH native/source/
concrete-class/original-parser qualification and an atomic completion marker.
Launchers check the completed bundle and private interpreter. This full runtime
assembly/installation workflow is implemented but has not yet run successfully;
the bootstrap pass is not substituted for it. Corresponding-source links,
exact upstream pins and local patches are included in the bundle source offer.

Expanded actual GPU self-test source covers all admitted packed K values,
5120/2304 projections, independent trellis/layout/rotation equations, width-5120
mHC, target/draft image-aware routing, stream/event dependencies and rollback
lengths 0..6. Required diagnostics run before full weights/eager cache allocation
and release their temporary allocator cache before admission. Synthetic native
file/Engram/locked-memory checks and Windows effective CPU feature reporting
are shipped too. GPU execution is still NOT RUN.

The expanded CPU suite finished with 347 passes, four real-Win32 skips and one
failure at the previously captured Linux Torch sqrt discrepancy, unchanged
comparison tolerances. A cold-import policy subprocess also exceeded its old
20-second startup deadline in an earlier focused run; its meaningful precision
assertions now allow 60 seconds for native Torch startup. The later full suite
passed that policy test. The cluster-gate/API/persistence/host focused suite
passed 33 tests after correcting its missing metadata fixture.

Actual installed-cluster probe and incarnation-bound speculation admission are
now integrated source (DSR-AUDIT-015). The probe invokes real stages and the real
snapshot coordinator; CPU receipt fixtures do not emulate a GPU model. Its
full-model parity/residency/image/restore checks remain unrun. The complete
candidate and independent Astra review remain pending.

## 2026-10-20 — operational measurements and actual client-adapter audit

Windows CPU run 39402801421 and isolated bootstrap run 39402801517 passed
9a3bb7a. Full vLLM native run 39400420994 remains in progress. Added fixed-name
numeric counters, monotonic histograms, bounded rotated logs, atomic periodic
metrics, stage CUDA-event spans and typed loading-area diagnostics. Cache hit
denominators, new/reused encoder work and full target replay remain separate.
RPC bytes exclude TLS framing; application file reads do not masquerade as
physical SSD misses. Unavailable hard-fault/pagefile/thermal sensors stay null.

Live host and offline per-role diagnostic archives read explicit public files,
not credentials or model/image/snapshot trees. The benchmark uses actual HTTP
SSE and authoritative usage, distinguishes headers from first visible output,
and refuses to attribute server counters across another observed operation.
No model throughput has been measured. The client benchmark test uses an
explicit synthetic HTTP protocol fixture, not an emulated model engine.

The operational full CPU suite passed 358 tests with four real-Win32 skips.
That pass does not repair the earlier captured Linux sqrt discrepancy. Focused
metrics/I/O/cache/transport/lifecycle checks passed 51/one skip, then 13; client/
diagnostic/benchmark checks passed 23, and final lifecycle/configuration checks
passed 27. Packed rotation bounds were reviewed before GPU execution: seven
butterfly sums, normalization multiplication and the rounded constant receive
a fixed gamma_9 bound. The mHC expansion bound now uses actual BF16 exponent
spacing. No GPU observation has been used to fit a tolerance.

Read-only OpenCode v1.19.3 source at e3c4eb242585dcd77be1a867391b7c25fd47ce52
identifies @ai-sdk/openai-compatible 2.0.41. Its actual integrity-locked adapter
passed three synthetic-wire checks for text/reasoning whitespace, split literal
tool arguments, image/tool-result/reasoning input and abort propagation. Both
captured actual request bodies pass the original Python tokenizer/template/image
preparation: 33 and 275 prompt tokens, 1M capacity and requested output 8192.
The clean npm-ci installation repeats this audit. No OpenCode process, user
Docker version, complete agent session or GPU model is claimed qualified.

The next installation check also examines actual Windows Triton compiler
requirements, not only DLL imports. Bounded inspection of the hash-pinned
Triton wheel finds bundled TinyCC, ptxas and CUDA headers/import library; the
pinned Python standalone archive contains Python.h, pyconfig.h and python312.lib.
Actual compiler/standalone-interpreter qualification is being added.

## 2026-10-20 — official eighth reference and measurement clock repair

DeepInfer 8b16149741a4935e1dec8cea27853f273b374c5b is hash-pinned as the eighth
source-only reference. No imports, upstream tests/examples/builds or model
payload downloads were run. docs/DEEPINFER_REVIEW.md records model/DSpark,
exact replay, confidence scheduling and sampler/regression findings. Its
bounded replay, 5e-5 biased correction and fixed-point sampling are not adopted.
Three additional independent fixtures check conditional two-token joint mass,
zero residual mass for equal coordinates and confidence-independent sampling.
The combined sampler suite passes 22 tests.

Windows CPU 39407156592 failed one of 362 tests at benchmark division by zero
under CPython 3.12's coarse monotonic clock. DSR-AUDIT-016 switches duration
measurements to perf_counter_ns and handles zero elapsed time without invented
throughput. Thirteen focused checks pass; the complete cloud CPU suite passes
362 tests with four actual-Win32-only skips (71.87 s). This passing run does not
repair DSR-AUDIT-011's previously captured CPU sqrt error. Windows follow-up
and full vLLM build status are currently blocked by GitHub HTTP 401 responses.

## 2026-10-20 — private JIT compiler policy and workflow qualification

GitHub API credentials recovered without user action. Windows CPU 39408672138
passed 441b344, including the real pinned SDK Node audit and original tokenizer/
template preparation of its captured text/image/tool requests. The same push
updated sources.lock.json for DeepInfer and unintentionally restarted vLLM:
39400420994 was cancelled; 39408672207 now builds the unchanged five-patch
profile. Running native builds are preserved on future pushes.

The full runtime workflow had an invalid plain YAML run scalar and had not
started a job. It is repaired and all seven workflow files parse with unique
mapping keys. This is syntax evidence, not installed-runtime qualification.

The actual pinned standalone Python archive and Windows Triton wheel are
inspected and their compiler/development inputs hashed. Before-import policy
pins bundled TinyCC/ptxas/CUDA inputs, private cache and IEEE FP32 dot default,
refuses interpreter/kernel/architecture overrides and binds compiler components
into installed stage identity. Added a CPU-only native Windows compiler-subset
workflow and a required full-installer check. These build/load an actual small
CPython extension and compile explicit-target SM89/SM120 cubins without a GPU
context. Their source is implemented; actual Windows results are pending.

The updated cloud CPU suite passes 363 tests with four real-Win32 skips
(62.97 s). Sixteen focused precision/policy/configuration/memory checks pass.
A passing run does not repair DSR-AUDIT-011. No GPU/model qualification is
claimed. Final runtime evidence copies only explicit bounded JSON reports;
regenerable compiler cache binaries and paths are not distributed as evidence.

## 2026-10-20 — actual compiler subset reaches Blackwell ELF inspection

Windows CPU 39410453663 passed 0e2bcfc, including current policy and SDK client
checks. Compiler subset 39410453853 built/loaded the actual TinyCC CPython
extension, compiled SM89 and compiled SM120 PTX/cubin, then failed its old ELF
field inspector. It captured ABI 8 flags 0x0600780a (SM120 in bits 8..15), while
ABI 7 uses bits 0..7. The pinned LLVM header confirms the two masks; new fixtures
include those actual observed bytes. The corrected full compiler run is pending.
No GPU driver/context or kernel execution is claimed.

README/capability/traceability text is refreshed to distinguish integrated
runtime source from missing hardware/install qualification. Earlier foundations
are no longer inaccurately described as unimplemented adapters/API/Engram paths.

## 2026-10-20 — Windows compiler and actual CUDA-Torch CPU qualification

Standalone compiler workflow 39411095629 passed source 27e7de9 with private
Python 3.12.14, real TinyCC CPython extension build/load and actual offline
Triton SM89/SM120 compilation. It verifies no GPU driver/context was initialized.
SM89: ELF ABI 7, flags 5834073, 2856-byte cubin SHA-256
c965e053e5292d192e2c117895b3a7be3a2d845e101bc80aba84218192912abd.
SM120: ELF ABI 8, flags 100694026, 5328-byte cubin SHA-256
3f54cded957bdc75e2f5d9a68624fc916a21c76ef83209b53f8ae274f9ae1adf.
Interpreter/kernel/architecture mutation guards passed. These are compiler
results, not GPU kernel execution or full-runtime installation qualification.

Windows CPU 39411095812 passed 374 tests/one intentional non-Windows refusal
skip, plus actual SDK client and outside-checkout wheel checks. Project native
39410211028 passed 6f0f086 ABI 2 compile/link/import and both architecture checks;
its CPU suite against actual Windows Torch 2.11+cu130 passed 366/one skip. No GPU
was present. The later cloud packet/ELF suite passed 374/four actual-Win32 skips
(63.96 s). DSR-AUDIT-011 remains open despite these passing suites.

The neutral checkpoint-27e7de9-windows-cpu-compiler tag is pushed as a precise
build/CPU-tested rollback point; it does not imply GPU/model/review approval.
The runtime installer workflow automatically follows a successful native vLLM
build on main and still validates all component/profile/source identities.
A whole-source review guide and packet builder are prepared. Packet creation
requires successful same-commit native Windows CPU and sealed complete private
runtime evidence; no intermediate candidate is offered as a finished review.

## 2026-10-20 — first-run device diagnostic state ownership

Source review found DSR-AUDIT-019: self-test could create a JIT cache before
claiming its state root, causing normal startup to reject it as an unrelated
nonempty directory. Device preflight now owns/locks that directory for the
entire diagnostic lifetime and releases it on both outcomes. Regression tests
use actual file locks and cache directories with an explicit I/O observer;
no GPU/model backend or result is simulated. Fifteen focused checks pass.
Windows CPU 39412586993 passed preceding f264367. Complete vLLM build and
installed-runtime qualification remain in progress/pending; no GPU result or
independent review is claimed.

## 2026-10-20 — reference path signature and actual local group qualification

DSR-AUDIT-011 now has a precise instruction signature: AVX2 input * hardware
rsqrt approximation reproduces all 1344 bad captured values bit for bit. MKL
AVX2 enhanced-performance mode also matches; high accuracy does not. The ATen
wrapper explicitly requests high accuracy. Selection cause remains open; no
oracle/tolerance/runtime override was changed. Tiny synthetic arrays, numeric
reports and project-owned probe source are preserved in cpu-sqrt-diagnostic/.

The installer now additionally checks actual local FileStore/Gloo startup and
one-rank CPU metadata collective, with spaced/non-ASCII temporary paths,
repeat initialization and complete group/file cleanup. It does not use Gloo
for inter-PC or GPU tensors. The exact function repeats successfully on the
Linux development CPU; its Windows check is newly added to CPU CI and the
complete private-runtime gate. The full cloud suite passes 377 tests/four real
Win32 skips (77.93 s); the real installed empty-vLLM source audit passes again,
including the concrete routed MoE class check. These are not GPU results.

## 2026-10-20 — actual Windows local-rendezvous failure and in-memory repair

Windows CPU run 39415738104 at 276178f passed 379 tests but exposed Torch
FileStore's failure opening an existing non-ASCII directory on Windows. The
startup path could have the same failure. DSR-AUDIT-020 now uses a real Torch
HashStore for the exactly one local rank, before the real vLLM world/model
setup. Startup and complete-runtime qualification share that initializer;
inter-PC TLS is unchanged. Tests exercise real Gloo collectives, repeated
creation/destruction and fresh-worker refusal without a namespace substitute.
The complete cloud CPU suite passes 378 tests/four actual-Windows-only skips
at unchanged mathematical tolerances. Native Windows confirmation follows the
push. Long native vLLM 39408672207 remains in progress and is preserved.

The saved cloud configuration draft now includes the eighth DeepInfer source
review and current whole-runtime continuation gates. Saving is confirmed;
publication is a separate environment settings action, not an implementation
blocker. No new secret, repository or model download was introduced.

## 2026-10-20 — qualify the actual Windows-supported Store contract and lifetime

Windows CPU 39416751099 rejected the HashStore attempt: Windows Torch omits
that native binding (379 other tests passed, two new Gloo checks failed, one
intentional skip). Pinned Torch 2.11 source confirms the omission and its
supported Python Store/C++ trampoline contract. This auxiliary API reference
has its own immutable source hashes; no upstream implementation was copied.

LocalRendezvousStore is an actual bounded, thread-safe Torch Store, used by
real Gloo on both startup and installation qualification. It has no filesystem
rendezvous or TCP rendezvous listener. Actual C++ PrefixStore tests exercise
get/set, counters, compare/set, waits, timeout, cloning, deletion and rejection
of oversized state. The C++ trampoline needs the Python owner kept alive:
a real extra Gloo group failed after that owner was dropped. Loading now
retains the owner through all actual vLLM model/world teardown, including
partial initialization failure. Qualification forces collection, creates an
additional real group, runs its collective, destroys all groups and repeats.

The complete cloud CPU suite passes 379 tests/four actual-Windows-only skips;
focused Store checks follow the final timeout/bounds addition. Native Windows
confirmation and complete private installation are still required. Long vLLM
39408672207 is unchanged and preserved; no GPU/model result is asserted.

## 2026-10-20 — Windows Store confirmation and bounded expert-event backpressure

Windows CPU 39418074217 passed clean be25537: 382 passed/one intentional
non-Windows refusal skip, actual Gloo additional-group/owner-lifetime checks,
wheel build/install and actual OpenCode SDK text/image/tool/reasoning/cancel
checks. The neutral remote rollback tag is checkpoint-be25537-windows-cpu;
an exact source archive is retained externally. Installed Linux wheel/source
identity and actual Gloo subgroup checks also passed outside the checkout.
Neither result qualifies a GPU or the complete private Windows installation.

DSR-AUDIT-021 fixes a source-level cache progress problem: an admitted one-slot
cache could reject the next expert if its unleased victim still had GPU work.
There is no routed caller retry for that rejection. It now waits on the actual
completion event with a 30-second bound and independent cancellation, keeping
all old storage intact until completion. Live leases remain non-evictable.
The closed metrics inventory includes expert_slot_wait. Real host-event tests
cover pending completion, cancellation and timeout without claiming GPU proof.
A real CUDA diagnostic uses one slot, finite GPU delay, complete synthetic
packed bundles, actual EXL3 calls, exact reused-output comparison and stale
lease refusal. It is implemented but unrun. Focused suite: 14 passed. Complete
cloud suite: 382 passed/four actual-Windows-only skips, unchanged tolerances.
The long native vLLM build is preserved and full installation is still pending.

## 2026-10-20 — Windows expert-wait confirmation and shared small-row tuning

Windows CPU run 39419557963 passed clean f4844a5: 385 passed/one intentional
skip, wheel/install and actual OpenCode SDK audits. This includes the host-event
wait/cancel/timeout coverage, not execution of the added CUDA cache probe.

Source review also found EXL3_GEMM_SMALLM_SHARED_TUNE, a separately cached
selector whose inherited value 0 disables the upstream shared row-1..8 tuning
bucket. DSR-AUDIT-022 pins it to the intended default 1 before import and checks
later mutation alongside all existing EXL3 precision selectors. Fourteen focused
policy/routed/wait tests passed. No native source/profile changed or build was
restarted. Actual CUDA row invariance remains an independent unrun gate.

## 2026-10-20 — prequalify the full private external dependency stack

The standalone Windows compiler workflow now follows its real TinyCC/Triton
checks with installation of the entire hash-locked external runtime stack into
the same pinned private Python 3.12.14. It runs actual pip check, CUDA-Torch
2.11/vision/audio/TVM imports, Gloo subgroup lifetime/cleanup and unloaded API
startup under a bare system PATH with GPUs hidden. This independent gate can
find dependency/import failures while the native vLLM build is still running.
It does not substitute any native library or claim the complete five-wheel
runtime is qualified. All seven workflow files and both nested Python child
programs parse locally; actual native Windows execution follows the push.

## 2026-10-20 — restored cloud verification and early offline/privacy policy

The restored cloud instance retained clean e298050, the pinned Python tools,
reference sources and evidence. GitHub API access works through the existing
platform credentials. Rebuilt/installed the current application wheel outside
the checkout; pip check, nine focused readiness checks, five actual installed
CPU vLLM/plugin checks and twenty original parser/tokenizer cases passed. This
validates the restored development instance, not any CUDA/model execution.
The saved repository configuration now records the verified existing
DeepSeekRPC mount and exact e298050 commit. Filesystem discovery found only that
actual project checkout; the empty workspace .git directory and uv's empty
cache marker are not repositories. Draft saving was confirmed; publication is
a separate product action and was not inferred from a reconnection event.

Windows run 39420558211 passed the exact e298050 CPU/build/install/client suite.
Run 39420558237 passed the private Python/TinyCC/Triton compiler checks plus the
entire locked external dependency installation, pip check, actual CUDA-Torch
2.11/vision/audio/TVM imports, Gloo subgroup lifetime and unloaded API checks
without the compiler SDK or visible GPU. Native application installation still
requires the actual vLLM/EXL3/project wheels; long vLLM 39408672207 is preserved.

DSR-AUDIT-023 establishes offline and telemetry opt-outs before upstream imports
and rejects an already cached enabled vLLM reporter. No unexpected transfer was
observed. The full local follow-up passed 381 tests/four actual-Windows-only
skips and reproduced DSR-AUDIT-011 once, with the same reduced-accuracy sqrt
signature. Its fresh synthetic capture is preserved externally. No oracle or
tolerance changed; a later pass would not resolve that vendor-library finding.

## 2026-10-20 — actual late DeepGEMM failure and corrected EXL3 native target list

Long Windows vLLM 39408672207 at 441b344 ran 2h24m and failed at 187/189:
DeepGEMM's nested deep_jit uuid.hpp includes Linux unistd.h. Real core and MoE
DLL links had passed at 185/186; no complete native wheel was produced. Logs
and build provenance are preserved externally. Automatic complete-runtime run
39426412511 was correctly skipped after this native failure.

The fifth exact-source owned-profile patch now excludes the unselected
DeepGEMM host extension in both CMake and setup. The BF16 attention/index/state,
packed EXL3 experts and explicit mHC implementation are preserved. The patch
and final file hashes were regenerated against the exact four-patch predecessor;
real apply/hash validation and idempotent reapplication passed. Runtime policy
pins VLLM_USE_DEEP_GEMM=0 and detects cached selection. The new native build
checks actual core and MoE/router imports and SM89/SM120 images, including
SDK-free imports; complete runtime qualification repeats both registrations.
Live build progress now prints bounded counters every 45 seconds.

Early offline/privacy policy passed all five real installed CPU vLLM/plugin
checks; the actual upstream usage reporter is disabled. The full local suite's
one failure remains the preserved DSR-AUDIT-011 CPU sqrt issue (381 passes/four
Windows-only skips). No numerical tolerance was changed. Focused policy/Gloo/
review-packet checks validate the current repair; actual Windows retry follows.

## 2026-10-20 — exact wheel metadata, current decoder hooks and private model metadata

DSR-AUDIT-025 admits the actual upstream metadata version
0.31.0+deepseekrpc2.cu132; SDK 13.2 adds that compiler suffix. No wheel is renamed
or repacked. A fresh supported empty CPU build contains all five current
patches; actual installed audits now check admitted vLLM source hashes and both
decoder factory parameters. Target Torch 2.11 CPU integration/parser checks and
pip check passed. Windows CPU 39434657393 and compiler/external-stack
39434657502 passed e97591d.

DSR-AUDIT-026 keeps vLLM cache/config metadata under owned worker state before
upstream imports. The actual CPU model-inspection writer saves its bounded
JSON there, then reaches the expected CUDA-only Engram validator; that probe
does not construct a successful complete CUDA model configuration. Seven real
installed integration checks pass. Windows CPU 39438027561 passed 3457a68 with
387 tests/one intentional skip; private compiler/external-stack 39438027672
also passed. Actual GPU/model execution remains unrun.

## 2026-10-20 — retained native notices and immutable Windows source handles

Commit b319948 retains thirteen exact native dependency license/AUTHORS files,
with parent/submodule revisions, lengths and SHA-256 values. Bundle assembly
checks them against actual fetched source provenance. Windows CPU 39441370032
passed 387 tests/one intentional skip. No native profile changed.

DSR-AUDIT-027 at 30b24d3 takes native deny-write/delete read handles before
weight/Engram validation and holds them until close. Snapshot readers preserve
their existing policy through the shared helper. Focused cloud checks passed
47/three Windows-only skips; the complete target Torch 2.11 suite passed
384/six Windows-only skips. Actual Windows CPU 39461505378 passed 389/one
intentional skip, including concurrent read/write/delete/replacement, existing
writer refusal, Unicode paths and failed-constructor cleanup. Both installed
cloud environments pass seven actual integration checks and pip check. Their
105-file application identity matches the actual Windows wheel.

Four pass-through observed Torch 2.13 CPU suites at preceding checkpoints did
not reproduce DSR-AUDIT-011. They do not explain or repair the preserved bad
sqrt bytes. No oracle, tolerance, vendor library or production math setting was
changed. The observer binary is external diagnostic tooling and is not shipped.

## 2026-10-20 — complete native vLLM and isolated Windows runtime pass

Native vLLM run 39428534610 at b8d3664 completed successfully after 3h50m.
The exact five-patch profile passes build/install, actual core and MoE/router
imports without the SDK, and SM89/SM120 inspection for both libraries. Wheel:
vllm-0.31.0+deepseekrpc2.cu132-cp312-cp312-win_amd64.whl, 210468623 bytes,
SHA-256 e6fe16765e0f908df5c0c7213e71a90b5e0f506d04e4e1ee474a9ec3af3d80ea.
Earlier failed/cancelled runs remain in this ledger; this pass does not erase
their diagnostics or the separate CPU sqrt finding.

Automatic complete-runtime run 39462821561 passed source 30b24d3 using that
wheel, qualified EXLlama 39345608258 and project ABI 2 39410211028. It builds the
actual application and plugin wheels, assembles app-local CRT and all source
notices, installs the full hash-locked dependencies into private Python 3.12.14,
then passes all eight required qualification checks. TinyCC extension load,
offline Triton SM89/SM120 compilation, bare-system-PATH native imports, actual
Gloo subgroup lifetime, concrete classes and twenty original parser cases pass.
All 81 retained files (247298333 bytes) match the sealed manifest after download.
The bounded actual reports are committed in evidence/whole-runtime-30b24d3/.

The final source/status freeze changes no native profile. Delivery still
requires matching Windows CPU and complete-runtime evidence for that exact
frozen commit, enforced by tools/build_review_packet.py. GPU/model execution,
full-size B residency, 1M behavior, distributed recovery, images and performance
remain unrun/unmeasured. Independent Astra review remains pending and is
user-mediated; no early hardware trial is requested.
