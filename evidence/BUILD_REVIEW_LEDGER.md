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
