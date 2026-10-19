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
