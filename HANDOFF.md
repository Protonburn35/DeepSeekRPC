# DeepSeekRPC engineering handoff

Goal: complete native Windows three-PC October 1017 EXL3 2.9 application under
docs/IMPLEMENTATION_SPEC.md. The user explicitly authorized implementation and
selected this existing repository; personal Google Cloud VM assumptions are
obsolete. Current stage: M0 design/audit plus M1 foundation and partial M2 storage.
This is not the first hardware-test release candidate.

Repository: `/workspace/DeepSeekRPC`. Reference sources/archives/metadata:
`/workspace/deepseekrpc-references`. Python environment:
`/workspace/deepseekrpc-tools/.venv`. Logs/current build evidence:
`/workspace/deepseekrpc-evidence`. All source pins and metadata hashes are under
provenance/. Astra's controlling prompt and the older handoff precedence are
recorded in docs/DECISIONS.md.

## Verification status

1. Source implementation: incomplete. See docs/REQUIREMENTS.md for every missing
   mandatory runtime component. Do not call a transport fixture an executor.
2. Native Windows compilation/link/package: pending. CPU Windows workflow is
   running in draft PR #1. The first native Windows job passed 129 tests, one
   non-Windows-only skip, wheel build and install; the installed CLI path check
   failed because base Windows Python stores scripts in Scripts/. That check
   now uses sysconfig. Future CI builds are Windows-only by user request.
   CUDA/MSVC tools are absent locally. Python wheel building works on Linux.
3. CPU checks: ownership/header parsing, real three-process mutual-TLS TCP,
   fragmentation, commit/cancel/resume, journal rollback, bounded RAM/resident
   storage, original Engram encoded rows and transactional hash history, pinned
   tokenizer/compressed map, official encoding, independent routing,
   unquantized expert and mHC methods, all mul1 windows. Exact counts and source
   identity belong to generated BUILD.json/JUnit logs for each run.
4. Independent Astra review: pending; user-mediated, no automated reviewer.
5. Actual SM89/SM120 execution: NOT RUN.
6. Full model/vision/session/long-context validation: NOT RUN.
7. Performance: NOT MEASURED.

## Important findings

The plugin source compatibility checker exits 0 against the pinned Windows
vLLM tree; its NOTICE and documentation BAD lines are retained in evidence/.
An actual installed CPU-only empty vLLM build and plugin entry-point audit also
passed; this is Python integration evidence, not native/CUDA qualification.
Current upstream
attention rejects missing local shared caches, the plain path returns BF16,
and bounded SWA replay defaults true. Both required BF16 attention/index support
and cross-stage replicas need real adapters; no guards were removed.

Every one of the 192495 pack tensor headers is accounted for; quantization
metadata alone misses 5389 descriptors. The generated inventory includes four
original Engram table/scale tensors, omitting redundant original projections
already represented in EXL3. B owns 17602641264 weight bytes, not including
2013265920 prospective BF16 source/shadow bytes, scratch, copies or OS reserve.
This is not proof of physical residency or full 1M fit.

CPU Torch 2.13.0+cpu is an independently pinned mathematical reference tool.
It is not the qualified Windows runtime Torch (the source requirements select
2.11.0+cu130 there). uv's initial Torch wheel extraction failed; pip downloaded
and installed it with the same required hashes and normal TLS verification.
No unchecked artifact was used and no checksum was changed.

NumPy is now 2.3.5 to satisfy the pinned vLLM source's mistral-common dependency
(NumPy <2.4); build tooling uses setuptools 80.10.2 (source requires <81).
The actual CPU source-audit dependency stack passes pip check. See
docs/FINDINGS.md for the official/vLLM Engram scale-byte-zero discrepancy.

## Commands and next work

Activate the existing environment and use two CPU threads for reference checks:

```bash
cd /workspace/DeepSeekRPC
source /workspace/deepseekrpc-tools/.venv/bin/activate
export PYTHONUTF8=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python tools/ci_check.py
PYTHONPATH=src python tools/build_inventory.py --metadata-root /workspace/deepseekrpc-references/metadata
```

Next required gates: qualify M1 on native Windows, obtain user-mediated review
of the source/precision/memory/protocol design, then continue owned vLLM loading,
replica registration/scatter, exact CED scheduling, full BF16 attention/index,
EXL3 bounded execution/cache promotion and disk Engram graph integration. Native
The user granted ongoing pushes/CI authorization; do not ask again for routine
repository work. Keep a neutral author identity and personal details out of
generated output. Do not claim Linux compilation as Windows evidence. Later
integrate sampling/DSpark, actual
vision/API, exact prefixes and bounded snapshots, build both SMs, and prepare
repeated Astra review packets before asking for hardware trials.

No new repo/worktree, full model download, rented GPU, public service or changes
to the working StrataRPC installation are authorized by these development tools.

## Latest local follow-up

SelectiveWeights verifies the pinned index/header and full local publisher file
SHA with bounded reads, rejects unowned tensor access and post-admission file
mutation. LockedOwnedWeights preloads B into checked Win32 locked host pages and
permanently closes the disk source. This does not qualify GPU copies, staging,
physical WDDM residency or a full-model run. Exact ExLlama policy explicitly
zeros GEMV/GatedResidual INT8 before import (DSR-AUDIT-002).

The first expanded Windows run 39310450571 found an OS-specific negative-test
expectation: the client rejects an untrusted CA and Windows delivers its close
to the server as ConnectionResetError rather than SSLError. The test now accepts
that specific reset while still requiring authentication to fail. New loader,
policy and affected transport/metadata tests: 39 passed, two Win32-only skips
locally. Refreshed reusable cloud installer: 141 passed, two Win32-only skips,
pip check and native C++ toolchain check passed.

The user explicitly authorizes merges/direct main commits and wants the first
Astra review only after the full application is implemented, including all real
CUDA execution paths compiled for SM89/SM120. Keep working after successful CI;
do not ask for intermediate reviews or early hardware trials. Actual hardware
validation remains later, after repeated complete-candidate reviews and fixes.

Windows foundation checkpoint d2be910 is now integrated into main; draft PR #1
was closed after integration. CI run 39311940687 passed 142 tests, one
non-Windows-only skip, wheel/install/CLI. Archive:
/workspace/deepseekrpc-evidence/milestones/d2be910/DeepSeekRPC-source.tar.gz.
Native CUDA operation source and a pinned Windows compile workflow now exist;
its build evidence is pending. See docs/CUDA_BACKEND.md. Next continue actual
packed EXL3 device loading, bounded caches and the vLLM layer-stage graph,
transactional BF16 source pools/index selection, exact CED scheduling, Engram
projection/gate, and the remaining application features. Do not request Astra
review until the complete application is implemented.

Latest precision requirements: BF16-only, unquantized KV/index caches; no FP16
cache option and no FP8/FP4/INT8 fallback. Stage activation/pre_mix wire is FP32,
explicit --layer-stage-wire fp32. These current user instructions override the
original spec's cache precision language. Weight-kernel FP16 is separate.

Native CUDA run 39314752138 passed for 21fe6b7: compile/link/install/import and
sm_89/sm_120 cubins verified, 160 CPU tests passed with actual Windows CUDA Torch,
one non-Windows-only skip. GPU operations have not executed. Compact evidence
is tracked. Full EXLlama/vLLM native integration and the actual staged model
are still required. New CUDA cache code uses fixed slots and event-held leases;
B's host path registers already locked pages instead of a second host mirror.
