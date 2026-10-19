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
   prepared; no remote CI execution has been authorized or run at this point.
   CUDA/MSVC tools are absent locally. Python wheel building works on Linux.
3. CPU checks: ownership/header parsing, real three-process mutual-TLS TCP,
   fragmentation, commit/cancel/resume, journal rollback, bounded RAM/resident
   storage, original Engram encoded rows, official encoding, independent routing,
   unquantized expert and mHC methods, all mul1 windows. Exact counts and source
   identity belong to generated BUILD.json/JUnit logs for each run.
4. Independent Astra review: pending; user-mediated, no automated reviewer.
5. Actual SM89/SM120 execution: NOT RUN.
6. Full model/vision/session/long-context validation: NOT RUN.
7. Performance: NOT MEASURED.

## Important findings

The plugin source compatibility checker exits 0 against the pinned Windows
vLLM tree; its NOTICE and documentation BAD lines are retained in evidence/.
No installed vLLM/plugin/CUDA qualification is inferred. Current upstream
attention rejects missing local shared caches, the plain path returns BF16,
and bounded SWA replay defaults true. Both required FP16 attention/index support
and cross-stage replicas need real adapters; no guards were removed.

Every one of the 192495 pack tensor headers is accounted for; quantization
metadata alone misses 5389 descriptors. The generated inventory includes four
original Engram table/scale tensors, omitting redundant original projections
already represented in EXL3. B owns 17602641264 weight bytes, not including
2013265920 prospective FP16 source/shadow bytes, scratch, copies or OS reserve.
This is not proof of physical residency or full 1M fit.

CPU Torch 2.13.0+cpu is an independently pinned mathematical reference tool.
It is not the qualified Windows runtime Torch (the source requirements select
2.11.0+cu130 there). uv's initial Torch wheel extraction failed; pip downloaded
and installed it with the same required hashes and normal TLS verification.
No unchecked artifact was used and no checksum was changed.

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
replica registration/scatter, exact CED scheduling, full FP16 attention/index,
EXL3 bounded execution/cache promotion and disk Engram graph integration. Native
Windows runner/publication permission must be resolved with the concrete
workflow; do not claim Linux compilation as Windows evidence. Continue useful
local work while that handoff is pending. Later integrate sampling/DSpark, actual
vision/API, exact prefixes and bounded snapshots, build both SMs, and prepare
repeated Astra review packets before asking for hardware trials.

No new repo/worktree, full model download, rented GPU, public service or changes
to the working StrataRPC installation are authorized by these development tools.
