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
