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
