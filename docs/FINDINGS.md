# Open source-audit and review findings

These are implementation findings, not independent Astra approval. Keep IDs and
dispositions across repair/re-review cycles.

## DSR-AUDIT-001 — Engram scale byte zero differs between sources

Severity: fidelity boundary to resolve during the actual disk-Engram adapter.

The official October `inference/model.py:ParallelEngramEmbedding.forward`
converts UE8M0 scale tensors with `scales.float()`. Standard Torch
float8_e8m0fnu represents byte zero as 2^-127. In the pinned Windows vLLM
`vllm/models/deepseek_v41/common/engram.py:_engram_lookup_kernel`, scales are
decoded using `(scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)`.
For nonzero finite bytes this agrees with the exponent encoding, but byte zero
becomes FP32 zero. The metadata-only source audit cannot establish whether any
valid row of the actual model uses scale byte zero; no real table payload was
downloaded or executed.

Current DeepSeekRPC's FP32 diagnostic decoder matches the official Torch
interpretation, including byte zero. Preserve raw table/scale bytes in the row
store. Do not replace this with the vLLM shortcut without resolving the semantic
difference. The full graph adapter must explicitly preserve original lookup
behavior and its BF16 output conversion. Add GPU regression coverage when that
path exists. CPU decode is not complete Engram graph qualification.

Evidence: all finite FP8 values across synthetic scale bytes 0/64/126/127/130
match Torch CPU exactly in test_all_finite_fp8_codes_against_pytorch_cpu. A named
byte-zero regression fixes the expected reference interpretation. Status: source
discrepancy documented; future GPU adapter and independent review remain pending.
