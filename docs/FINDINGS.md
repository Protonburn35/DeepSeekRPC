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

## DSR-AUDIT-002 — ExLlama cached defaults disagree across Python/CUDA

The pinned `exllamav3/util/backend.py` defaults INT8_GEMV_MODE to 2 on CUDA,
while `exllamav3_ext/quant/exl3_gemv_int8.cu` defaults the cached native mode to
zero. Documentation also calls mode 2 the default. Do not infer effective
precision from any one of these sources. `modules/hyperconnections.py` also
defaults EXL3_GR_INT8 to 1 for its gated-residual shortcut.

runtime_policy.configure_exact_exl3 sets both EXL3_INT8_GEMV and EXL3_GR_INT8
to zero before ExLlama import, refuses configuration after import, and detects
subsequent environment changes. Its subprocess regression starts with both
shortcuts enabled and verifies explicit zero plus refusal after mutation.
The future native adapter must call this before importing any ExLlama module.
This establishes policy code, not native execution or a full graph fidelity pass.

## DSR-AUDIT-003 — routed weight placement differs in the upstream EXL3 loop

Official October Expert.forward multiplies the FP32 SwiGLU activation by its
routing weight before converting that input for w2. The pinned plugin's
apply_exl3_python_loop instead multiplies the FP32 down-projection output by
that weight. These orders have the same exact-real algebra but differ under
FP16 input conversion and packed matrix execution. Do not infer full-model
equivalence from that algebra.

The bounded executor preserves the official pre-w2 weighting and asymmetric
clamp (gate maximum 10, up range -10..10). Its CPU activation regression
compares the unchanged official Expert.forward body and demonstrates that
moving the multiply across the half conversion changes the result. Actual
quantized GPU output and independent review remain pending. Shared experts
remain a separate contribution from the real vLLM runner.
