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

## DSR-AUDIT-004 — CUDA mHC imports require an excluded Windows dependency

The pinned Windows vLLM requirements exclude TileLang because its TVM-FFI pin
conflicts with the Windows xgrammar/TVM-FFI requirements. However the CUDA
DeepSeek model imports TileLang mHC operations, and vllm.tilelang_utils requires
TileLang immediately on CUDA. An empty CPU import audit avoids this CUDA branch;
it cannot qualify the actual Windows model import/execution.

The real stage adapter needs an explicitly selected Windows CUDA mHC backend
and corresponding source patch, or a fully compatible validated TileLang build.
Do not remove the import guard and then leave a missing operation, inject
placeholder modules, or treat the native library build as graph qualification.
An explicit bounded CUDA mHC implementation now preserves the upstream delayed
pre/post equations using native Torch CUDA products and reductions. The exact
source patch binds this backend in the V4.1 decoder and defers unused TileLang
imports on Windows. Its arithmetic helpers compare against independent FP64
equations on CPU. Native Windows graph import, patched wheel qualification and
actual CUDA differential execution remain pending; the finding is still open.

## DSR-AUDIT-005 — packed compressor/index keys and independent output rotations

The complete pack has 668 non-routed packed matrices, including the head and
three draft blocks. The pinned compressor constructor and indexer's `wk`
constructor instead force dense weights. The input-projection helper directly
reads compressor `.weight`, and stock output projection assumes a different
grouped layout. Each of the eight `wo_a.slice.G` matrices has its own `suh` and
`svh`; merging those matrices under one shared rotation changes the model.

`dense_weights` explicitly maps every physical non-routed matrix, keeps one
admitted packed arena and binds the actual EXL3 implementation through the real
vLLM linear interface. Compressor projections explicitly return FP32. A complete
inventory check covers all 668 matrices exactly once. The layer constructor,
attention overrides and actual CUDA execution still require integration; this
finding remains open.

## DSR-AUDIT-006 — single-token EXL3 scratch is shared between projections

The pinned backend caches single-row Hadamard scratch by input shape, so
independent packed linears cannot safely launch overlapping single-row products
on auxiliary compute streams. The permanent packed store records and enforces
one compute stream. The complete layer adapter must also disable upstream
auxiliary GEMM overlap and join expert-cache work before consuming results.
This is an execution dependency, not a reason to change packed arithmetic.

## DSR-AUDIT-007 — MSVC rejects upstream SM12x quantized GEMM launch parameters

Native Windows run 39320169086 reached the actual compiler and failed with
C2719 in SM120 FP8 CUTLASS GEMM host launch stubs: parameters request alignment
128, which MSVC cannot pass by value. Both the official CUDA 13.2 and 13.0.2
headers retain 128-byte tensor-map alignment. Reducing that alignment in a
global CUDA header would change a GPU descriptor's ABI and is not an admitted
repair.

The exact-source build patch introduces an explicit Windows EXL3 profile that
omits the unused SM12x FP8 and NVFP4 matrix/MoE families. The selected application
uses EXL3 weights and BF16 vision/KV/index storage. Query FP8 preparation remains
separate and available. Capability probes now report only compiled dispatches;
an omitted SM12x call fails rather than dispatching to an unrelated SM80 kernel.
The profile changes no CUDA SDK file. Its Windows compile/link and runtime
execution still require qualification; this finding remains open.

## DSR-AUDIT-008 — FP32 router output does not establish FP32 router computation

The pinned V4 MoE creates GateLinear with out_dtype=FP32 but no forced compute
dtype. On the selected SM89/SM120 GPUs its default parameters can be BF16, and
the plain path can compute at that dtype before promoting the returned logits.
The exact stage-hook patch sets the existing force_fp32_compute option. For
these consumer architectures this creates FP32 router parameters and retains
the real GateLinear/selection graph. Native F16 checkpoint values can be
promoted exactly. GPU router comparison and full-model qualification remain
pending; a dtype label alone is not their substitute.

## DSR-AUDIT-009 — whole-store CUDA registration is unnecessary on B

The first uploader registered B's complete locked host allocation with CUDA.
The owned source is about 16.4 GiB, close to or above a 32 GiB Windows machine's
usual shared-memory allowance. No hardware failure was observed here; this is
an avoidable admission risk and a distinct contract from VirtualLock residency.

The uploader now shares one 1 MiB pinned transfer tile across boot/native/expert
copies. A native bounded memcpy reads directly from locked canonical pages;
the tile then uploads synchronously. Whole-store CUDA registration is removed.
Every canonical host page is checked using QueryWorkingSetEx after preload.
New DXGI budget queries support actual device admission/pressure monitoring.
Windows CPU run 39334269398 passed the actual all-page lock, direct-copy and
physical-memory checks. Native ABI 2 run 39334269334 passed compile/link/import,
both architecture inspections and import with the SDK removed from PATH.
Actual GPU counters and full-size hardware residency remain unrun; no absolute
WDDM pin guarantee is claimed.

## DSR-AUDIT-010 — Windows INFINITY macro is invalid in a CUDA float constexpr

Native run 39329361817 progressed to 80/203 compile units, then nvcc rejected
the grouped-top-k constant. MSVC's INFINITY macro expanded to a cast of 1e300
to float, which overflows during constant evaluation. The exact-source patch
uses the already included CUDA standard library's numeric_limits<float>
infinity directly. Its negative-infinity score semantics are unchanged. The
next native compile/link run must qualify this repair; no GPU result is claimed.

## DSR-AUDIT-011 — intermittent Linux CPU reference sqrt discrepancy

The frozen official router occasionally disagrees with the independent NumPy
baseline on this cloud host. The new real-ATen observer narrows it to sqrt:
matrix-multiply errors stay below 1.1e-6 and softplus below 5.9e-7 against FP64,
while sqrt can differ by 3.0e-4 on part of the [7,384] result. The resulting
router weight error reaches 4.6e-5. Expert IDs remain equal on the fixture.

This is observed with Torch 2.13.0+cpu, Intel MKL 2024.2, AVX512 and two Torch
threads. A subsequent diagnostic suite passed, and 16 fresh cold plus 16 warmed
isolated processes did not reproduce it. Those passes are not a repair. The
failure diagnostic now captures sqrt input/output immediately, checks later
mutation and repeats the same actual input; optional bounded synthetic array
artifacts support diagnosis if it recurs. Keep the frozen oracle body and
rtol=2e-6/atol=1e-6 unchanged. Native Windows CPU/runtime and GPU qualification
remain separate; no CUDA model discrepancy is established by this CPU failure.

## DSR-AUDIT-012 — unused FlashKDA/Qutlass extensions also fail MSVC alignment

Run 39336142640 passed the previous grouped-top-k infinity failure and reached
107/203 compile steps, then FlashKDA and Qutlass host stubs failed C2719 on
128-byte-aligned parameters. These implement GLA and NVFP4 paths outside the
selected DeepSeek EXL3 graph. The explicit Windows profile now omits these two
optional extensions in both CMake and setup.py while retaining core/MoE targets.
It does not change SDK headers or parameter alignment. The next native build
must qualify this adjustment.

The same owned profile retains the complete common vLLM Python dependency layer
plus pinned Torch/Triton and Windows runtime helpers. Unused optional CUDA
backends do not introduce unrelated Git wheels or conflicting Torch pins.
Complete Windows Python dependency resolution/import qualification is pending.

## DSR-AUDIT-013 — expert BF16 rounding and the complete shared/routed sum

The original Expert.forward rounds each projection to the caller's state dtype,
computes clamped/weighted SwiGLU in FP32, rounds that activation back to the
state dtype, then executes w2. EXL3's intrinsic FP16 kernel operand is a separate
boundary. Directly rounding its FP32 output to FP16 skips the BF16 boundary.
Likewise, rounding the routed sum before adding the shared expert changes the
original MoE accumulation order.

The owned executor now preserves the caller-dtype boundaries before w2 and
before each expert's FP32 accumulation. Its routed sum remains FP32 through the
real MoERunner's shared-expert addition, and delayed mHC rounds the complete
result to the BF16 residual dtype. Frozen original Expert/Block comparisons
cover the conversions and demonstrate a differing direct-FP16 shortcut. Twelve
focused expert/mHC/boundary tests passed. Packed GPU and full graph execution
remain unrun; no optional FP16 cache mode is introduced.

The latest DSR-AUDIT-011 capture confirms the bad sqrt bytes exist immediately,
with no later mutation. The first 1344-element parallel partition is affected;
three new calls on the captured input are accurate. The broader suite has one
failure, 304 passes and four actual-Win32 skips. This remains an open CPU issue.

## DSR-AUDIT-014 — ambient EXL3 GEMV can force a different accumulator

The pinned ExLlama small-M QTIP GEMV mode 2 uses FP16 MMA accumulation followed
by an FP32 fold. The generic EXL3 GEMM uses FP32 accumulators on the selected
SM89/SM120 architectures; its half-accumulator specialization is SM86-only.
Although the default Ada/Blackwell heuristic does not select that GEMV path,
an inherited EXL3_GEMV=2 could force it. Merely disabling the INT8 flags is
insufficient to make the selected execution path explicit.

The before-import policy now also sets EXL3_GEMV=0 and
EXL3_GEMM_LEGACY_TILES=0, and refuses later mutation. The actual CUDA diagnostic
uses native EXL3 K2/K3/K4/K5/K6/K8 matrices, exact independent trellis/fragment
reconstruction, 5120/2304 projections and independent normalized rotations.
Its dot/rotation envelope is computed a priori from FP16 input rounding and
FP32 accumulation bounds, not fitted after observing GPU output. The plugin's
independent scalar-codebook source now has a required runtime hash.
These diagnostics are implemented but have not executed on a GPU.

## DSR-AUDIT-015 — speculation must be qualified before admission

RequestSettings and omitted HTTP speculative controls now default to false.
Explicit speculative=true is rejected until the actual installed cluster probe
passes on this loaded component identity and authenticated mesh incarnation.
Reports and edited files cannot enable that in-memory gate. A replacement mesh
or process requires qualification again. Paused speculative snapshots can be
restored after the new process has qualified; they cannot bypass the gate.

The shipped probe uses actual three-stage text/image generation, matching
logical BF16 source/shadow samples, all-page B weight residency, normal/reused
target repeatability, greedy target-versus-DSpark parity, sampled repeat/RNG
consistency, and actual pause/save/restore/continuation. It also runs independent
tiny verifier/correction equations. Split-versus-itself comparisons are explicitly
metamorphic evidence; they are not independent full-model goldens. Local GPU
checks cover overwritten row rollback at every accepted length 0..6. Neither
the cluster probe nor these new CUDA checks has run on the user's hardware.
