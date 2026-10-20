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

## DSR-AUDIT-016 — Windows 3.12 measurement clock resolution

Windows CPU run 39407156592 passed 360 tests but failed the real HTTP benchmark
fixture: GetTickCount64-backed monotonic_ns returned zero elapsed time and the
throughput division failed. All application nanosecond duration measurements
now use the monotonic high-resolution performance counter consistently across
start/end boundaries. A zero-duration observation reports null throughput; no
artificial elapsed time is invented. The actual HTTP fixture covers both a
normal clock and a frozen-clock boundary. Thirteen focused checks passed;
Windows CI must still qualify the repair.

## DSR-AUDIT-017 — ambient Windows JIT compiler and interpreter selection

The pinned triton-windows 3.6.0.post26 wheel bundles TinyCC and CUDA ptxas,
headers/import libraries. The pinned Python standalone archive includes Python.h,
pyconfig.h and python312.lib. Source inspection also finds ambient CC/MSVC/CUDA
selectors, interpreter mode, architecture and kernel/cache overrides. Bare PATH
alone does not exclude these selectors.

Before importing Triton, the worker now verifies pinned compiler inputs, selects
its private bundled tools/cache and IEEE FP32 dot default, disables interpreter
and kernel overrides and refuses later mutations. Installed component identity
includes Triton source, binaries and compiler libraries. A real Windows compiler
subset workflow builds/loads a TinyCC CPython extension and compiles explicit
SM89/SM120 Triton inputs, inspecting PTX targets and CUDA ELF architecture flags
without initializing a GPU context. The full installer requires that check too.
These compiler execution checks are implemented but not yet Windows-run at this
checkpoint; policy configuration is not an execution claim.

## DSR-AUDIT-018 — runtime workflow YAML and native build cancellation

The manual runtime workflow's plain run scalar contained an unquoted colon at
`--only-binary=:all:`, so GitHub rejected its YAML before any job could start.
It now uses a folded scalar. CPU checks parse all shipped workflows with unique
mapping keys before running tests; this checks YAML, not every Actions semantic.

Adding DeepInfer to sources.lock.json also matched the long vLLM workflow's
source-pin trigger and cancelled run 39400420994 after about 69 minutes. No
compiler failure was reported in that run. Replacement 39408672207 is building
the same five-patch profile. The vLLM workflow now preserves running builds on
later pushes, as the EXLlama workflow already did. Exact source/patch checks
still determine whether a retained native component can enter a runtime bundle.

The first actual DSR-AUDIT-017 Windows compiler run built/loaded TinyCC's Python
extension and compiled the SM89 input. The SM120 input also compiled, but the
inspector incorrectly read the old low-byte architecture field. Diagnostic run
39410453853 captured CUDA ELF ABI 8, header
7f454c46020101410800000000000000 and flags 0x0600780a. The pinned LLVM ELF
reference documents ABI 8's SM mask 0xff00 and offset 8. The inspector now reads
the correct ABI-specific field, still requiring CUDA ELF64/little endian,
explicit requested PTX and matching embedded SM. A regression uses the actual
captured header/flags; unsupported ABI/layouts remain errors. This is an
inspector repair, not a change to generated code or a GPU execution result.

The corrected standalone compiler check passed Windows run 39411095629 at
27e7de9: actual TinyCC CPython build/load, SM89 ABI 7 and SM120 ABI 8 cubins,
selected bundled tools and mutation guards all passed without a GPU context.
The compiler selection/ELF inspection portion of DSR-AUDIT-017 is qualified on
that exact toolchain. Complete application installation and real GPU/model math
remain separate pending gates. DSR-AUDIT-018's YAML now parses in CPU CI; running
native builds are preserved and duplicate pending native runs were cancelled.

## DSR-AUDIT-019 — claim self-test state before JIT cache creation

The new before-import compiler policy can create a role's kernels/triton cache.
Standalone `serve --self-test` previously called that policy before own_state
had created its ownership marker. A later normal serve would then correctly
refuse the nonempty unowned directory. The hardware path has not been run;
this is a concrete first-run ordering defect found in source review.

Device preflight now claims/locks the same real state directory before any JIT
cache write and holds ownership throughout diagnostics. It releases ownership
on success or failure, allowing later normal startup to reuse the marked cache.
An already loaded owner excludes a concurrent self-test. Regression fixtures
exercise real file locks/directories and interrupted lifetime, with an explicit
I/O observer instead of a fabricated GPU result. Fifteen focused state, config
and precision-policy checks pass. Actual GPU diagnostics remain unrun.

The latest DSR-AUDIT-011 instruction probe reproduces every one of the 1344
captured bad values with AVX2 input * approximate rsqrt, bit for bit. An isolated
MKL enhanced-performance call with AVX2 forced matches those same values;
explicit high-accuracy calls do not. Normal AVX512 high accuracy matches the
accurate second partition. The installed ATen wrapper header explicitly asks
for high accuracy. This identifies the reduced-accuracy path signature, but
why it was selected intermittently remains unresolved. Actual captured arrays,
source probe and scoped numeric reports are preserved under
evidence/cpu-sqrt-diagnostic/. No oracle, tolerance or production setting changed.

## DSR-AUDIT-020 — local Windows rendezvous must not depend on narrow paths

Actual Windows CPU run 39415738104 failed when Torch FileStore opened an
existing directory containing a non-ASCII character (DistStoreError: no such
file or directory). The other 379 tests passed. This exposed a startup risk
when the application's state directory has such a path; it was not a GPU error.

Each owned stage is exactly one local rank. An initial HashStore change passed
the cloud CPU suite but failed Windows run 39416751099 because Windows Torch
intentionally omits that binding. The pinned Torch 2.11 source confirms this
and provides a supported Python Store/C++ trampoline. Auxiliary source hashes
are in provenance/torch-store-reference.lock.json; no upstream code was copied.

Startup and runtime qualification now share a bounded, thread-safe in-process
Torch Store/Gloo initializer, with no rendezvous filename, TCP rendezvous
listener or libuv requirement. The actual vLLM setup accepts that already
initialized group and creates its own local world/model groups. The three-PC
TLS transport is unchanged. The store implements real C++ callbacks, waits,
timeouts, atomic counters and compare/set, with key/value/total inventory bounds.

A real additional-group check also exposed a lifetime defect: C++ PrefixStore
retains the C++ trampoline but not the Python methods/state. Dropping the Python
store owner caused a subsequent real Gloo new_group to fail after collection.
The loader now retains that owner in its resource stack until actual vLLM model
and world groups are destroyed, including partial setup failure. Qualification
forces collection before creating a real additional group and running its
metadata collective. It repeats setup/teardown with non-ASCII output paths;
separate C++ PrefixStore checks cover the actual callbacks and bounds. Native
Windows repeat qualification and complete runtime installation remain required.

## DSR-AUDIT-021 — wait for an unleased expert slot's actual CUDA completion

Source review found that the device cache could reject a new bundle when all
unleased slots still had pending use events. One-slot admission is legal, but
the routed caller has no retry for that rejection. CPU/GPU timing may mask this
case; no actual model failure or GPU reproducer is claimed here.

The cache now chooses an unleased victim and polls its actual completion event
before closing matrices or overwriting storage. The wait has a 30-second limit
and observes the independent request cancellation flag. Live leases remain
non-evictable; a failed/cancelled wait retains the old entry and storage. The
new expert_slot_wait timing records this backpressure explicitly.

Real host-event control tests cover completion, cancellation and timeout. They
do not emulate a CUDA result. The GPU self-test now contains a real one-slot
packed-cache probe with finite queued GPU delay, two actual EXL3 calls, exact
reused-output comparison and stale-lease refusal. That GPU probe remains unrun
until the later hardware phase.

## DSR-AUDIT-022 — pin the shared small-row EXL3 tuning policy

The pinned EXL3 GEMM source shares its tuning bucket for row counts 1..8 to
keep normal decode and speculative verification on the same shape/K split.
Its unset default enables this behavior, but an inherited
EXL3_GEMM_SMALLM_SHARED_TUNE=0 disables it before a cached first use. Merely
disabling the INT8/GEMV shortcuts does not constrain this separate selector.

The before-import exact policy now explicitly sets the shared-tuning selector
to 1 and refuses later mutation. The existing real-process policy regression
starts with it disabled and exercises mutation refusal for every pinned EXL3
selector. No native source or build profile changed. This fixes selection of
the intended upstream policy; actual CUDA row invariance and full-model
speculative qualification remain separate unrun gates.

## DSR-AUDIT-023 — establish offline/privacy policy before upstream imports

The model configuration set offline/telemetry controls, but owned adapter
imports occur earlier during startup. The pinned vLLM usage module also caches
its enablement on first use. No unexpected download or telemetry transmission
was observed; this is a source ordering gap.

The common before-import policy now explicitly disables vLLM/Hugging Face
telemetry and requires local-only Hugging Face/Transformers access. It checks
later environment mutation and refuses an already cached enabled vLLM reporter.
The real-process policy fixture begins with every control disabled. Installed
CPU integration and complete Windows qualification query the actual upstream
reporter's enablement without sending a report. Native sources are unchanged.

## DSR-AUDIT-024 — selected EXL3 build must exclude DeepGEMM's host helper

Long Windows run 39408672207 failed at step 187/189 in DeepGEMM's host binding:
its nested deep_jit uuid.hpp includes Linux unistd.h. The actual core and MoE
DLLs had linked at steps 185/186. There is no completed wheel from that run.

The fifth exact-source profile patch now omits DeepGEMM in both the CMake
external-project list and setuptools extension list, alongside the existing
FlashKDA/Qutlass exclusions. The selected graph executes its own BF16 CUDA
attention/index/state, packed EXL3 experts and explicit Torch/Triton mHC; those
paths are preserved. DeepGEMM selection is explicitly disabled before import
and a cached enabled selector is rejected. No native placeholder is supplied.
The updated patch was applied, hash-checked and reapplied idempotently against
its exact four-patch predecessor. Existing native ABI 2/EXLlama wheels remain
usable because their actual source and build profiles are unchanged.

The next build also loads both actual core and MoE/router libraries, checks
their exported schemas and SM89/SM120 images, then repeats import without the
SDK. Complete installation repeats both native registrations. The build emits
bounded progress counts every 45 seconds without publishing raw paths. Actual
fresh Windows compile/install qualification is still required.

## DSR-AUDIT-025 — admit the compiler-qualified vLLM wheel version

The retained metadata log from run 39408672207 identifies the distribution as
0.31.0+deepseekrpc2.cu132. Upstream appends the compiler's CUDA version when it
differs from its default 13.0. Our actual SDK is 13.2, but runtime admission
expected only 0.31.0+deepseekrpc2, so complete installation would reject the
compiled wheel. That earlier run failed before producing a wheel; its metadata
is not native qualification evidence.

Runtime admission now expects the actual .cu132 distribution version. The
native source, build command and Torch 2.11.0+cu130 dependency are unchanged.
No wheel is renamed or repacked. Full installation must still verify the
completed wheel's metadata, source hashes and actual native registrations.

The separate CPU Python audit now verifies the installed patched vLLM file
hashes and real decoder factory signature, in addition to importing the actual
adapter base classes. An older CPU audit wheel did not contain the current
stage hooks; its earlier import checks did not cover those hooks. Refreshing
that supported empty build provides Python integration evidence only, without
claiming Windows compilation, GPU execution or full-model qualification.
The fresh Torch 2.11 CPU audit passed all six installed integration checks,
all 20 original parser checks and pip check. Complete Windows qualification now
also imports the actual decoder and checks both factories before completion.

## DSR-AUDIT-026 — keep vLLM metadata/config state inside the worker directory

An actual CPU metadata-only configuration probe tried to save model inspection
information below the default user-profile .cache/vllm directory and hit the
cloud's read-only profile. Configuration then reached the expected CUDA-only
Engram validation, which the empty CPU audit build cannot satisfy. This is not
a successful complete model configuration or a Windows/GPU failure report.

Stage startup now selects VLLM_CACHE_ROOT and VLLM_CONFIG_ROOT under its own
state_root/kernels/vllm directory before any vLLM import. Runtime verification
without an explicit worker directory uses the private interpreter directory.
Inherited profile/shared cache selectors are replaced; a later
directory or cached-selector change is refused. No HOME value is changed.
Installed qualification and CPU integration use their own evidence directory
for this state. Installer sealing and review evidence require the private-cache
check. This metadata cache never provides a model-weight fallback on stage B.

Real filesystem/process checks cover non-ASCII paths, inherited selectors,
directory changes, mutation refusal and preserving an existing file. Native
source and the running CUDA build are unchanged.
The refreshed actual installed audit passed all seven checks. A fresh real
model-inspection probe wrote its 873-byte JSON into the owned cache directory,
then hit the same expected CPU Engram refusal. Actual worker/early API startup
preserves the before-import ordering, and a separate actual vLLM import followed
by late cache selection is refused. Reports and exact application identity are
retained in evidence/owned-cache-*-audit.json and
evidence/owned-cache-audit-provenance.json. Full native Windows installation and
complete CUDA configuration remain separate gates.
