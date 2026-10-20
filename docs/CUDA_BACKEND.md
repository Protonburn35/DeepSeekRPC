# Native CUDA operation contract

The native extension is part of the runtime under qualification. Windows compile/link,
wheel install, CPU import and both SM89/SM120 cubins passed in run 39314752138.
The expanded checks also passed in run 39316435141. ABI 2 compile/link/import,
both architectures and imports without the compiler SDK passed in 39334269334;
its report/listing are retained as evidence/native-abi2-*. Actual GPU execution
remains unrun. The current project ABI-2 build passed again in 39410211028.
The complete vLLM owned profile passed 39428534610, and complete private-runtime
installation passed 39462821561 at source 30b24d3. Exact baseline reports are in
evidence/whole-runtime-30b24d3/; the final packet requires matching reports for
its frozen source revision.
The CUDA build workflow uses a CPU Windows runner and never executes kernels.
The intended deployment OS is Windows 11 on all three PCs.

The user's current requirement is BF16-only, unquantized KV/index storage. FP16
cache mode and quantized-cache fallbacks are excluded. This supersedes the
original implementation prompt's FP16-default language; FP16 inside existing
EXL3 weight kernels remains independent of this cache policy.

`native/operations.cu` implements these operations without INT8 activations or
implicit lower-precision cache storage:

| Operation | Storage and computation | State contract |
| --- | --- | --- |
| Sparse latent attention | Two BF16 KV pools, FP32 query/accumulation/output; online softmax includes the learned per-head sink as a zero-value term | Int32 per-query slots select the small SWA staging pool or persistent compressed source pool; -1 is padding, duplicates retain their multiplicity. Invalid slots/nonfinite scores fail. No full cache concatenation or dense 1M score matrix. |
| Index scores | BF16 stored keys, FP32 query/dot/rectification/head-weight reduction | A bounded tile of at most 8192 positions; explicit per-query visible counts; unreachable/padded positions are -inf. Candidate filtering and global top-k require the surrounding adapter. |
| Original Engram row decode | Exact FP8 E4M3 values multiplied by standard UE8M0 scales in FP32, then explicitly rounded to BF16 | Bounded encoded [rows,264] input, standard scale-byte-zero interpretation; NaN/overflow fails. Learned projection/gating is a separate graph operation. |
| Cache scatter | Exact BF16 copies into existing slot pools | Checks physical bounds, duplicate targets and expected int64 slot generations. Failed operations may have tentative writes; transaction rollback is mandatory before reuse/publication. |

Python wrappers inspect the native status after stream completion before
returning a result. This synchronous baseline establishes an ordering boundary;
future overlap must retain equivalent event and transaction dependencies. The
wrappers require the real compiled extension and a real CUDA device. They do not
substitute CPU model computation.

The attention equation and index ordering follow official October
`inference/kernel.py:sparse_attn_kernel` and
`inference/model.py:Indexer.forward`. Source hashes are in models.lock.json.
Independent tiny CPU equations in reference_attention.py compare against dense
Torch softmax/matrix operations; these fixtures do not validate CUDA execution.
ExLlama supplies the real packed EXL3 matrix kernels separately; implementing
these four operations does not complete EXL3 loading or the distributed graph.

The build pins Torch 2.11.0+cu130 (the selected Windows vLLM requirement) and the
official CUDA 13.2 compiler packages used by the selected ExLlama build recipe.
Their hashes and exact byte counts are recorded in the SDK lock. Compiler minor
version interoperability, both embedded cubins, extension loading and full
upstream native builds must be verified before qualification. Compilation does
not establish numerical GPU correctness or hardware residency.

The separate `windows-exllama.yml` job builds the full pinned EXLlama source,
without namespace replacements or a JIT-only wheel. Its helper verifies the
source archive hash, extracts a separate writable build tree, compiles both
architectures, imports the installed native extension and Python package, and
checks dependency consistency. This is separate from the four project kernels
and from the separately qualified native vLLM build. The upstream compiler math flags
are preserved; numerical packed-weight execution must be qualified separately.

The first full upstream EXLlama qualification passed in run 39318861176:
compile/link, wheel installation, native/Python imports, dependency consistency
and both SM89/SM120 cubins. The retained wheel SHA and build report are in
`evidence/exllama-first-build.json`. The first run's import used the SDK PATH;
future qualifications also remove SDK DLL directories for a separate installed
native import check. Later run 39345608258 passed the full build, package and
native imports with the SDK removed from PATH, dependency consistency and both
architectures. No EXL3 GPU product or actual model
has executed here.

`windows-vllm.yml` separately builds the full pinned upstream native wheel with
its native build dependencies. This job checks the actual installed Torch
library and cubins; it does not qualify the complete serving dependency stack.
External CMake source commits are recorded in vllm-native-deps.lock.json. The
helper checks the two upstream tag references and retains fetched commit IDs,
submodule state, and hashes of upstream-applied build patches. Four exact
non-build `.claude/skills` aliases are omitted during Windows extraction, avoiding
a symlink privilege requirement; runtime/build source files remain intact.

The vLLM wheel selects the explicit `VLLM_DEEPSEEKRPC_EXL3` build profile and
distribution version `0.31.0+deepseekrpc2.cu132` (the upstream compiler suffix
for the pinned CUDA 13.2 SDK). Exact before/after hashes and patch hashes are
recorded under `patches/`. This profile omits upstream SM12x FP8/NVFP4 matrix
and MoE kernels, which are unused by this model and fail MSVC C2719 in the
unmodified build. It preserves CUDA's descriptor alignment. Native capability
probes must reject those excluded formats; this wheel is for the selected
EXL3 application. Run 39428534610 passed the exact five-patch profile, wheel
build/install, actual core and MoE/router imports with the SDK removed from PATH,
and SM89/SM120 inspection for both libraries. Its wheel SHA-256 is
e6fe16765e0f908df5c0c7213e71a90b5e0f506d04e4e1ee474a9ec3af3d80ea.
Complete installation 39462821561 also passed the exact admitted version,
dependency/source/native identity, private compiler, Gloo and original parser
checks under private Python 3.12.14. This does not execute GPU kernels.

`deepseekrpc selftest-cuda` now runs independent small attention, index, Engram
and scatter comparisons on an actual SM89 or SM120 GPU. It is intentionally
excluded from CPU CI and remains unrun here. It loads no model and establishes
neither full-model correctness nor physical residency. The bounded expert
executor and vLLM quant-method adapter are integrated into the owned staged
graph. Independent packed reconstruction/rotation/projection GPU comparisons
are implemented but unrun.

`index_selection.TiledIndexer` scores at most 8192 keys at a time, gathers BF16
keys through the local physical slot map, and carries a bounded global top-512
selection. Source 20 also publishes at most 2048 candidate block IDs (eight
positions each), pinning the newest partial block. Later indexers score only
those blocks. Logical IDs are sorted for consumption and padded with -1. Equal
scores use lower logical IDs as a defined deterministic tie rule; equality with
the upstream CUDA top-k's unspecified tie choices is not claimed. The owned
graph provides actual query/weight projection, RoPE and exact CED visibility;
their GPU/model qualification remains unrun. Cancellation can stop between
tiles; cache readers hold a window lease
through CUDA completion so another thread cannot restore slots while in use.

`vllm_engram.DiskEngram` subclasses the actual vLLM common Engram graph and
replaces its table embedding with bounded encoded-row reads/uploads and the
project's explicit BF16 decoder. vLLM retains its learned wkv/q/k parameters and
Triton gate. Production construction requires the original publisher file's
complete hash/header admission, stage A and TP=1. Lookup staging reserves 2 MiB;
the base graph's bounded BF16 staged rows count separately. Hash stamps prevent
consuming rows prepared for another lookup. Actual graph construction, learned
CUDA execution and model integration remain unrun; the installed CPU audit only
checks the real upstream class relationship and imports.

`vllm_mhc` is the explicit bounded CUDA backend in the exact-source Windows
mHC patch. It preserves BF16 residual/layer-input storage and FP32 coefficients
from the pinned graph; the eventual boundary adapter must promote those values
to FP32 for activation/pre_mix transport. The separate exact Torch math policy
disables TF32 and reduced-precision auxiliary GEMM reductions before execution
and refuses later policy changes. The patch selects this path explicitly and
retains a real deferred TileLang import for unused operations; it supplies no
placeholder kernel/module. Complete patched model execution remains unqualified.

`dense_weights.FixedPackedWeights` holds owned non-routed EXL3 matrices in one
admitted GPU arena and uses the real plugin's packed kernels. Its physical
bindings cover all 668 matrices, including independent output-group rotations
and packed compressor/index-key projections. Startup reads remain bounded; B
requires its completed locked resident store. Dense input/output workspaces and
upstream scratch count separately in the full graph's memory ledger. Actual
packed execution and the complete layer construction remain unqualified.

`vllm_attention.OwnedBF16Attention` is a concrete subclass of the actual pinned
attention class. It uses real vLLM linears, fused Q/KV RMSNorm, RoPE, and the
existing Windows FP8 query preparation with its scale folded into index weights.
Index-key storage stays BF16. It stages only the causal SWA tail plus at most
64 current rows, maps sparse logical IDs through the local KV slot map and
projects the eight output groups independently. Source 20 has an explicit
encoder-only projection entry; phase-specific visibility remains separate from
window execution. This class imports in the installed CPU source audit, but
CUDA construction/execution and complete stage orchestration remain unrun.

`compression_state` calls the real vLLM compression kernel with FP32 pair
state, fixed before-images and at most 64 original score rows. An accepted
prefix reconstructs the original ring stores without reducing state precision.
The normalized latent output is BF16. `attention_state` validates owned versus
replicated pools and window-scoped top-k/candidate publications. The complete
phase coordinator must validate immutable history and the all-stage encoder
barrier before rewinding SWA/pair state for exact replay. These components do
not by themselves establish that global barrier or distributed commit.

`native_weights` copies learned native parameters through a bounded startup
tile, including declared source/destination dtype conversions in its manifest.
B's source is exclusively the locked resident store. Native parameter memory,
boot scratch and packed storage have separate explicit admission. The full
vision/DSpark binding plan and actual graph bootstrap are integrated source;
their hardware execution remains unrun.

`stage_graph.OwnedStageGraph` now constructs only each stage's actual patched
decoder layers, the native A vision tower/aligner and both disk Engrams, and C's
three actual draft blocks and shared DSpark heads. Every native tensor must map
exactly once to an owned physical source. The packed constructor binding plan
also covers absolute draft layers 40–42 and the shared context combiner.
The graph flushes delayed mHC at boundaries, promotes activations to FP32 and
checks exact reconstruction of internal BF16 values on receipt. The CPU boundary
test verifies BF16 bit preservation and unchanged FP32 mix coefficients.

Draft attention has a distinct bidirectional five-row backend. Each draft
layer's persistent context is seeded from that layer's own projection of target
attention-input features at layers 37/38/39. Proposed rows never enter that ring.
Graph methods expose full draft logits and sequential Markov bias/confidence;
the global coordinator verifies and commits accepted prefixes across all three
actual stage services. C's explicit
text-embedding replica and its additional memory are recorded in MEMORY_PLAN.md.
Runtime construction checks pinned package versions and exact patched source
hashes. All GPU graph constructors/execution remain unrun. Bootstrap, global CED
scheduling/commit, original image processing/API, snapshots and prefix reuse
have integrated source. Complete Windows runtime installation and full-model
qualification remain open gates.

Native ABI 2 retains the four CUDA operations and adds an actual Windows DXGI
budget query using CUDA's adapter LUID. The extension links the system DXGI
library; the query runs only on an actual Windows GPU host. CPU build/import
checks do not execute it. Subsequent native qualifications also import with the
compiler SDK removed from PATH, checking the installed runtime dependency path.
The original successful ABI 1 build remains recorded as historical evidence;
it does not qualify the added telemetry or ABI 2 wheel.
