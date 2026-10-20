# Third-party source provenance

DeepSeekRPC is licensed AGPL-3.0-only. Inspection is not integration: the initial
source audit did not copy third-party inference code into the runtime. Subsequent
incorporation of official encoding, Engram hashing and test-only oracle methods is listed below. Each
future incorporated file or patch must record its origin, commit, license and
local changes here and in the dependency manifest.

| Reference | Observed license | Current use |
| --- | --- | --- |
| SystemPanic/vllm-windows | Apache-2.0 | Pinned graph/build source audit; intended runtime foundation |
| vcruz305/vllm-exl3 | AGPL-3.0-only, additional Apache notices | Compatibility checker run externally; intended EXL3 integration |
| vcruz305/exllamav3 | MIT | Quant/kernel and Windows build audit |
| sybil-solutions/dsv41-flash-offload | MIT | Cache/numerical failure reference |
| Zanooda/deepseek-v4.1-flash-cmp170hx | Apache-2.0 | Replica and Engram patch reference |
| Protonburn35/StrataRPC | MIT | Lifecycle/protocol reference; no deployment changes |
| MiaAI-Lab/DeepSeek-V4.1-Flash-EXL3-2x-DGX-Sparks | AGPL-3.0 | Source/stream/autotune reference |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | Model repository LICENSE (MIT text observed) | Pinned config, encoding and independent numerical reference |
| LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw | Model repository LICENSE (MIT text observed) | Metadata and bounded shard headers only |

Incorporated source:

- `src/deepseekrpc/_vendor/deepseek_1017/encoding.py` is unchanged from the pinned
  official model's `encoding/encoding.py`. Its MIT LICENSE and SHA-256 are
  packaged alongside the file. The upstream encoding tests only adapt imports
  and fixture paths; five frozen input/output pairs and two original examples
  retain their pinned provenance.
- `tests/oracles/official_methods.py` extracts four unchanged mathematical
  methods from official `inference/model.py`. Its provenance records source
  lines, segment hashes and CPU-only binding. The original MIT license is in
  tests/oracles/LICENSE.
- `src/deepseekrpc/_vendor/deepseek_1017/engram.py` is unchanged from official
  `inference/engram.py` at the same pinned October revision. It supplies the
  compressed token map and n-gram hash state, under the adjacent MIT license.
  `provenance/engram-reference.lock.json` records its hash and the unmodified
  tokenizer metadata compressed in tests/fixtures/tokenizer.json.gz. The frozen
  hash fixture was generated directly with that upstream file, independently
  of the transactional adapter. It contains no learned weights.
- `tests/oracles/exl3_dequant.py` is unchanged from vllm-exl3
  `src/vllm_exl3/dequant.py` at the locked commit, under AGPL-3.0-only. It is a
  test-only scalar procedural-codebook oracle, not a production executor.
- `src/deepseekrpc/vllm_mhc.py` adapts the delayed mHC mathematical operations
  from the pinned Windows vLLM `model_executor/kernels/mhc/torch.py`, under
  Apache-2.0. It adds bounded CUDA entry checks, explicit RMSNorm, exact-math
  policy and the stage interface. The license and original file hash are
  packaged in `_vendor/vllm_windows/`. The exact-source Windows mHC integration
  patch and before/after hashes are in patches/; graph execution is unqualified.
- `src/deepseekrpc/vllm_attention.py` adapts learned projection construction
  from the pinned Windows vLLM V4.1 attention class under Apache-2.0. It retains
  the real upstream linear, norm, RoPE and query preparation operations while
  explicitly binding owned BF16 pools, source visibility and packed matrices.
  The adjacent packaged provenance records the original file hash. Exact
  decoder factory and FP32 router source patches are recorded under patches/.
- `reference_packed.py` expresses the pinned ExLlama trellis and 16x16 MMA
  fragment format as independent NumPy diagnostic equations. The format source
  is the MIT-licensed ExLlama revision in provenance/sources.lock.json. CUDA
  tests use the separately pinned AGPL plugin's scalar codebook implementation;
  it is not an inference fallback. Runtime source identities include that file.

Complete reference archive hashes and model metadata hashes are in provenance/.
No full model payloads, credentials, copied proprietary SDKs or compiled
third-party binary artifacts are included. Distribution of an integrated
application requires preserving all applicable notices and corresponding-source
requirements, including the plugin's network-service obligations.
