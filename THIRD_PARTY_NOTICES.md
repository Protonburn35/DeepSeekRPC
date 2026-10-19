# Third-party source provenance

DeepSeekRPC is licensed AGPL-3.0-only. Inspection is not integration: the initial
source audit did not copy third-party inference code into the runtime. Subsequent
incorporation of official encoding and test-only oracle methods is listed below. Each
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
- `tests/oracles/exl3_dequant.py` is unchanged from vllm-exl3
  `src/vllm_exl3/dequant.py` at the locked commit, under AGPL-3.0-only. It is a
  test-only scalar procedural-codebook oracle, not a production executor.

Complete reference archive hashes and model metadata hashes are in provenance/.
No full model payloads, credentials, copied proprietary SDKs or compiled
third-party binary artifacts are included. Distribution of an integrated
application requires preserving all applicable notices and corresponding-source
requirements, including the plugin's network-service obligations.
