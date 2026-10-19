# Compatibility audit

Status: M0/M1 engineering checkpoint, not a release candidate. The user authorized
implementation on 2026-10-19 and selected the existing Protonburn35/DeepSeekRPC
checkout. The historical draft-only language in IMPLEMENTATION_SPEC.md records
the earlier authorization state; it does not cancel that subsequent instruction.

## Foundation and provenance

Retain the specified SystemPanic Windows vLLM graph as the foundation, with a
DeepSeekRPC stage adapter and dedicated TCP transport. There is no authorized
model, platform, or fidelity substitution. Reference archives are outside the
implementation checkout under `/workspace/deepseekrpc-references/source`.
`provenance/sources.lock.json` records exact commits, archive SHA-256, URLs and
observed licenses for all seven required references. Source acquisition used
codeload archives without submodule or LFS materialization. No upstream files
were edited. The original StrataRPC installation is untouched.

The selected EXL3 pack resolves to
`0be63c3337efbb2a0061d5b8bb4006fa186d433f`; original model metadata is pinned to
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Metadata hashes and publisher LFS
hashes are in `provenance/models.lock.json`. Publisher file hashes are not claims
that full weights were downloaded or verified locally. No weights or Engram
payloads are permitted in Codex. Config, indexes, quantization metadata and
bounded safetensors headers are sufficient for inventory.

## Source evidence

Paths in this table are relative to the pinned reference indicated.

| Surface | Evidence | Consequence |
| --- | --- | --- |
| Plugin ABI source surface | vllm-exl3 `tools/check_vllm_compat.py`, executed against Windows vLLM `vllm/`; exact output in `evidence/vllm-compat.log` | Exit 0: imports, plugin group and required abstract methods resolve. One guarded fallback NOTICE and four documentation-only BAD lines are preserved. This is not installed import or GPU qualification. |
| Actual geometry | Model `config.json:text_config` and vllm-exl3 `src/vllm_exl3/deepseek_v41.py:DEEPSEEK_V41_HIDDEN`, `DEEPSEEK_V41_INTERMEDIATE`, `native_p2b_geometry_supported` | 40 layers, 5120 hidden, 2304 expert intermediate, 384 routed experts, top-6, one shared expert. |
| Cross-rank state | Windows vLLM `vllm/models/deepseek_v41/attention.py:DeepseekV4Attention.__init__` | Compressed consumers select the latest preceding KV/index sources. Missing local index K source raises NotImplementedError. Default boundaries require source-14 and source-20 replicas, top-k and candidate state. Removing guards alone is incorrect. |
| FP16 requirement | `attention.py:_resolve_dsv4_kv_cache_dtype`; `nvidia/flashinfer_sparse.py:DeepseekV4FlashInferMLASparseBackend`, `DeepseekV4FlashInferSM120Attention` | Packed paths accept FP8/FP4 formats; plain fallback returns BF16. Backend lists BF16 compute, not a qualified FP16 path. New FP16 storage, index selection and sparse attention adapters are required. A dtype flag cannot satisfy R07. |
| Replay default | Windows vLLM `vllm/config/cache.py:CacheConfig.swa_bounded_replay` | Upstream defaults true. DeepSeekRPC exact mode must set false and verify the resulting backend setting. |
| Engram allocation | Windows vLLM `vllm/config/engram.py:EngramConfig`; `nvidia/engram.py:DeepseekV41EngramEmbedding` allocation paths | Pinned whole-table CPU/UVA storage is incompatible with the 32 GiB hosts. Implement bounded exact row lookup on A. |
| Disk-row reference | Zanooda `patches/README.md`, v41 patches 0009, 0010 and 0013 | Source/shadow transfer and exact disk gather are relevant references. Linux preadv/pthread/runtime GCC details require Windows equivalents. Packed reference row sizes are not FP16 sizes. |
| Expert offload | vllm-exl3 `docs/CPU_OFFLOAD.md`, `src/vllm_exl3/uva_offload.py` | UVA placement is distinct from CPU computation and does not implement bounded SSD/RAM/VRAM tiers. |
| Quantization diversity | Selected `exl3_k_map.json` and `quantization_config.json:tensor_storage` | Routed K2 on 18–22, K3 otherwise; shared K4/K5 including mixed layer 29, attention K5, indexer K8, head K6, draft K4. Test all actual layouts, not K2/K3 alone. |
| Quant metadata completeness | Metadata flattening: 187106 stored tensor descriptors vs 192495 index entries | 5389 tensors lack tensor_storage entries; metadata-only byte totals undercount. Use shard headers for complete inventory; do not treat absent descriptors as zero bytes. |
| DSpark ownership | Windows vLLM `nvidia/dspark.py:DeepseekV41DSparkModel`, `combine_hidden_states`; model `dspark_target_layer_ids` | Targets 37/38/39 belong to C. Full graph/loader and transactional acceptance audit remains required; leave speculation off until verified. |
| Routing and expert math oracle | Original `inference/model.py:Gate.forward`, `Expert.forward` | sqrt-softplus scores; bias affects selection only; separate image bias; unbiased weights normalize then scale 1.5; gate clamps above, up branch both sides before SwiGLU. |
| Vision and mHC oracle | Original `inference/model.py:Transformer.forward`, `merge_image_embeddings`, `Block.hc_pre`, `Block.hc_post` | Image/token positions and four-stream pre_mix are semantic state. Target draft features are attention inputs, not final outputs. |
| Lifecycle reference | StrataRPC `include/strata/core/layer_stage_rpc.hpp:LayerStageExecutor`, `LayerStageCancelMode`; `docs/LAYER_STAGE_RPC.md` | Persistent ownership and finish-window cancellation are reference patterns, not a transferable Qwen model implementation. |

## Toolchain and platform

Current host: Linux x64, Python 3.12.14, GCC/G++ 14.2, four-core CPU quota,
32 GiB memory limit, approximately 29.8 GiB workspace free at initial audit.
No CUDA toolkit/GPU, MSVC, PowerShell or native Windows runner is present.
Initial bootstrap packages are pinned separately from the application stack.

Windows vLLM `requirements/cuda.txt` selects Torch 2.11.0+cu130 and torchvision
0.26.0+cu130 on Windows, but Torch 2.13.0 on non-Windows. Its pyproject build
requirements also name Torch 2.13.0. The README specifies a preinstalled Windows
Torch and `--no-build-isolation`. Resolve that exact Windows build rather than
allowing isolated build requirements to select a different ABI. ExLlamaV3
`setup.py` has explicit Windows compiler handling; its `pyproject.toml` includes
triton-windows. CUDA 13.0–13.2 alignment concerns are documented upstream and
remain unqualified; do not patch an installed global SDK blindly.

Required CUDA architecture targets are SM89 and SM120. Neither is established
by SM121 Spark results. Both must be explicitly compiled on a CPU-only Windows
builder and subsequently executed on the user's hardware. No Linux pass is a
Windows build pass. No package version combination is called qualified yet.

## Licensing

DeepSeekRPC's new implementation uses AGPL-3.0-only to accommodate the intended
AGPL-3.0-only plugin integration. Preserve the vLLM Apache-2.0 notices and MIT
notices for ExLlamaV3/StrataRPC/other reused source, plus per-file provenance and
patch records. `THIRD_PARTY_NOTICES.md` distinguishes inspected sources from
actually incorporated code. The model repositories provide their own LICENSE.
Do not advertise a combined runtime as MIT-only or distribute wheels without
their licenses and corresponding source obligations.

## Qualification gaps

Installed vLLM/plugin registration, real CUDA kernels, Windows extension builds,
FP16 attention/index execution, physical laptop residency, exact CED scheduling,
vision, distributed full-model behavior and throughput remain unverified.
These are implementation/build gates, not waived requirements. Windows CI use
and remote publication are separate from local implementation authorization;
prepare a concrete workflow before requesting any missing runner/publication
permission. Astra review is user-mediated and has not happened.
