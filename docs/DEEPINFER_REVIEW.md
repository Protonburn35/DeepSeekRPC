# Official DeepInfer source review

Reviewed `deepseek-ai/DeepInfer` at
`8b16149741a4935e1dec8cea27853f273b374c5b` (2026-10-20). This is the eighth
reference, additional to the original seven. `provenance/sources.lock.json`
pins its 2,090,556-byte archive with SHA-256
`32794e849ce469967c1f215736a6a4307ec1c1db485b5dbb4f6a845ff43d0197`.
`provenance/deepinfer-review.lock.json` records the inspected source surfaces.
The Apache-2.0 source was inspected without importing/building DeepInfer,
running its tests/examples, or downloading any model. No DeepInfer runtime
code is copied into this application.

The README identifies V4.1-Flash support, and `examples/_config.py` defaults
to `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Its Linux/GCC 13+/CUDA 13 NVIDIA
path targets SM100; Ascend 950 is also supported. The published offline release
explicitly leaves a service KV cache system, disaggregated deployment and load
balancing to the integrator. It does not qualify Windows, SM89/SM120, our EXL3
pack, bounded SSD/host/GPU expert caching, or the three-PC transport.

All source paths below are relative to that immutable revision. They describe
inspected code, not executed upstream behavior.

| Surface | Source evidence | Effect on DeepSeekRPC |
| --- | --- | --- |
| HF model mapping | `deepinfer/hugging_face/config.py`: `_GPT_DEFAULTS`, `_shared_layer_cfg`, `_n_prefill_layer`, `_build_dflash_cfg` | Confirms nearest preceding KV/index/index-K sources, distinct index-K sharing, ratio-1 consumer boundary, original DSpark dimensions. Crucially, the HF mapping sets `yoco_swa_share_hidden=False` and `use_qnorm=False`; generic attention defaults are insufficient evidence to change these settings. |
| Shifted mHC | `models/v4_modules/hc.py`, `models/deepseek.py`, `models/v4_modules/heads.py` under `deepinfer/model_executor/` | Confirms the delayed previous pre-mix and final collapse using the last sublayer's pre-mix; shifted checkpoints do not build an HCHead. Keep FP32 stage pre-mix and exact BF16 residual boundaries. |
| Draft target features | `models/decoder_layer.py:_store_dflash_input`, `layers/tilelang_ops/cuda/dflash.py:dflash_input_store_tl` | Captures the materialized entering-attention residual, FP32-averages its HC streams and writes BF16 feature rows. Confirms our 37/38/39 attention-input means rather than final layer outputs. |
| Draft execution | `models/mtp/dflash.py` and `models/decoder_layer.py:DFlashDecoderLayer` | Confirms first seed plus noise input rows, target-feature projection/norm, per-draft-layer main-context KV, sequential Markov correction from the previous token, and confidence from the pre-output-norm hidden plus Markov embedding. Keep proposal KV ephemeral and commit only verified target-derived context. |
| Draft visibility | `input_metadata/dflash.py`, `layers/tilelang_ops/get_kv_index.py`, `layers/tilelang_ops/cuda/get_kv_index.py` under `deepinfer/model_executor/` | With last accepted target position `a`, draft positions are `a+1..a+5`; each draft row sees target positions `max(0,a-127)..a` and all five draft rows. Confirms 128 prior target keys plus five bidirectional draft keys, distinct from a target SWA row's 127 past keys plus itself. |
| Compressor and index | `models/v4_modules/compress.py`, `attn.py`, `index_attn.py` | C1 uses BF16 projected KV, C2 retains FP32 score/latent state. Index K consumes the post-norm, pre-RoPE compressed latent. Preserves the implemented original C1/C2, source/shadow and independent index-selection semantics. DeepInfer's FP8/FP4 cache kernels are not our BF16 cache implementation. |
| Routed FFN | `models/v4_modules/moe.py` and `mlp.py` | Confirms sqrt-softplus, separate image bias and routing weight applied to SwiGLU before the down projection. Its datacenter EP/EPLB/FP8/FP4 implementation does not replace our packed EXL3 expert executor. |
| Exact rejection equation | `layers/tilelang_ops/cuda/fused_sample.py` | Uses target/draft logprob acceptance and positive residual mass. Our verifier retains exact `min(1,p/q)` and normalized `max(p-q,0)`. Its complete arithmetic is not a drop-in exact sampler; see below. |
| Confidence scheduling | `csrc/engine/mtp_varlen.hpp:MTPVarlen::schedule` | Explicitly requires future confidence values not to influence whether an earlier draft is verified. The scheduler separates historical total-budget selection from request prefix allocation. Our fixed five-row proposal path keeps confidence diagnostic only and does not prune from it. |
| Replay truncation | README, `models/v4_modules/yoco.py`, `input_metadata/yoco.py`, `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp` | `compact` gathers retained suffix rows; `mask` changes SWA visibility to the retained rectangle. Both are bounded replay. Neither is enabled or substituted for our full target replay. |
| Regression methodology | `tests/models/test_sampler.py`, `test_kl.py`, `test_batch_invariant.py`, `test_ve_batch_invariant.py`, `test_offline_inference*.py`, `tests/worker/test_state_cache_ops.py` | Useful designs: distribution tests, decode-versus-full-prefill logprobs, fixed golden numerical baselines, batch/layout invariance, and state save/restore boundaries. Upstream tests and published PPL goldens require their model/platform/parallel layout and have not run here. |

## Numerical choices retained

DeepInfer's `layers/tilelang_ops/common.py` sets
`DRAFT_PROB_PENALTY = 5e-5`. Its rejection residual uses
`max(p_target - exp(-5e-5) * p_draft, 0)`; the same file documents bounds on
the resulting distribution bias. The GPU sampler also encodes probabilities
as fixed-point integers. These choices are not adopted. In particular, a
coordinate where target and proposal probabilities are equal must receive
zero exact correction mass. We retain full FP64 CPU sampling metadata,
separate transactional RNG domains and unmodified exact rejection correction.
This metadata arithmetic does not move model execution to the CPU.

The README's deterministic sampling recipe explicitly disables variable-length
verification and enables batch invariance. Identical seeds alone do not imply
identical normal-versus-speculative sampled token sequences, because those
algorithms consume random draws differently. Our qualification compares greedy
tokens across modes, repeatability and restore/RNG identity within the same
sampled mode, and independent probability laws. It does not require the wrong
cross-mode seeded-token identity.

DeepInfer's DFlash path explicitly casts target-context and draft SWA keys to
FP8 (`CastArgs("e4m3", ...)` in `DFlashAttention`). This is not adopted: the
user's unquantized BF16 requirement covers persistent draft context and
ephemeral proposal keys too. No inference is made from its quantized-cache
numerical baselines to our cache precision or Windows kernel qualification.

The README proposes 140 retained decoder tokens. Its source enforces retention
at least SWA plus backtracking, but that condition does not prove full-prefix
equivalence through all decoder layers. Retained suffix boundary changes can
propagate through causal layers. DeepSeekRPC therefore still replays the entire
target prompt with exact causal SWA and full encoder-source visibility. Encoder
prefix reuse does not shorten this target traversal.

## Changes and verification prompted by this review

The pinned provenance and this comparison are added to the review packet.
New independent sampling fixtures enumerate the exact two-token target joint
law with conditional proposals and rejection, require zero correction mass on
equal-probability coordinates, and vary every diagnostic confidence value while
requiring identical actual proposal/verification/RNG receipts. These complement
the existing all-rejection-length, EOS, output-limit and rollback checks.
No upstream test imports or model execution are used for these fixtures.

The independent probability fixtures are stronger than repeatability alone;
they still do not establish full-model correctness. Actual SM89/SM120 GPU math,
full-model decode-versus-prefill logit comparison, full-context replay and
Windows three-PC execution remain hardware qualification work after review.
No source-only finding here waives those gates or proves a throughput result.
