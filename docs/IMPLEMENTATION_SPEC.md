# DRAFT — FOR USER AND CHATGPT REVIEW, NOT EXECUTION

# GPT-6.1 Sol: DeepSeek-V4.1-Flash-1017 EXL3 2.9 on three native Windows PCs

Prepared: 2026-10-19. Intended implementer: GPT-6.1 Sol. Independent reviewer: Astra, through explicit review handoffs managed by the user.

**Authorization boundary:** This file is an implementation-prompt draft. Its creation authorizes no implementation, agent launch, installation, repository change, cloud setup, CI run, publication, or push. Execute the engineering instructions below only after the user separately approves the final prompt and explicitly instructs you to begin. Historical permission concerning StrataRPC does not authorize a new GitHub destination. Local development and remote publication are distinct actions.

## 1. Mission and acceptance contract

After authorization, build a separate, production-oriented native Windows x64 inference application for **DeepSeek-V4.1-Flash-1017, LibertAIDAI EXL3 2.9 bpw**, using all three of the user's existing PCs. Preserve the working Qwen StrataRPC installation as a fallback. Prioritize correct inference, useful memory capacity, predictable operation, and OpenCode agent compatibility. Performance goals are **500 prompt-prefill tokens/s and 40 generated tokens/s**; these are stretch goals, not measured expectations or permission to reduce fidelity.

The **first user-facing release candidate must contain the complete real three-PC CUDA execution path**, with persistent local layer ownership, working native TCP RPC, local weights, distributed attention state, and all launch/configuration materials. Do not deliver a single-machine engine, two-PC product, mock executor, or unfinished transport as satisfying this milestone. Internal component and simulated-stage milestones are required engineering work, not substitutes for the release.

Development will run on **Codex-managed cloud computers, which are CPU-only**, not the user's personal cloud VM. Do not assume a fixed disk quota, home-directory layout, persistent filesystem, root access, or preinstalled reference bundle. Therefore distinguish these statuses in every candidate report:

1. Source implementation complete or incomplete, with named missing requirements.
2. Native Windows compilation/link/package checks passed or pending.
3. CPU mathematical and real three-process TCP integration checks passed or pending.
4. Independent Astra review completed or pending, with unresolved findings.
5. Actual CUDA execution on SM89 and SM120 passed or **not run**.
6. Full three-PC model, vision, session, and long-context validation passed or **not run**.
7. Performance characterized or **not measured**.

A candidate may be ready for the user's first hardware trial after statuses 1–4 pass while 5–7 remain pending. Label it **“build- and CPU-integration-tested; GPU hardware validation pending.”** Do not call it production-validated, numerically GPU-verified, or known to meet throughput targets. Compilation and repeated review cannot establish those facts.

### Fixed requirements

- Exact October **1017** model and matching Engram, tokenizer, vision, and draft data. No September substitution, V4-0731, EXL3 2.8, GGUF conversion, Q3_K_M side project, or model pruning.
- Native Windows serving on all three machines. Linux is allowed for development and CPU tests, not as a WSL/Linux deployment requirement.
- Genuine three-stage RPC from the first release candidate; one active inference request/conversation at a time is acceptable. Ray is unnecessary.
- All six normally router-selected experts and the shared expert, with reference gating, biases, image masks, and model operations preserved. A cache miss waits for the exact expert or produces a clear error. Never substitute or skip computation.
- Full **1,048,576-token context**, with upfront allocation of the selected persistent KV/index capacity. Context includes input, output, reasoning, image tokens, and required reserves. No silent truncation, history summarization inside the engine, or reduced context to make memory fit.
- **FP16 KV/index storage is selectable and enabled by default.** Implement a genuine supported FP16 path; unsupported requests must fail clearly. BF16 or backend-native formats are explicit user selections, never automatic fallbacks. Wire precision is separate; default activation wire is **FP32**.
- **Exact SWA replay by default**, actively overriding upstream bounded-replay defaults. Offer bounded replay only as explicit, documented, non-bit-exact opt-in.
- Both desktops use bounded application-managed SSD → RAM → VRAM expert caches. The laptop fully preloads its assigned model state and performs **no inference-time model/expert SSD reads or model-dependent pagefile I/O**.
- Both Engram tables remain on the 4090, including when DSpark is enabled. No Engram files required on either worker in the supported default arrangement.
- Vision, sampled temperature including temperature 1, OpenAI-compatible streaming/tool/reasoning behavior, cancellation without ordinary model unloading, exact prefix reuse, and optional manual three-stage disk save/restore are part of the application scope.
- Implement DSpark with exact target verification and a non-speculative baseline. It may remain explicitly disabled pending validation; never enable unverified draft acceptance. Report any disabled feature plainly.
- No extra activation INT8 shortcut. Optional AVX-512 is an ISA choice, not permission to change precision. All required behavior must work with AVX-512 disabled.
- Keep every existing StrataRPC file, installation, snapshot, secret, repository, and launcher intact.

The selected quant and platform are settled. Do not restart generic model shopping. Resolve routine implementation choices autonomously after authorization. Report evidence and options when a fixed requirement is genuinely blocked; do not silently weaken it.

## 2. Source of truth, provenance, and investigation limits

This prompt consolidates the latest decisions in `DEEPSEEK_EXL3_HANDOFF_UPDATED.md`, especially Sections 17–18, 21, 23–24, 26, and 28–29, and follows `ASTRA_SOL_PROMPT_AUTHORING_BRIEF.md`. **Subsequent user override, 2026-10-19:** Sol will use Codex-managed CPU-only cloud computers instead of the personal cloud VM. This supersedes that VM's fixed 100 GB budget, assumed home-directory paths, prepared local bundle availability, snapshot facilities, and permission for destructive VM changes. Preserve the separation of references and new project code within the actual permitted Codex workspace. The three Windows deployment machines and all inference requirements are unchanged. Earlier handoff proposals for two-PC-first delivery, an optional laptop, mandatory stage repacking, dynamic KV allocation, GPU rentals, or a different quant are superseded. The user's current authorization permits drafting this file only; it supersedes the old prohibition on writing the draft, not the prohibition on engineering execution.

Input-file SHA-256 values for this draft:

| Attachment | SHA-256 |
| --- | --- |
| Authoring brief | `697a6b332b64089612420f4349c66f9e309f632a8c5ed4f6c60d8cb6070ef96c` |
| Updated technical handoff | `27d2689c2b02fceba7419af1b5f0eff9edb112d3d7431d86802dc400a6c5240a` |

**Evidence boundary:** The requested `~/deepseek-rpc-references/source/` bundle and `MANIFEST.tsv` were not present in the draft-authoring environment. Consequently, dependency source findings and pins below are attributed to the technical handoff and require your verification against the actual source bundle. Earlier Astra review spot-checked public project documentation; it did not qualify an integrated Windows stack. The local StrataRPC reference was available at `9bd85e5b0ad0f71cabc578fa5c38f5c438bbd5a7`; its protocol declarations and session/cancellation documentation were inspected read-only for this draft. No new dependency build, model download, GPU test, or benchmark was performed.

Your source audit must produce file/symbol/commit evidence, not merely repeat README claims. If a path moved, record its actual replacement. An absent preinstalled checkout is normal and calls for source acquisition from the listed URLs; unavailable network access or an unretrievable pin is a limitation to report, not a reason to invent findings.

### Reference inventory

**Expected Codex setup: these repositories are not preinstalled.** Fetch the source repositories using the explicit GitHub URLs in the table below after implementation is authorized. There is no need to rediscover them through a broad web search. If a checkout or archive happens to be supplied, validate its provenance and reuse it; otherwise perform source-only acquisition into the separate reference area. Avoid automatic Git LFS/model payload downloads and unexamined recursive submodule fetches. Retrieve the listed exact commits where supplied; resolve an unpinned branch to a concrete SHA once and record it before development. Keep downloaded references read-only and vendor/patch only inside the new project as appropriate. Record source URL, resolved commit, license, archive checksum, relevant symbols and local patches in the dependency manifest. If access or a recorded commit is unavailable, report the specific gap and use only permitted source access; do not invent source findings or silently substitute a similarly named project.

| Source | Starting pin or branch from handoff | Inspect for |
| --- | --- | --- |
| [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows) | `vllm-for-windows`, `3b5b6bcac87589cf35297c0283f0ba500e867645`, reported v0.31.0 | Windows runtime; `vllm/models/deepseek_v41/attention.py`, `nvidia/model.py`, `nvidia/dspark.py`, `nvidia/engram.py` under the model directory; `vllm/config/engram.py`; loader, scheduler, cache allocation, CED/SWA controls |
| [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) | `2a740a3773601c994072295fde45738ef8cea55e`, reported package 0.5.0 | `tools/check_vllm_compat.py`, `docs/VLLM_COMPATIBILITY.md`, `docs/CPU_OFFLOAD.md`, plugin entry points, packed tensors, routed/dense/MTP execution, native ABI |
| [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) | `master`; obtain exact SHA from bundle | Plugin-required kernel additions, EXL3 K2/K3 and actual other K values, Windows extension build, codebooks/rotations, CPU execution |
| [sybil-solutions/dsv41-flash-offload](https://github.com/sybil-solutions/dsv41-flash-offload) | `main`; obtain exact SHA | Cache slots, prefill staging, CPU/GPU expert scheduling, Engram disk I/O, numerical guard failures; do not copy its full ~190.5 GiB pinned-RAM mirror |
| [Zanooda/deepseek-v4.1-flash-cmp170hx](https://github.com/Zanooda/deepseek-v4.1-flash-cmp170hx) | `31318fd35837403a78ea1609d158613452257a1e` | `patches/README.md`; `patches/v41/0009-DeepSeek-V4.1-on-SM8x-PP-8-replicate-kv-source-compr.patch`, including `pp_share.py`; disk Engram patches; 1017 encoding/tool-result changes |
| [Protonburn35/StrataRPC](https://github.com/Protonburn35/StrataRPC) | `rpc-v0.1.43.1-exp29`; expected `9bd85e5b0ad0f71cabc578fa5c38f5c438bbd5a7` | `include/strata/core/layer_stage_rpc.hpp`, `layer_stage_codec.hpp`, `rpc_transport.hpp`; corresponding `src/core/` files; `conversation_snapshot.*`, `session.*`; `docs/LAYER_STAGE_RPC.md`; `tests/core/layer_stage_rpc_test.cpp`, `rpc_transport_test.cpp`; `tools/package_rpc_windows.py`; API cancellation/vision/session integration |
| [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) | `main`; obtain exact SHA | DeepSeek EXL3 integration, stream hazards, autotuning before graph capture, draft/vision and deployment assumptions |

Optional references: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), particularly `doc/env_vars.md` and `exllamav3/exllamav3_ext/cpu/moe_mul1.cpp`; [ashhart/TensorFold](https://github.com/ashhart/TensorFold); [Abyss's V4.1 geometry](https://github.com/zephel01/Abyss/blob/main/docs/deepseek41/SPEC.md). These are sources to investigate, not additional runtime dependencies by default.

Fetch only small official 1017 metadata/reference files when needed: `config.json`, tensor indexes, tokenizer/chat template/encoding, and `inference/*.py`. Do not use an unrestricted repository snapshot command that also fetches weights.

## 3. Hardware, topology, storage, and placement

| Role | Machine | Proposed inclusive layers | Required placement |
| --- | --- | --- | --- |
| A: coordinator/input stage | RTX 4090 24 GB, Ryzen 7800X3D, 32 GB RAM, Windows, SM89 | 0–17 | Tokenizer, input/vision, both Engram modules at 1/14, primary KV/index sources 2/8/14; desktop expert SSD tier |
| B: middle stage | RTX 4060 Laptop, assumed 8 GB VRAM, Ryzen AI 9 HX 370, 32 GB RAM, Windows, SM89 | 18–22 | Entire assigned model working set resident; source-14 shadow and source-20 primary; no inference model disk reads |
| C: terminal stage | RTX 5070 Ti 16 GB, Ryzen 9800X3D, 32 GB RAM, Windows, SM120 | 23–39 | Source-20 shadow, later index sources 24/28/32/36, output/head and proposed DSpark ownership; desktop expert SSD tier |

Treat **18/5/17 as the default recommendation, not a hard-coded architectural assumption**. Derive valid contiguous partitions and all shared-state dependencies from configuration. Retain all three real compute stages. Reject partitions that violate memory constraints, move an Engram module off A without an explicitly approved ownership redesign, or leave state consumers unsupported. A smaller resident B range is a possible investigation, not permission to silently remove B or transfer its computation elsewhere.

The HX 370 has 4 Zen 5 + 8 Zen 5c cores, 12 cores/24 threads, and a Radeon **890M** iGPU. Do not count its UMA reservation as extra RAM. Leave iGPU acceleration out of the first implementation. The laptop's roughly 45 W CPU and 45 W GPU figures are estimates, with sustained joint power and thermals unverified. Confirm physical GPU memory and runtime device properties on first launch.

```text
4090 NIC: 2.5 GbE ── 2.5 GbE switch ── 1 GbE laptop
                            │
                         1 GbE
                            │
                      1 GbE router ── 1 GbE 5070 Ti
```

The terminal worker's known LAN address is `192.168.50.143`. The laptop and host LAN addresses remain configuration inputs. Do not guess them. Host aggregate traffic may exceed 1 Gb/s across independent links; each worker link and the router uplink still have their own limit. Stage RPC transports activations and required state, not complete expert weights over Ethernet. Measure link negotiation, throughput, latency, and contention independently of the older Qwen setup.

### Model and disk manifest

- Selected pack: [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw), reported **43 safetensors shards, approximately 196.3 GiB**. Pin its exact revision; do not invent a SHA absent from the bundle.
- Include `exl3_k_map.json`, quantization metadata, tensor indexes, tokenizer/chat template, vision and MTP tensors. Discover actual names/layouts from the pinned artifacts.
- Original 1017 source revision reported by the handoff: `1bae0e1088993b9b82de08f59de5be5a34e42170` at [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
- Engram source files on A only: `model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors`, approximately 203.07 decimal GB / 189.1 GiB combined. Obtain authoritative hashes from the pinned publisher/source metadata and verify them. Never fabricate checksums or mix September shards.
- Full pack copies on **all three local SSDs are allowed**. Stage repacking is optional, never a mandatory setup prerequisite. Materialize only owned tensors. Workers must skip absent Engram shards before opening or allocating them.
- Host pack plus Engram is approximately **385.4 GiB before** generated indexes, temporary downloads, snapshots, dependencies, and safety space. B/C each need the pack plus local application space. Report bytes and GiB explicitly and inspect free disk before user-side downloads.
- Full weights and Engram must never be downloaded to the Codex development environment, regardless of its available disk quota. Small synthetic safetensors/EXL3 fixtures and header/index metadata are sufficient there. If exact packed offsets require shard headers, use bounded header retrieval or a metadata-only tool run later on the user's files; label missing evidence.

## 4. First engineering checkpoint: audit before invasive implementation

After execution is authorized, inspect the actual Codex environment, repository attachments, applicable workspace instructions, writable roots, operating system, available CPU/RAM/disk, toolchains, network policy and authorized credentials without exposing secrets. Create a dedicated new implementation Git repository in a permitted workspace location, for example **`<CODEX_WORKSPACE>/projects/<descriptive-name-you-choose>/`**, or use an explicitly attached new-project checkout. Do not put it inside the existing StrataRPC repository. The old `~/projects/` path is not mandatory in Codex and may not be writable. Keep any supplied `~/deepseek-rpc-references/` bundle read-only; if absent, obtain permitted source-only references into a separate workspace reference area, pin them, and record its resolved location. Do not put application edits, build outputs, logs, or test history in reference clones. Choose a separate Windows install root such as `C:\DeepSeekRPC` and data root `C:\DeepSeekRPC-data`.

Produce three design artifacts before broad runtime changes:

1. `docs/COMPATIBILITY.md`: pinned source/toolchain matrix, symbol-level integration evidence, licenses, unresolved blockers, and a justified foundation decision.
2. `docs/MEMORY_PLAN.md` plus a machine-readable tensor/allocation manifest: actual per-stage weights, host/device residency, cache replicas, lifetime peaks, and laptop preflight rules.
3. `docs/RPC_PROTOCOL.md`: versioned type table, state machine, shared-state dependency graph, transaction invariants, failure handling, and test vectors.

These artifacts should lead directly into engineering. Do not endlessly repeat feasibility research or demand early full-model hardware experiments.

Audit in this order:

1. **Provenance:** reconcile `MANIFEST.tsv`, source commits, licenses, selected 1017 metadata, and the exact EXL3 K-map. Record checked files and symbols.
2. **Windows vLLM graph:** verify V4.1 CED/CSA2/mHC, Engram, sparse index, image routing, DSpark, exact replay, loader ownership, and serving entry points at the pinned 0.31 fork. Identify the source-local cache checks that reject the proposed partition. Do not just delete their `NotImplementedError` guards.
3. **Plugin interfaces:** inspect and run `tools/check_vllm_compat.py` against the actual Windows tree using that revision's CLI. Record every pass/failure and adaptation. Its written 0.30 audit and source-only CI are not a qualified Windows 0.31 runtime. Add actual installed-integration imports/registration checks where CPU-only execution permits; report driver-dependent imports separately.
4. **EXL3 execution:** audit ABI, codebooks, rotations, scales, tensor strides and packing, routed/shared/dense/head/MTP handling, K2/K3 and every other precision present in the pack. Check actual hidden width **5120** and expert intermediate width **2304**, one-row decode and chunked prefill. A fast kernel demonstrated for other dimensions or K4 does not establish our dispatch. Record real fallback paths and their workspace costs.
5. **Native Windows and CUDA:** lock a compatible Python version (plugin reports >=3.12), Torch, CUDA, ExLlama extension, Triton-Windows, FlashInfer/FlashMLA/DeepGEMM or other actually used dependencies, compiler and build tools. Resolve SM89 and SM120 compilation and runtime dispatch separately. SM121 Spark support is not SM120 proof. Check known sparse/vision shape issues #56837 and #59203 against the exact source before applying old fixes.
6. **Disk Engram:** locate native table loading and lookup hooks; replace the full-table pinned-RAM requirement with exact bounded Windows disk lookup. Audit source hash/lookback, quant scales/offsets, and downstream absence of Engram files.
7. **FP16 attention feasibility:** inspect packed `fp8_ds_mla`/`nvfp4_ds_mla` expectations and any plain BF16 paths. Specify the actual FP16 compressed-KV/index storage and supported attention/index arithmetic on both architectures. A dtype flag, allocation cast, or BF16-capable kernel is insufficient. If a new backend/adapter is needed, name its operations, workspace and test plan.
8. **Cache/graph interactions:** establish how dynamic expert slots update EXL3 pointer tables, lifetimes, capture/replay, and asynchronous completion safely. Stock UVA means GPU reads pinned host RAM; it provides neither CPU MoE computation nor a bounded SSD expert cache.
9. **1017 client encoding:** compare selected chat template/encoding with October tool-result, reasoning, and DSML semantics. Pin the official/reference rendering behavior and any existing parity vectors.

Prefer Windows vLLM as the foundation because it supplies the DeepSeek graph and serving machinery. Reuse narrow StrataRPC lifecycle patterns rather than Qwen model code. If evidence makes that foundation impractical, document the smallest alternative assembled from existing verified components, costs, and affected requirements, and seek user approval before a fundamental architecture change. Do not silently pivot to a different model, OS, or neural implementation.

## 5. Architecture and module responsibilities

Use a small, explicit set of modules around the chosen runtime. Exact directory names are yours to choose, but preserve these ownership boundaries and test seams:

| Module | Responsibility and invariant |
| --- | --- |
| Configuration and provenance | Validate complete role configurations, artifact hashes, precision choices, layer boundaries, dependencies, secrets and effective settings. Produce one run fingerprint and separate numerical/session identities. |
| Stage planner and loader | Map every tensor and model-state dependency to an owner; generate replica plans; selectively load owned tensors without full-shard materialization; enforce per-node allocation ledgers. |
| Model adapter | Preserve the pinned 1017 graph, routing, CED schedule, mHC, image masks, and exact SWA behavior. Expose bounded stage execution and state transactions. |
| EXL3 executor | Execute correct packed experts and other quantized components; expose supported shapes/precisions; manage CPU/GPU dispatch and lifetimes; reject unsupported modes. |
| Expert storage/cache | Read exact local experts; manage bounded RAM/VRAM residency, asynchronous transfers, leases and eviction; resident-only B backend has no inference miss-to-disk path. |
| Engram backend | Exact October lookup on A using bounded caches and disk I/O, preserving token lookback and quant metadata. No giant pinned table. |
| Shared attention state | Own source caches, replicas, compression phases, SWA, sparse index/candidate state, local block maps, tentative writes, rollback and prefix generations. |
| Coordinator and transport | Authenticate ranks, negotiate capabilities, schedule one request, carry versioned tensors/state deltas, commit/abort work, handle cancellation and faults. |
| Terminal sampling and DSpark | C owns output selection and draft execution. Implement requested sampling and exact target verification; only globally committed tokens are emitted. |
| Vision and API | A encodes images, renders exact 1017 prompts and exposes OpenAI streaming/tool/reasoning semantics. Workers retain required image/position metadata. |
| Session persistence | Consistent three-stage snapshots, bounded streaming, compatibility validation and safe restore. Prefix reuse is independently testable. |
| Diagnostics and packaging | Metrics, deterministic self-tests, capability report, Windows launchers, dependency locks, release manifests and review packets. |

Start with a correctness-first scheduler and conservative stream ordering. Optimize overlap after state invariants hold. Avoid speculative prefetch that assumes future expert IDs are known before routing; estimates may guide prefetch but never alter selected experts or bypass required reads.

No requirement for tensor parallelism, NCCL collectives, Ray actors, multi-agent concurrency, a public inference service, or a GUI. If an upstream dependency requires distributed primitives, audit their Windows/process assumptions; do not let an incidental dependency force a different deployment topology.

## 6. Tensor loading, memory planning, and strict laptop residency

### 6.1 Count actual allocations

Read the exact tensor index, quantization metadata, packed byte ranges, and lifetime graph. Build a per-node allocation ledger showing **disk bytes, owned model bytes, host committed bytes, host physically resident/locked bytes, device allocated/reserved bytes, and transient high-watermarks separately**. Estimate first; on hardware, replace estimates with observed allocations.

The simple routed-expert estimate is:

`384 × 3 × 5120 × 2304 = 13,589,544,960 parameters/layer`.

According to the selected pack description, layers 18–22 use 2-bit routed experts and the other 35 layers use 3-bit routed experts. Before extra metadata/non-routed tensors, this implies roughly 85.4 GiB on A, 15.82 GiB on B, and 80.7 GiB on C. These are **owned expert bytes on disk**, not claims that desktop weights fit in RAM/VRAM. Confirm the K-map rather than averaging the whole checkpoint over 40 layers.

Account separately for:

- Routed experts, shared experts, dense projections, router/bias/normalization parameters, token embeddings, output head and any required tied-weight copies.
- Draft weights versus draft working state. The source has three draft layers and 128 experts per draft layer; obtain actual EXL3 bytes, not an original FP4 estimate.
- A's vision tower plus aligner, approximately 0.90 GiB of included weights; the aligner alone is approximately 140 MiB. Do not add those weights twice to pack storage. Include image activations/crops and encoder workspaces separately.
- Logical KV/index sources, downstream replicas, local SWA/compression rings, later index-source state, candidates/top-k, block tables, allocator overhead and version metadata.
- Expert reconstruction/GEMM scratch, CPU kernels, pinned/DMA staging, CUDA graphs, autotuning peaks, RPC frame assembly, token/image history, snapshot staging and runtime libraries.
- Host/device duplicate copies and whether a GPU cache entry evicts or retains its RAM copy. Count every physical copy exactly once.
- Windows/display reserve, currently available RAM/VRAM, driver/WDDM budgets, lock limits, fragmentation and other processes. Report units consistently; nominal RAM+VRAM is not a unified allocation pool.

Include boot, autotuning, ordinary decode, largest accepted prefill, image+prefill, speculation, cancellation, save and restore lifetime peaks. Do not simply add mutually exclusive buffers or overlook overlapping ones. If vision encoding can safely precede prefill, schedule it to reduce peaks and document that lifetime proof.

### 6.2 Eager context allocation and replicas

The illustrative shared FP16 cache formula is:

`(3 × 0.5 + 1) × (512 + 128) × 2 × 1,048,576 = 3.125 GiB`.

This estimates selected long-context KV/index-key arrays, not total runtime memory and not an existing qualified vLLM FP16 allocation. With the proposed replicas:

| Node | Illustrative long-context arrays | Estimated GiB |
| --- | --- | ---: |
| A | Primary sources 2, 8, 14, each half-rate | 1.875 |
| B | Source-14 shadow plus full-rate source-20 primary | 1.875 |
| C | Source-20 shadow | 1.250 |
| Aggregate | Sources plus two replicas | 5.000 |

Derive actual sizes from the implemented backend, including each index source and all other state. Allocate persistent capacity upfront for the chosen 1M format; logical token validity can grow without allocating a larger pool. Do not materialize a full `[prefill_rows, 1M]` score matrix or retain all full-prompt mHC activations without budgeting it. One million FP32 `[4,5120]` activation rows alone are approximately **80 GiB**. Exact CED/replay must have a bounded, justified storage/recomputation schedule; the logical KV estimate does not solve this issue.

### 6.3 Loader behavior

Construct an ownership/load index before opening large payloads. Use bounded reads/views of needed safetensors ranges. Avoid `load-all-then-discard`, whole-shard temporary CPU tensors, full checkpoint CUDA allocation, or opening absent Engram shards on B/C. Account for small required non-Engram tensors sharing a source shard; assign or extract those explicitly rather than requiring workers to carry the large table.

Validate exact EXL3 trellis/rotation/scale metadata and ownership for dense/MTP tensors too. Keep tensor-name aliases explicit, checked, and tied to the pinned version. A missing tensor is an error, never a randomly initialized module or silently skipped weight.

### 6.4 Laptop admission and runtime rules

B's model-dependent inference working set must stay in physical RAM/VRAM. Startup SSD loading is allowed. After warmup/admission, no weight demand reads, file-mapping hard-fault dependency, lazy module/model initialization, or pagefile model offload is allowed.

1. Compute and reserve separate host/device budgets with conservative OS/display headroom and worst accepted prefill/vision/speculation buffers.
2. Preload all assigned tensors and necessary metadata, initialize the execution paths, allocate eager caches, touch host pages, and establish a documented residency mechanism.
3. Check all locking/pinning results and OS limits. `VirtualLock`, registered/pinned CUDA memory and working-set limits have different contracts; increasing a working-set minimum or turning off pagefile is not proof of physical residency. Inspect WDDM device budgets/eviction implications too.
4. Make the resident-only storage backend structurally reject attempted inference-time expert file reads. Close or otherwise remove access to backing payload handles after preload where feasible; do not retain demand-mapped model tensors as the only copy.
5. Instrument resident/locked bytes, model-dependent hard faults, allocation pressure and unexpected reads. Reserve scratch up front or use a bounded preallocated arena. Do not depend on a monitor discovering paging only after the guarantee has already failed.
6. If required allocations/residency cannot be established or maintained, refuse admission or invalidate/stop the request safely. Never automatically page, lower context/precision, drop layers, or remotely fetch B's expert weights.

The promise concerns model-dependent disk I/O, not all Windows/driver activity. Do not claim the application can forbid unrelated OS disk activity. If the chosen Windows allocation APIs cannot support the required guarantee, report that as a real design blocker with evidence; do not label an observed low hard-fault count a hard guarantee.

Snapshots are administrative operations outside active inference. Prefer streaming B's snapshot state to host storage so no laptop snapshot file is required. Restore/preload must finish and residency be re-established before inference admission. Keep diagnostic logs distinct from model data I/O and bounded.

## 7. Desktop expert cache and exact Engram backend

### 7.1 Three-tier expert cache

Each desktop owns its own local cold expert files and independently sized RAM/VRAM caches. Use keys containing model fingerprint, layer, expert, projection/segment, quant metadata identity, and slot generation. Track a complete expert bundle or explicitly track all required segments; partial presence must never be treated as a complete expert.

Specify cache transitions such as `ABSENT → READING → RAM_READY → COPYING → GPU_READY`, with `FAILED` and safe eviction paths. These describe availability, not permission to mutate tensor math. Required guarantees:

- Deduplicate in-flight reads; hold leases while CPU/GPU consumers or DMA reference a slot.
- Wait for completion events before publishing data or recycling memory. A failed/short read never marks a slot ready.
- Bound file I/O queues, pinned staging, reconstructed weights, read-ahead, and GPU promotion. Prevent unbounded memory growth during long prefill.
- Use exact `ReadFile`/OVERLAPPED or a qualified equivalent on Windows. Test buffered and optional unbuffered I/O, alignment, offsets, short reads and cancellation. Mapping metadata is acceptable; OS pagefile is not the offload algorithm.
- Record RAM copies retained behind GPU entries; avoid wasting most capacity on an unintended full mirror. Adaptive hotness/frequency/recency and per-layer protection may change residency, never routing.
- On pressure, evict eligible application cache entries while retaining required in-flight data. Do not request the OS to absorb an oversized model into pagefile.
- Keep cache identity and pointer tables synchronized with EXL3 handles and CUDA graphs. Disable graph capture initially if dynamic slots cannot safely update captured addresses; correctness comes first.

Use separate prefill/decode policies. Large chunks can select almost every expert in a layer; process routed rows in bounded expert groups to reuse a loaded expert across its selected rows. Preserve token scatter, selected-expert weights, and accumulation semantics. Prefetch only within bounded budgets and with correct cancellation/epoch handling.

Do not adopt Sybil's reconstructed-FP16 fast prefill blindly: its reported D061 numerical guard failed. Such a path needs independent reference evidence before becoming default. Its ~190.5 GiB pinned host mirror and GPU-hit figures do not predict this system's SSD misses.

CPU expert compute and GPU reads from pinned host RAM are distinct choices. If CPU compute is used, supply a faithful EXL3 executor and test its actual precision. The plugin's UVA offload does not implement CPU MoE. If the only available fast CPU path quantizes activations to INT8, retain a validated higher-precision baseline or GPU execution; do not enable that shortcut implicitly.

Persisting hotness metadata across runs is optional. Expert payload identity remains tied to the model/K-map, and a stale profile can only affect placement/performance.

### 7.2 Engram on A

Implement a Windows disk-backed exact-row gather backend for both October tables. Use the pinned source's hash, lookback, row indexing, scaling/dequantization, and projection behavior. Keep file reads and host/GPU staging bounded. Original quantized table values and scales must be preserved; do not approximate a missing lookup, use September data, or reinterpret encoded values as a different format.

Audit and adapt Zanooda's `storage=disk`/row gather concepts. Linux `preadv`, pthread helpers, runtime GCC compilation, and allocation assumptions require deliberate Windows equivalents. Do not assume the old patch applies directly to the current source layout.

Cache exact rows and batch/coalesce reads where useful. Test hashes and lookup results across prompt chunks, prefix reuse, speculative rejection, cancellation, image sentinels and restore. A's Engram lookup history follows the committed/tentative token timeline, not merely the last network packet.

C's draft implementation must not accidentally load or apply Engram locally. Inspect `dspark.py` and its required feature inputs at targets 37/38/39; if an unexpected earlier-layer dependency exists, account for it in the protocol or identify a blocker. Do not resolve it by copying the tables to C or skipping target computation on A.

## 8. Precision, attention, CED and fidelity policy

Keep these choices separate in config, handshake, snapshots and logs: weight quantization, source cache storage dtype/layout, index-key storage, attention/index compute and accumulation dtypes, activation transport dtype, expert execution mode, and SWA replay policy.

Default contract: EXL3 2.9; requested FP16 KV/index; FP32 activations and `pre_mix` on the wire; exact SWA; all selected experts; no added activation INT8. Token IDs, block IDs, candidate indices and positions remain integers with lossless encoding. Cached FP16 rows may travel in their exact storage representation; FP32 activation transport does not mean all metadata or cache payloads must be converted to FP32.

Print requested **and effective** types per component. A backend storing FP8/FP4 while advertising FP16 is a defect. Converting FP16 storage into a lower-precision kernel operand is not automatically an equivalent FP16 path; expose all conversions and validate the stated numerical contract. FP16 is not automatically more faithful than BF16/native formats, especially for range/overflow. Test finite values and define justified per-mode reference tolerances.

The model geometry to verify includes 40 layers, hidden width 5120, `hc_mult=4`, window 128, KV sources 2/8/14/20, index sources 2/8/14/20/24/28/32/36, candidate source 20, 2,048 candidate blocks of 8 positions, and index top-k 512. Derive consumers from actual config and graph.

CED requires explicit phase semantics. Layers 0–19 are the encoder; layer 20 builds global decoder KV over prompt positions. B spans layers 18–22, so it crosses the relevant phase boundary. Source Section 29 reports that processing only the last 128 rows in layers 21–39 is the **bounded replay** optimization, not an unconditional exact schedule. Derive exact prefill/replay from the pinned implementation/reference; do not promise its reduced decoder work in default exact mode. Actively set the equivalent of `--no-swa-bounded-replay` when exact is selected and test the effective backend setting.

Offer `--swa-replay exact|bounded`; `bounded` must be user-selected, visibly reported, and part of session identity. Run dedicated comparison fixtures and document its divergence. This opt-in does not authorize other approximations.

Offer `--cpu-isa auto|avx2|avx512` when implemented. Check CPUID and OS XSTATE before dispatch, retain a validated baseline, reject unsupported forced AVX-512, and never execute AVX-512 just because a compiler produced it. The HX 370's narrower execution datapath is not absence of ISA support. Audit native Windows dispatch separately from GNU/Linux attributes. AVX-512 optimization is nonblocking for the initial release; report availability honestly.

Freeze numerical reference fixtures independently of optimized/distributed code. For each operation, record reference provenance, shapes, input generation, dtype, absolute/relative error bounds, exact metadata expectations and rationale. Require exact expert IDs/tables/positions where defined; investigate tie behavior explicitly. Track per-layer outputs, logits, distributions and deterministic completions. Do not relax tolerances merely to make an optimization pass. Sampled temperature-1 text equality is not a valid parity test by itself.

## 9. Distributed protocol contract and state machine

Write and implement a versioned protocol specification before integrating the whole model. Use real persistent TCP connections with typed, bounded serialization. A broker or supervisor does not count as the inference transport. No pickled arbitrary objects, remote command execution, raw CPU/GPU addresses, or model-weight streaming as a substitute for stage ownership.

A coordinates control; computation flows A → B → C; C returns proposed/committed output results to A. Separate or prioritize control traffic so a large prefill payload cannot indefinitely block cancellation/health messages. Authenticate every peer and bind worker listeners to explicit LAN interfaces. Use a reviewed authenticated transport/handshake; do not invent cryptography or ship a fixed shared secret. Keep secrets out of command-line examples, logs, source and artifacts.

### 9.1 Envelope and message schema

Define fixed-width integer encoding, byte order, tensor layouts, supported dtype identifiers and length limits. A v1 envelope must identify at least:

- Magic, protocol major/minor, message kind, header/payload byte lengths, flags and integrity information.
- Run/config identity, sender/receiver stage IDs, session/request IDs and generation epoch.
- Step/operation ID, chunk/sequence ID and any acknowledgement/replay identity.
- Absolute token positions/validity, transaction identity, slot-map generation, and bounded tensor descriptors.

Tensor descriptors identify a semantic field/source layer, dtype, rank/dimensions, layout, exact byte count, and destination logical rows. Validate lengths with checked arithmetic before allocating. Never trust a peer's size to allocate a full-context activation matrix. Fragment large tensors into bounded frames with backpressure; count both serialization and reassembly memory.

Specify these message families, or an equivalently complete typed API:

| Family | Required contents / outcome |
| --- | --- |
| `HELLO`, `CAPABILITIES`, `PLAN_ACK`, `READY` | Model and component hashes, runtime/kernel ABI, owned ranges, dependency graph, formats, exact/bounded policy, limits, resource preflight and authenticated peer identity |
| `BEGIN`, `PREFIX_PROBE`, `PREFIX_ACCEPT`, `RESET` | Full numerical/session identity, token/image prefix identity, candidate reusable positions and per-state generations; unanimous safe reuse decision |
| `STEP_BEGIN`, `SOURCE_DELTA`, `STAGE_INPUT`, `STAGE_DONE` | Phase, rows, absolute positions, validity, hidden states, pre-mix, new KV/index rows and sparse metadata, explicit dependencies and completion |
| `DRAFT`, `VERIFY_RESULT`, `COMMIT`, `COMMIT_ACK`, `ABORT` | Proposal identity, target verification results, accepted count and terminal correction/bonus token semantics, committed/tentative state boundaries and acknowledgements |
| `CANCEL`, `CANCEL_ACK`, `HEARTBEAT`, `ERROR` | Request/epoch and safe boundary; typed recoverability, owner operation and diagnostic code; no silent state reuse after uncertainty |
| `SNAPSHOT_PREPARE`, `SNAPSHOT_CHUNK`, `SNAPSHOT_SEAL`, `RESTORE_PREPARE`, `RESTORE_APPLY`, `RESTORE_ACK` | Snapshot ID, per-stage inventory, offsets/hash/length, compatibility identity and coordinated completion |

Exact names may change, but document the implemented mapping and prove every required transition. Use immutable transaction descriptors and explicit acknowledgements; networking success alone is not state commit.

### 9.2 Source-to-shadow data

For the default split:

| Boundary | Required data |
| --- | --- |
| A → B | `hidden_states [rows,4,5120]`; `pre_mix [rows,4]`; token IDs/positions and image/validity metadata; new source-14 compressed KV and index-K rows; required source top-k and compression/window metadata |
| B → C | Corresponding activations/pre-mix and token/position metadata; new source-20 compressed KV/index-K rows; layer-20 candidate blocks and required top-k data |
| C local | Index sources 24/28/32/36, downstream consumer state, output/head, draft target features and sampling |

Verify these fields against the actual graph; add missing semantic state rather than assuming this table is exhaustive. Register replicas with compatible source identities/specs and translate to each recipient's **local** physical slots. Identical layer names do not imply identical physical block addresses. Associate deltas with logical positions, validity/compression phase, block generations and transaction IDs; scatter only after mapping validation and before consumer execution.

Forward only needed newly written rows and sparse metadata during ordinary steps. Full history transfer is for explicit synchronization/restore if needed, not every token. Do not copy reference constants of 584-byte KV rows or 132-byte index rows into the FP16 implementation; those belong to another packed format. Include candidates/top-k, scales and validity in byte accounting.

At 5120 × 4 × 4 bytes, FP32 activations alone are **81,920 bytes/token/boundary**, about **328 Mb/s at 500 tokens/s**, before pre-mix, cache/index deltas and framing. A 4096-row activation tensor is 320 MiB. These are payload calculations, not throughput predictions. Budget simultaneous buffers and both worker links, and distinguish sequential decode latency from overlapped chunk throughput.

### 9.3 Lifecycle states

| State | Entry work | Exit condition |
| --- | --- | --- |
| `BOOT / PREFLIGHT` | Validate files/config/device; load owned tensors; allocate persistent pools; establish B residency; run required startup diagnostics | Local capability and memory plan accepted |
| `CONNECTING` | Authenticate, compare identities, establish topology and source/replica mappings | All three ranks agree and report ready |
| `IDLE` | Model remains loaded; no active tentative work; optional reusable committed session | Coordinator admits one operation |
| `PREFILL` | Render/tokenize/encode, negotiate prefix, run bounded exact phase-aware chunks and source deltas | All required prompt state committed consistently |
| `DECODE / VERIFY` | Run the next target step or verified proposal window through A/B/C | All ranks acknowledge commit, or rollback/invalid-state handling completes |
| `CANCELLING` | Stop admission; complete or abort bounded in-flight work; reconcile tentative state | Return to coherent `IDLE`, or mark session invalid |
| `SAVING / RESTORING` | Quiesce all ranks; stream/validate versioned state transaction | Globally consistent idle state or explicit invalidation |
| `FAULTED_SESSION` | Reject further use of uncertain cache state; retain healthy loaded weights where possible | Coordinated reset/replay or explicit restore succeeds |
| `FAILED_DEVICE / STOPPED` | Report fatal device/runtime failure; no partial inference continuation | Explicit process/device recovery and fresh handshake |

### 9.4 Execution invariants

**Startup:** all ranks agree on model/quant/component hashes, stage plan, capacity, requested/effective precision, replay policy, protocol/ABI and feature capabilities. B/C may lack Engram payloads because they do not own Engram operations. Reject accidental duplicate/overlapping/missing layers or mismatched components. Avoid rehashing hundreds of GiB every request; use validated manifests and an explicit startup/integrity policy.

**Prefill:** run the pinned exact CED schedule, including layer-20 global-source production and required later replay. Make dependency barriers explicit; consuming stale/partial source state is forbidden. Chunk boundaries preserve Engram history, compression pairs, absolute positions and image masks. Odd-sized chunks must not reset half-rate compression alignment. Prefix reuse may reuse encoder state while requiring decoder-state replay; derive that rule from the graph, not token-prefix equality alone.

**Decode:** one logical target forward traverses all necessary stages. C performs sampling using validated per-request settings and returns compact token/result data; do not transmit the full vocabulary distribution on the normal hot path. All ranks agree on the committed timeline before A publishes the result. Track **token-history length, KV-valid positions per source, emitted output cursor and any sampled-but-not-yet-forwarded token separately**. They are not universally equal. Define EOS, stop-string and output-limit handling without off-by-one cache writes.

**Speculation:** C owns DSpark proposals and required targets 37/38/39 after source audit. All target verification still traverses A, B and C, including A's Engram. Record tentative KV/shadow/index/SWA writes, draft state, history and RNG effects. Implement a justified exact acceptance algorithm for both greedy and sampled target modes; rejection sampling requires the correct proposal/target probabilities and correction distribution. A heuristic confidence threshold alone is not a distribution-preserving verifier. Discard rejected suffix state on every rank, including overwritten ring rows and source replicas. Commit only accepted/output-relevant positions; test zero/full/partial acceptance, correction/bonus tokens, EOS inside drafts, and output exhaustion. Keep speculation off until its validation gates pass; never silently fall back to a different sampling distribution.

**Cancellation:** HTTP disconnect, Stop, Ctrl+C, or a replacement request must not normally exit/unload healthy model processes. Finish a bounded chunk/verify+commit window or perform a coordinated abort. After acknowledgements, discard tentative state and accept the next request only from a known coherent boundary. Keep network/device deadlines active. If safe reconciliation fails, invalidate the session and reset/replay; never continue from uncertain state. The StrataRPC cancellation regression is a required test: interrupt while thinking/generating, then immediately submit a new request without reloading weights.

**Retries and faults:** replay a completed operation only when its identity and payload digest match and the cached result remains valid. Reject conflicting duplicates and stale epochs. Do not claim exactly-once execution across process crashes without a durable design. If a disconnect leaves commit outcome uncertain, mark the session unusable until a coordinated reset or validated checkpoint restore; do not guess the committed length. A timeout is not permission to skip a rank or expert. Retain healthy model allocations unless actual corruption/device failure requires rebuilding them.

**Prefix reuse:** compare canonical token history, image/content/position identity, tokenizer/template, model and precision policy, source generations, Engram lookback, SWA and draft validity. A token-equal prompt with changed image data is not a hit. On mismatch or missing compatible checkpoint, replay exact state safely. Log reused versus replayed/new tokens and the reason. Test repeated tool turns and OpenCode compaction changing the prefix.

### 9.5 Three-stage snapshots

Implement manual save/restore as an optional feature. Save only while the cluster is idle and all three engines remain loaded. Serialize inference state, not the OpenCode chat database. Document that the client must retain messages, tool results and original images independently.

Use a coordinator-controlled transaction with a unique snapshot ID, per-stage checksums/lengths, accepted timeline, pending-token/RNG semantics, numerical/session fingerprint, state inventory and committed manifest. Include source and shadow KV or a proven exact reconstruction plan, local SWA/compression state, sparse candidate/index state, Engram lookback/hash state, image positions/identity, and draft state or a proven exact reset/reconstruction rule. Do not serialize reusable expert cache payloads or model weights as conversation state.

**Recommended initial storage policy:** stream bounded chunks from B/C to a host-owned snapshot directory, and write A's chunks locally. This keeps laptop model/session storage out of the inference path and gives the user one complete snapshot bundle. It costs network time; report that cost and preflight host disk space. Per-stage desktop-local storage can be an alternative only with a complete manifest and missing-file validation. Do not inherit StrataRPC's assumption that only a small descriptor crosses the network if the new policy actually streams state.

Write temporary files, verify all stage streams, then publish the complete manifest atomically. An incomplete snapshot is never listed as restorable. Validate all manifests, bounds and file hashes before mutating live state; refuse foreign formats/configurations. Use opaque snapshot IDs and confined paths, not peer-supplied arbitrary filenames.

Save and restore must use bounded staging, not an extra full multi-GiB RAM copy on each 32 GB node. A full double-buffered restore may not fit. If restoring in place, quiesce all nodes and expose the new session only after every rank acknowledges success. On partial failure, invalidate all affected sessions and retain loaded weights for reset/replay; do not pretend the old session survived. Re-establish B residency before returning to ready.

Conservatively fingerprint model/K-map/Engram/vision/template, split, context, cache layout/precision, wire mode, SWA, draft configuration and snapshot ABI. Any future relaxation of compatibility needs a proven conversion path. Define same-seed continuation semantics explicitly; don't assume the old Strata snapshot format saved sampling settings.

## 10. Vision, OpenCode and user-facing behavior

A executes the pack's own vision tower and aligner. This EXL3 pack includes them; do not demand Qwen's separate `mmproj` or `strata-vision.exe`. Include actual vision memory in preflight, correctly render image markers/grid/positions, and propagate IDs/masks and state needed by downstream MoE/attention. Test multiple images, text following an image, changed images with equal token counts, image prefixes, speculative verification and restore. Do not disable vision to evade an SM120 shape/kernel problem.

Expose an OpenAI-compatible `/v1` API with `/v1/models` and streaming chat completions. Preserve the exact October 1017 tokenizer/template, reasoning controls/content and DSML tool behavior, including tool-result-only turns and reasoning retention around tool calls. Reuse qualified parsers where possible; do not invent lossy regex cleanup as a substitute for correct encoding.

Required client tests include:

- Ordinary and streamed responses; correct deltas, finish reasons, usage/error reporting and stream termination.
- Temperature 0 and 1, explicit seed/settings, and truthful handling of supported top-p/top-k/min-p or other sampling fields. Reject unsupported requested controls clearly rather than silently ignoring them.
- Tool-call IDs, streamed JSON arguments, multiple calls if emitted by one response, tool results and subsequent reasoning continuation. One active request does not prohibit multiple tool calls in an answer.
- Image input using the supported OpenAI-compatible representation; limits and clear errors for unsupported payloads.
- Cancellation and immediate next request, exact prefix reuse, manual save/restore, and client history after restore.
- True 1M context accounting; separate requested output limits and client defaults from context capacity. Do not equate the old OpenCode 32,000 output cap with model context or prefill chunk size.

The handoff reports official guidance recommending OpenCode >=1.14.24; verify the actual installed version and adapter before final instructions. The user's provider is `@ai-sdk/openai-compatible` inside Docker, using `http://host.docker.internal:8081/v1`. Advertise the correct context and configure explicit output limits without claiming unsupported reasoning-effort behavior. Keep new DeepSeek model/provider configuration separate from the Qwen profile.

Bind the client API to `127.0.0.1` by default. Verify Docker Desktop's host route on the user's deployment; do not silently bind to `0.0.0.0` to fix connectivity. Any intentionally broader binding requires authentication and a documented, narrowly scoped networking change. Worker RPC is authenticated and LAN-bound separately.

Because Qwen already uses port 8081, provide a clear port-conflict refusal. The requested DeepSeek 8081 profile is used when the user has stopped Qwen; provide an explicit coexistence example using 8082 and a separate client base URL. Do not terminate Qwen or overwrite its configuration automatically.

User-visible diagnostics should explain the failing stage and actionable cause. Keep implementation details out of normal chat flow. Verbose logs are opt-in and redact prompts, images, tool payloads and secrets by default.

## 11. Windows builds, packaging, configuration and self-tests

### 11.1 Reproducible build matrix

Pin supported versions after the source audit; do not guess a compatible Torch/CUDA/Triton combination from version numbers alone. Record source commits, patch hashes, Python/wheel versions, compiler versions, CUDA architecture flags and commands in a lockfile/build manifest. Preserve third-party licenses and required source/attribution. The plugin is reported AGPL-3.0-only and vLLM Apache-2.0; audit actual reuse/redistribution/network-service obligations before choosing the project's license or publishing binaries. Do not label the combined application MIT-only by assumption.

| Environment | Required checks | What it cannot prove |
| --- | --- | --- |
| Codex-managed CPU-only development environment; detect actual OS and limits | Metadata parsing, independent numerical fixtures, cache/state/serializer tests, three real processes over TCP with simulated computation, API/schema tests, source/package checks | Windows ABI, WDDM, CUDA execution or real model speed |
| CPU-only Windows x64 CI | Native Windows builds/import/link/package checks, actual Windows TCP and file-I/O tests, PowerShell/BAT validation, CPU kernels and supported SIMD dispatch, explicit SM89 and SM120 CUDA compilation | GPU math, GPU memory fit, CUDA runtime/driver behavior or distributed model quality |
| User 4090 and 4060, after reviews | SM89 kernel goldens, actual memory/residency, owned stage execution, source/replica and vision checks as applicable | SM120 correctness |
| User 5070 Ti, after reviews | SM120 kernel goldens, sparse/image shapes, terminal head/draft and cache behavior | Whole-cluster correctness alone |
| User's full three-PC system | End-to-end text/image/tool flows, prefixes, cancellation, snapshots, context progression and performance | General hardware-independent guarantees |

Compile both target architectures explicitly rather than auto-detecting a GPU on CI. Standard CMake CUDA targets may use `89;120`; Torch extension/build systems require their corresponding explicit architecture settings. Audit generated binary targets and package the required runtime DLLs under their licenses. If a dependency is JIT-only or needs a GPU to compile/autotune a path, record that as first-run work and do not report it as CI-executed.

Build the actual Windows application and native extensions, not just a stub launcher. Inspect dependency DLLs and exercise device-independent startup/config commands on the Windows runner. A driverless failure in a GPU-only operation must be classified honestly, not counted as a passed import/launch test. Avoid build logic that silently skips CUDA targets when no GPU is detected.

Use bounded kernels/chunks and startup autotuning under WDDM. Do not instruct the user to disable TDR in the registry globally. Complete required EXL3 tuning before capture; qualify stream concurrency and dynamic-cache pointer safety. Start serialized if necessary. After a device reset, surface the failure and invalidate affected inference state rather than reusing corrupt allocations.

### 11.2 Package contents

Ship versioned Windows binaries/wheels or a pinned install bundle for all roles, with:

- `BUILD.json` or equivalent: source SHA, dependency/patch versions, target architectures, protocol/snapshot ABI, build time and verification status.
- SHA-256 checksums, license notices, reproducible build instructions and dependency installation requirements.
- Separate A/B/C configuration examples and BAT/PowerShell launchers; one source of truth for shared identity/precision/split settings.
- `--version`, `--print-effective-config`, `--preflight`, `--self-test`, diagnostic bundle and benchmark modes.
- Model inventory/integrity tooling that operates on local files and does not automatically download the full pack into the Codex development environment.
- OpenCode example, startup/shutdown order, manual snapshot commands, recovery instructions, known limitations and a first-hardware-run checklist.

Use clear binary versions rather than relying on a folder named for an older experiment. Log each rank's build and protocol identity during handshake. PowerShell scripts must derive paths safely inside the script body; do not repeat the old `$PSScriptRoot` default-parameter initialization problem. Quote Windows paths, propagate exit status, and test spaces in paths. No application install files belong under `C:\Strata`.

### 11.3 Configuration contract and launch examples

The names below are **proposed deliverable interfaces**, not existing commands or code to run during prompt review. Implement them or document a consistent equivalent, and test that every shipped example parses and matches actual supported flags.

| Setting | Default / required behavior |
| --- | --- |
| `--role` | `host`, `middle`, `terminal`; each owns real computation |
| `--stage-split` | `18,5,17`; planner regenerates dependencies for any accepted alternative |
| `--max-context` | `1048576` |
| `--kv-precision` | `fp16`; `bf16`/`native` only if actually implemented and explicitly selected |
| `--kv-allocation` | Eager full selected capacity |
| `--wire` | `fp32`; lower-precision modes optional and explicitly selected/tested |
| `--swa-replay` | `exact`; explicit `bounded` option with fidelity notice |
| `--cpu-isa` | `auto`; safe baseline/AVX2 and optional supported AVX-512 |
| Expert storage policy | A/C: bounded explicit SSD/RAM/VRAM tiers; B: resident-only |
| Cache budgets | Per-node explicit limits or preflight-derived limits with a reported ledger; never allocate by naive aggregate free-memory sum |
| Prefill chunk | Validated bounded size chosen from actual stage workspaces; do not copy Qwen's fixed 2048 without evaluation |
| `--spec` | `off` baseline; DSpark optional after validation with all effective settings reported |
| Vision | Included/supported; example for enabling it with explicit memory allowance |
| Sessions | Optional manual snapshots; enabled profile for testing vision and sessions together |
| Auth | Token/key file or another secure config mechanism; no hard-coded/default secret |
| API | Loopback 8081 requested profile; explicit 8082 coexistence profile |
| Worker endpoints | Proposed new ports 39011 for B, 39012 for C; do not reuse Qwen's 39001 |
| Logs | `info` default; opt-in `debug`/`trace` and component tracing |

Example target user workflow, after installing the reviewed candidate and copying the correct model files:

```powershell
# 5070 Ti: start terminal listener first.
cd C:\DeepSeekRPC
.\RUN-TERMINAL.bat --config .\config\terminal.json

# Laptop: start middle listener, connecting to terminal 192.168.50.143:39012.
cd C:\DeepSeekRPC
.\RUN-MIDDLE.bat --config .\config\middle.json

# 4090: local preflight, cluster self-test, then API startup.
cd C:\DeepSeekRPC
.\RUN-HOST.bat --config .\config\host.json --preflight
.\RUN-HOST.bat --config .\config\host.json --self-test
.\RUN-HOST.bat --config .\config\host.json
```

Specify actual lifecycle semantics so these commands do not accidentally launch duplicate loaded models. For example, worker listeners can admit diagnostic/control operations before serving and the host launcher can select a diagnostic-only mode or a persistent process mode. B/C load and verify once per intended process lifetime. Placeholder laptop/host IP fields must be obvious setup inputs and refused if left unresolved. Start C, B, then A; normal shutdown stops A admission first, drains/cancels safely, then stops workers.

### 11.4 Shipped first-run diagnostics

Provide two clearly separated levels:

1. **Small local self-test:** no full model required; device/driver/SM inventory; K2/K3 and actual additional quant-format tiny EXL3 operations against independent CPU goldens; relevant 5120/2304-shaped projections with manageable fixtures; attention/index/rotation/layout checks for selected precision; stream/event correctness; CPU ISA checks; Windows I/O/Engram synthetic fixtures. Test SM120 sparse and image-related shapes explicitly.
2. **Installed-model cluster preflight/probe:** validate full file/component identities, selective owned loading, eager allocation and B residency, all-rank handshake, source/shadow synchronization and a short deterministic text/image probe. Reference checks must use actual independent expected data where available. A plausible answer alone is not a parity test.

These self-tests will execute on the user's hardware only after reviews. In the CPU-only Codex environment record them as **implemented but not GPU-run**. Do not embed fabricated GPU golden outputs. If deterministic full-model goldens cannot be obtained within the resource constraints, state that limit and use independent CPU operation fixtures plus later bounded reference comparisons, rather than calling split-versus-itself a correctness oracle.

## 12. Internal milestones and test gates

Milestones may overlap, but every dependency has an explicit gate. Build the complete three-stage path throughout; do not turn intermediate harnesses into a reduced user deliverable.

| Milestone | Required outputs | CPU/build evidence before moving on |
| --- | --- | --- |
| M0: audit and design | Source/license manifest, compatibility report, tensor/allocation plan, protocol/state specification, blocker list | Plugin compatibility output; metadata/ownership checks; source-level FP16 and disk-Engram plan; three-node dependency and memory arithmetic checked |
| M1: Windows skeleton and reference fixtures | Real Windows transport/config/I/O build, independent tiny numerical fixtures, typed state machine | Windows target compilation; three processes over real sockets; bounded serialization, authentication, stale epoch and order tests; fixture provenance |
| M2: owned loading, caches and Engram | Selective EXL3 loader, bounded SSD/RAM/VRAM APIs, resident-only B, Windows exact-row Engram, layout adapters | Synthetic safetensors/K-map tests; large logical offsets without large allocations; short reads, eviction/lease races, no laptop reads; CPU parity; native code compiled for both SMs |
| M3: actual distributed graph | Real A/B/C CUDA execution wired to transport, source replicas, FP16 backend, exact CED/SWA, non-speculative decode | CPU graph/state fixtures against independent reference; odd/even chunk cases; same operations with/without transport; real CUDA paths compiled and reviewed, runtime GPU tests explicitly pending |
| M4: application features | Sampling, DSpark integration, vision, OpenCode parsing/streaming, reuse/cancel, bounded three-node snapshots | Distribution/rollback fixtures; client/tool/image fixtures; cancellation without unload; snapshot interruption and restore consistency; Windows launcher tests |
| M5: reviewed release candidate | Complete native Windows three-role package, configs, self-tests, diagnostics, audit reports and rollback archive | Required CI/build/CPU suites pass on exact candidate SHA; repeated independent review findings resolved/retested; no mandatory stub/placeholder/fake success |
| M6: user hardware qualification, later | First real SM89/SM120 and three-PC runs, progressive context/fidelity checks and measurements | Actual hardware logs supplied by user; fix/review/rebuild loop for failures; no early request for trial-and-error hardware testing |

Do not make the number of builds an artificial quota. Expect many iterations, possibly 20+, because the user values repeated review. Progress is judged by resolved findings and evidence. Internal source/build checkpoints are allowed to be incomplete; M5 must not hide that incompleteness under a “release” label.

### Required fidelity test families

- EXL3 reference reconstruction/execution, quant metadata and exact routing for actual K-map formats; per-expert output, accumulation and downstream logits.
- Exact 1017 Engram row/hash/scale behavior across chunk boundaries, new tool turns, odd compression positions and rejected drafts.
- Independent tiny CED/CSA2/mHC/reference graphs; all source/consumer boundaries, `pre_mix`, sparse candidates/top-k, source-to-shadow mapping and generations.
- Cache-hot/cold/eviction equivalence, delayed reads, failed reads, stale GPU slot pointers, asynchronous leases, no skipped expert under pressure.
- Exact versus bounded replay as separate expected behaviors; no implicit bounded default. Test any config-to-backend translation directly.
- Prefill chunk-size variation with identical logical context, changed images with equal token IDs/counts, prefix reuse after compaction or restore.
- Sampled and greedy generation, speculation off/on, every rejection length, RNG/position consistency, EOS/stop/output boundaries and cancellation at each transaction boundary.
- Snapshot save/restore with partial files, incompatible identities, lost worker, corrupted chunk, in-place restore failure and no second full-state RAM copy.
- Long-context addressing near 262K, 512K and 1M using bounded/sparse CPU fixtures; no integer overflow or false “1M passed” claim from geometry alone.

After hardware becomes available, progressively exercise 8K/32K/128K/256K/512K and near-1M context with output space reserved, plus realistic OpenCode coding tasks. Compare selected-precision, identical-checkpoint reference behavior. A full-model reference may run slowly through a controlled test harness; it need not require fitting the whole checkpoint into one GPU. Do not promise such a baseline exists before implementing/auditing it.

## 13. Requirements-to-tests traceability

Maintain this table in the project with concrete implementation symbols, test IDs and result links. Severity **F** means fidelity/silent-corruption blocker; **R** means core runtime/release blocker; **O** means optional optimization. Both F and R block the corresponding claimed capability; neither may be waived by a performance result. GPU checks remain pending until actually run.

| ID / requirement | Implementation module | Required CPU-only or build check | Eventual hardware check | Severity |
| --- | --- | --- | --- | --- |
| R01 Exact 1017 EXL3 2.9/components | Provenance/loader | Reject mixed revisions, missing tensors, bad K-map/checksums; selective-file tests | Full local manifest verification on A/B/C | F/R |
| R02 Three real owned stages | Planner/model/transport | Coverage/no-overlap; true three-process TCP; compiled production executor reachable | Trace actual layers on all three GPUs and compare results | R/F |
| R03 All top-6 plus shared experts | Router/executor/cache | Exact selection, bias/masks, missed/failed expert behavior | Expert outputs and logits with hot/cold cache | F |
| R04 B resident-only | Loader/allocator/storage | No inference read entry point; synthetic budget/lock refusal; Windows allocation API checks | Physical residency, no model hard-fault/pagefile I/O under worst accepted workloads | R/F |
| R05 Bounded desktop tiers | Cache/I/O | Lease/eviction/short-read/failure tests; byte/budget accounting | Bounded RAM/VRAM, correct results, measured SSD reads | R/F |
| R06 Exact disk Engram, A only | Engram/loader | Reference hash/row/scale/history fixtures; B/C load without tables | Actual October row parity, bounded A memory and disk behavior | F/R |
| R07 FP16 default and explicit alternatives | Attention/config | Requested/effective checks; unsupported-mode refusal; numerical/storage fixtures; backend compile | FP16 execution and parity on both SMs, including sparse/image shapes | F/R |
| R08 Eager real 1M capacity | Planner/cache/attention | Layout/bounds/overflow and allocation-lifetime tests; no silent context reduction | Eager capacity fits; progressive long-context inference with no truncation | R/F |
| R09 Sources/shadows and sparse metadata | Shared state/protocol | Independent delta/scatter, slot generation, index/candidate and order fixtures | State/logit comparisons across both real boundaries | F |
| R10 mHC/pre-mix and CED phases | Model adapter/scheduler | Four-stream and phase-aware reference tests; no exact-mode tail shortcut | Exact prefill/decode versus selected baseline | F |
| R11 Exact SWA default | Config/attention | Backend flag audit; exact/bounded differential fixtures and snapshot identity | Mode-specific reference checks at short/long context | F |
| R12 Correct sampled output | Sampling/API | Temperature/seed/distribution and output-boundary fixtures; no hidden controls | Temperature 0/1, deterministic settings and statistical behavior | F/R |
| R13 DSpark exact verification | Draft/state/coordinator | All accept/reject cases; rollback/RNG/Engram/shadow consistency | Off/on target-distribution checks and acceptance telemetry | F; disabled until valid |
| R14 Vision and image positions | Vision/model/API | Template/mask/position fixtures; changed-image prefix rejection; shape compilation | Real images on A with correct B/C paths, restore and SM120 tests | F/R |
| R15 Cancel without routine unload | Coordinator/API | Interrupt every phase, immediate next request, duplicate/stale messages | Cancel while generating/prefilling, reuse same loaded processes | R/F |
| R16 Exact prefix reuse | Session/model adapter | Token/image/format/epoch matching; CED-dependent replay; compaction tests | Warm continuation versus fresh replay | F/R |
| R17 Consistent bounded snapshots | Persistence/coordinator | All-stage prepare/seal/apply faults, checksum/bounds, no full-state clone | Near-capacity save/restore, memory peaks, post-restore output parity | F/R |
| R18 OpenCode/tools/reasoning/context | API/template/parser | October fixtures, DSML/SSE tool loops, output/context accounting | Actual Docker OpenCode session with images/tools/cancel | F/R |
| R19 Native Windows SM89/SM120 | Build/kernel dispatch | Reproducible CPU-only Windows compile/link/package; exact targets | Both architectures execute goldens without TDR/stream faults | R/F |
| R20 Optional ISA without INT8 change | CPU executor/dispatch | CPUID/XSTATE fallback and numerical fixtures; inspect CPU/GPU INT8 paths separately | Supported CPUs compare ISA results and throughput | O; F if enabled incorrectly |
| R21 Auth, framing and bounded resources | Transport/config | Bad identity/auth, length overflow, oversized frames, secret redaction | All three listeners confined/authenticated; bounded large transfers | R |
| R22 Preserve Qwen/install isolation | Packaging/launchers | Path/port checks, no C:\Strata writes or destructive scripts | Independent install and explicit port-conflict behavior | R |
| R23 Honest diagnostics/build provenance | Release/metrics | Manifest consistency, exact-SHA logs, skipped-test statuses, counter definitions | Actual phase timings and useful failure bundle | R |
| R24 Throughput goals | Scheduler/cache/benchmarks | Instrumentation and workload definitions; no invented numbers | Sustained fresh/warm prefill and decode, stage/IO/network analysis | O, not a quality waiver |

Add new regression rows for every substantive Astra finding. A test that compares two calls into the same flawed implementation is not an independent numerical oracle; label metamorphic/state-consistency tests separately from reference correctness tests.

## 14. Observability and performance investigation

Provide lightweight default logs and opt-in `--log-level error|warn|info|debug|trace`, component tracing such as `--trace-rpc`, `--trace-experts`, `--trace-cache`, `--trace-io`, `--trace-kv`, `--trace-cuda`, `--trace-vision`, `--trace-spec`, and a machine-readable metrics file. Exact names may change consistently. Redact content/secrets by default and bound log volume.

For each node and request, record:

- Build/kernel/SM identities, requested/effective modes, stage ownership, source/replica inventory, load time and allocation ledger/high-watermarks.
- GPU hits, RAM-served GPU misses, true SSD misses, evictions, speculative read-ahead and wasted/cancelled reads. Define denominators and distinguish expert requests from unique experts.
- SSD bytes and read count per generated token and prefill chunk, latency distributions/queue depth, intentional model I/O versus pagefile/hard faults, Engram row reads/cache hits.
- CPU/GPU expert work, PCIe transfer bytes/waits, dependency stalls, stage compute time, RPC bytes and timings by boundary/type, serialization overhead and backpressure.
- Fresh versus reused prompt lengths, exact replay work, TTFT, effective prefill rate, steady-state generation rate and total wall time. Exclude reused tokens from “new tokens processed” when reporting compute rate.
- DSpark proposed/verified/accepted tokens, correction/rollback events, and useful output tokens per target pass; no inflated rate counting rejected drafts.
- B residency/pressure and available thermal/power indicators; do not imply unavailable sensors are measured.

Use a break-even model rather than predicting speed from aggregate VRAM. For single-request decode, account for the dependency chain's compute, uncached disk stalls, PCIe transfers, network and synchronization; useful overlap must be demonstrated. More desktop SSD bandwidth does not automatically double token rate. For prefill, examine expert reuse within chunks, exact CED replay, source-state transfers and B workspace/power before increasing batch size.

Use matched workloads and unchanged model/precision/routing when comparing optimizations. The old Qwen measurements, Sybil's large-RAM setup, and Spark high-bandwidth results are context only. Do not import their scores or throughput as estimates for this system. The current task is engineering, not a requirement to fabricate benchmarks before GPU access exists.

## 15. Sol ↔ Astra review workflow, history and storage discipline

Astra is a separate reviewer. Do not claim to invoke Astra automatically, substitute an unrequested agent, or mark a self-review as independent approval. At each meaningful candidate checkpoint, create a review packet the user can pass to Astra. Continue independent authorized work while feedback is pending, but do not mark the candidate review-approved or ask for early hardware testing before the required reviews.

Each packet must contain:

1. Candidate/source SHA and tag, dependency lock/patch manifest, exact build artifacts/checksums and current verification statuses.
2. Concise changes and rationale, affected requirements, relevant file/symbol links, architectural decisions and changed defaults.
3. Build/CPU/Windows-TCP test commands and results; explicit skips, failures and GPU tests not run. Include logs for the exact candidate, not a nearby successful SHA.
4. Updated memory ledger, protocol schema/state changes, precision/dispatch matrix and supported-mode list.
5. Focused review questions for catastrophic runtime failures and any possible intelligence-affecting defect: incorrect routing, quant layout, Engram, attention, replicas, replay, vision, speculative acceptance, reuse or snapshots.
6. Known limitations/blockers, new numerical tolerances with justification, license/provenance changes, and first-run diagnostics status.
7. Previous finding IDs, dispositions, fixes, regression tests and remaining reviewer disagreements. Do not silently close an unresolved concern.

On feedback: reproduce or reason from evidence, fix substantive issues, add an appropriate regression test, rerun affected checks, self-review the full resulting change and submit an updated packet. A repair that changes numerical/state code requires focused fidelity re-review. Rare non-fidelity edge cases may be documented and deferred with rationale; build blockers, mandatory-feature omissions and silent intelligence-affecting failures may not.

Preserve source history in Git with meaningful commits and candidate tags. Keep an append-only review/build ledger and changelog; retain a last-known-good **build/CPU-tested** rollback point without implying it is GPU-validated. Create source archives at milestones and before major refactors. Do not require a full model load or checkpoint download for each review.

Respect the **actual Codex workspace resource and persistence limits**. Inspect and record free disk, any known quota, available RAM/CPU, permitted writable locations and retention/export facilities; do not assume either the former 100 GB limit or unlimited capacity:

- Keep source, tests, tiny fixtures, pinned manifests, review reports, compact logs and selected release archives.
- Avoid duplicate complete upstream trees/toolchains/venvs per experiment and unbounded CI artifact downloads.
- Use size-aware cleanup of regenerable caches and obsolete intermediate builds; never delete source history, reference pins, review ledgers or designated milestone/rollback archives.
- Set and report free-space headroom before large builds/downloads, based on the actual environment. Keep the same no-full-model policy: no 196.3 GiB checkpoint, Engram shards, or multi-hundred-GiB model backup in Codex.
- Do not assume personal-VM snapshots, background-process survival, or local Git history will persist across Codex tasks/environment replacement. Export source/review archives through available artifacts and use an explicitly authorized remote when available. A local commit alone is not an off-environment backup. Never push to a guessed destination.

Maintain a compact `HANDOFF.md` in the implementation project with current source SHA, resolved paths/pins, completed gates, open findings, next steps, exact build commands, artifact locations and CI run IDs. Update it at review/build boundaries so another Codex task can resume without repeating research. Exclude secrets. Use reproducible setup scripts after execution is authorized; do not rely on undocumented global installs or the personal VM's earlier blanket permission. Verify the new environment before resuming a saved task, and recover only from identified archives/remotes.

A CPU-only Codex environment does not imply that a native Windows runner, CUDA toolkit, persistent CI service or remote publishing access is available. Use existing authorized CPU-only Windows CI if available. If a new remote or CI destination is needed, prepare the workflow and concrete reviewable source first, then request only the missing publication/runner authorization. Do not publish to the existing StrataRPC fork by inheritance. No rented GPUs or external GPU runners. Lack of a Windows build runner means Windows compilation remains pending and is a real release blocker, not permission to label Linux compilation equivalent.

## 16. Known blockers and permitted contingencies

| Risk / blocker | Evidence to establish | Permitted response; forbidden shortcut |
| --- | --- | --- |
| Windows vLLM 0.31/plugin ABI mismatch | Checker output, installed imports, symbols and Windows build failures | Adapt narrowly and test; propose a source-backed foundation change if fundamental, obtaining approval before switching. Do not claim plugin installation proves compatibility. |
| Proposed split rejected by source-local KV checks | Exact attention guard and source-consumer mapping | Implement qualified replicas and metadata transfer; never just remove the guard or invent absent state. |
| True FP16 sparse KV/index path unavailable | Kernel layout/dispatch audit on both SMs and compile evidence | Implement a valid backend/adapter or report blocker/options. A clear startup error is necessary but is **not completion of requested default FP16 support**. Native/BF16 modes remain explicit alternatives, not a hidden waiver. |
| Laptop cannot physically hold accepted working set | Tensor ledger, lifetime peak, lock/WDDM constraints | Investigate a justified contiguous split/bounded exact schedule retaining three real stages; report if none fits. Never drop B, page it, reduce context or silently change precision. |
| Exact CED/SWA needs more replay/workspace than expected | Reference schedule and peak allocation analysis | Implement exact bounded scheduling/recomputation if valid; report costs. Bounded replay requires explicit user selection and cannot justify the default performance claim. |
| Engram backend attempts ~200 GB pinned allocation | Loader/lookup allocation trace | Implement bounded exact Windows disk rows on A; never use stock whole-table pinning, approximate rows or require worker Engram. |
| EXL3 fallback has wrong layout, unsupported shape or excessive scratch | Actual K/shape dispatch and numerical/workspace fixtures | Use a correct bounded baseline or implement missing kernels; do not silently switch quant or use extra INT8 activation math. |
| Cache slot/graph/stream lifetimes unsafe | Pointer ownership, events, capture and replay analysis | Disable optional capture/overlap initially and fix leases; never recycle live slots. |
| SM120 vision/attention or WDDM failures | Pinned issue/fix status, build evidence, eventual self-test logs | Correct dispatch/bounded launches; no hidden vision disable or global TDR-disable prerequisite. |
| SSD misses make speed targets unattainable | Real later miss traces and stage timing | Resize caches, tune exact scheduling/prefetch/split, explain measured bottleneck. No expert skipping or changed model. |
| Snapshot restore cannot fit a second copy | Lifecycle allocation ledger | Bounded in-place restore with cluster invalidation on failure, or another proven bounded design; do not remove RAM safeguards and rely on paging. |
| No compatible GPU available during development | Environment inventory | Build/CPU-test/review and ship pending-hardware status with self-tests. No invented CUDA pass or rented GPU requirement. |
| Required files/pins/licenses/CI unavailable | Concrete missing artifact/access and affected requirement | Complete unaffected authorized work; report the specific blocker and smallest needed input/action. Never fabricate a pin, license grant, test result or download. |

Do not ask the user to approve naming, module boundaries, standard error handling, ordinary tests, or other routine choices. Report a true blocker when meeting a fixed requirement is impossible on the audited stack, requires changing the fundamental architecture, or needs missing external authorization/resources. Explain the evidence, attempted compatible remedies, and concrete options without repeating questions already resolved by the handoff.

## 17. Release-candidate exit criteria and handover

Before offering the first hardware-test candidate, verify all of the following:

- The actual production entry points execute owned A/B/C stages and connect real transport; mocks are confined to tests. No mandatory runtime TODO/stub returns fabricated success.
- Required FP16/default exact mode, full 1M capacity architecture, desktop cache, strict B residency, disk Engram and shared-state plans are implemented and compiled. If any is blocked, report an incomplete candidate instead of claiming delivery.
- Vision, sampled decoding, OpenCode/API, cancellation, prefix reuse and optional snapshots are integrated with consistent config/capability negotiation. DSpark status is explicit and unsafe modes unavailable.
- Windows builds/package checks and relevant CPU/three-process Windows TCP tests pass on the delivered source SHA. Every skipped GPU check is named.
- Multiple meaningful Astra reviews and repair/re-review cycles have addressed release/fidelity blockers. Review evidence is included; no invented reviewer signoff.
- Three role-specific launch/config bundles, local model-file map, clear ports/auth, installation/rollback guide and small self-tests are complete. Existing StrataRPC paths and data are preserved.
- Allocation ledgers, effective precision telemetry, diagnostics, source/license manifest, build hashes and checksum files are packaged.
- There is a practical first-run plan with short diagnostics before full loading and progressive tests afterward. The user has not been required to test incomplete kernels throughout development.
- The package is labeled build-/CPU-integration-tested with hardware validation pending until real evidence changes that status.

When approved engineering work is complete, deliver source/archive location, candidate checksums/version, authorized build/download locations if publication was separately approved, concise status, known limitations, and the first-run instructions. Do not claim a release link exists before it does. If a mandatory capability is still blocked, deliver the evidence and usable review materials with an honest incomplete status instead.

## 18. Review flags for the user

These are the nontrivial assumptions and recommendations in this **draft**, for user/ChatGPT review before execution:

1. **Local source bundle missing here.** The draft uses the latest handoff's detailed audit/pins and prior Astra review; Sol must verify the actual local sources. The available StrataRPC protocol/docs were inspected read-only. No integrated dependency stack or GPU behavior was newly qualified.
2. **FP16 support is substantial work.** “Selectable and on by default” is interpreted as requiring a genuine working FP16 mode, not merely a flag that always errors. Clear refusal is the failure behavior; it is not acceptance of an unfinished mandatory default. BF16/native alternatives are explicit selections.
3. **18/5/17 remains conditional.** Its five 2-bit laptop layers are promising, but true tensor allocations, full exact-mode prefill peaks, physical locking and WDDM behavior remain unmeasured. Configurable splits must retain all three stages and host-only Engram under current requirements.
4. **Exact replay can change the performance picture.** The draft explicitly disallows treating the 128-row bounded decoder shortcut or its reported speedup as default exact behavior. Full-context intermediate activation storage also requires a real bounded schedule, not just a small KV estimate.
5. **Snapshot storage recommendation.** B/C state streams to a host-owned bundle by default, using bounded staging. This differs from StrataRPC's worker-local paired files, adds network save/restore time, and avoids requiring laptop snapshot storage. It is a recommended design choice, not an already implemented or benchmarked feature.
6. **API port reconciliation.** The requested 8081 endpoint is retained for exclusive use after the user stops Qwen. An explicit 8082 profile supports coexistence. New worker ports are proposed as 39011/39012. No existing service is stopped or config overwritten automatically.
7. **Residency claims need actual API guarantees.** The draft requires more than mmap/preload or post-hoc hard-fault monitoring. If Windows cannot provide the required model residency within B's budget, Sol must report a blocker. Incidental OS/logging I/O is distinct from prohibited inference model reads.
8. **DSpark and AVX-512 qualification.** DSpark is implemented behind a mode and remains off until correctness gates pass, as permitted by the handoff; optional AVX-512 does not block the baseline. Neither can silently enable an INT8 or approximate acceptance path.
9. **Independent reviews and publication are separate handoffs.** Astra reviews are user-mediated; Sol does not fabricate or automatically invoke them. A new GitHub destination/CI use/push needs explicit authorization if not already supplied for the new project. This draft authorizes none of those actions.
10. **Development environment changed by the user.** Sol now runs on Codex-managed CPU-only computers. The former personal-VM 100 GB limit, home-directory project path, preloaded references and snapshot assumptions are superseded. Sol must inspect actual limits, use a separate permitted project checkout, and export reproducible handoff/review artifacts. Native Windows builds still require an available Windows build environment or authorized CPU-only CI; Codex itself supplies no implied GPU validation.
11. **Code completeness is not hardware proof.** The first candidate contains the full real three-PC implementation, but no amount of CPU testing or review proves it runs correctly on all three GPUs. Actual GPU/1M/fidelity/performance status stays pending until the user's later trials. No speed guarantee is implied.

## 19. Concluding instruction to Sol — applies only after explicit execution approval

Once the user separately approves this prompt and tells you to begin, inspect the CPU-only Codex environment and any supplied pinned reference bundle, choose an isolated new-project directory within its permitted workspace, and produce the compatibility, memory and protocol artifacts. Then implement, test, build and iterate toward the complete three-PC Windows candidate under the gates above. Continue routine authorized engineering without unnecessary confirmation requests. Preserve the user's settled model/platform/fidelity requirements and working StrataRPC installation. At genuine blockers, report concrete evidence and options; never substitute hidden approximations, missing stages or invented validation.

**Until that explicit approval, this document is a review draft only. Do not initiate implementation from its presence in a conversation or workspace.**
