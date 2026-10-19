# Decisions and source precedence

1. Current user instructions override earlier attachment authorization language.
   Implementation is authorized; reuse Protonburn35/DeepSeekRPC. Do not create
   another project repository or worktree.
2. Astra's IMPLEMENTATION_SPEC.md controls conflicts with the older technical
   handoff. The handoff supplied afterward has SHA-256
   `27d2689c2b02fceba7419af1b5f0eff9edb112d3d7431d86802dc400a6c5240a`, exactly
   matching the input hash recorded by Astra's prompt. It adds historical
   context, not a different approved architecture.
3. Use the actual Codex CPU workspace. Disregard personal Google Cloud VM access,
   home paths, preloaded bundles, provider snapshots and fixed 100 GB assumptions.
4. Use the settled October 1017 EXL3 2.9 model, all three native Windows stages,
   full eager 1M capacity, the BF16 cache policy in item 12 and exact SWA. Earlier model shopping,
   optional laptop, two-PC-first and bounded-replay-default proposals are obsolete.
5. No full model/Engram download into Codex. Source-only archives and strictly
   bounded header/metadata reads are permitted and recorded.
6. Additional handoff repositories (SM89 vLLM fork, TensorFold/Spark recipes,
   Xolotl, Abyss, demand-I/O GGUF recipe) are optional technical references.
   They do not silently become new runtime foundations or alternate quants.
7. Independent Astra review remains a user-mediated handoff. Do not launch an
   unrequested agent or record self-review as independent approval.
8. The user explicitly authorized the first push/draft PR/CPU CI run, then
   granted ongoing repository push and CI authorization. Future CI builds are
   Windows-only at the user's request. Native Windows remains the deployment
   target for all three PCs; cloud CPU source checks are development evidence.
9. Keep the user's personal identity private. New commits use the neutral
   DeepSeekRPC Automation identity. Do not include personal names or email
   addresses in generated commits, PR text, documents or build metadata.
10. The user authorized merges and direct commits to main, and asked for the
    Astra handoff after the full application is implemented. Continue internal
    checks and development checkpoints; do not pause for intermediate reviews.
11. All deployment PCs run Windows 11. Stage C is the RTX 5070 Ti with a
    Ryzen 7 9800X3D. "Terminal" is an internal last-stage protocol role, not an
    operating-system choice. User-facing launch instructions identify stages A/B/C.
12. The user clarified that avoiding quantized KV storage is the actual cache
    requirement. Use BF16-only KV/index caches; drop the optional FP16 cache mode
    and prohibit FP8/FP4/INT8 or automatic quantized-cache fallback. This explicit
    instruction supersedes the original prompt's FP16 default/native alternatives.
    Existing FP16 EXL3 weight-kernel inputs/rotations are a separate format.
13. The user reaffirmed --layer-stage-wire fp32 as the default. Stage activations
    and mHC pre_mix require FP32 throughout the RPC boundary. BF16 KV/index
    storage and lossless integer metadata remain separate typed semantics.
