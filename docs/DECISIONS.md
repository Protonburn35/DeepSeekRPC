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
   full eager 1M capacity, default FP16/index and exact SWA. Earlier model shopping,
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
