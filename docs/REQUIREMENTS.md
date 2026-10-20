# Requirement traceability

Engineering status only. A protocol enum, arithmetic estimate or CPU simulation
does not complete a model capability. F/R gates from IMPLEMENTATION_SPEC.md are
retained. All full-model and GPU checks are NOT RUN; performance is NOT MEASURED.

| ID | Current implementation/evidence | Remaining gate |
| --- | --- | --- |
| R01 Exact 1017 | Complete inventory, publisher-verified selective loader and actual owned graph/bootstrap source | Actual GPU constructors/loading and full-model qualification |
| R02 Three real stages | Actual stage actor/service, dependency mesh, full-prompt TLS transfer, CED/speculative coordinator and persistent Windows startup; real TLS incarnation recovery checks | Real three-PC graph execution and hardware recovery |
| R03 Top-6/shared | Bounded actual EXL3 executor retains every choice, official pre-w2 weighting and actual shared expert path | GPU routing/tie/numerical qualification |
| R04 Resident laptop | VirtualLocked preload, all-page Win32 residency tests, closed disk source and one shared MiB pinned transfer | Full 16.4 GiB preload, WDDM/device working set and worst-case hardware proof |
| R05 Desktop tiers | Bounded RAM reads, permanent packed arena, GPU expert arena and event/lease ownership integrated in owned stage loading/execution | Actual GPU stale-transfer/lease checks, cache performance and measured memory peaks; read-ahead is not issued |
| R06 Disk Engram | Original encoded rows/hash history, learned projection/gate, bounded pinned/native decode and synthetic >4 GiB offset host-check source | Actual host/GPU numerical and original text/image full-model qualification |
| R07 Unquantized BF16 caches | BF16-only planner/wire/native operations/actual adapter source; both architectures compiled | Actual GPU model/backend/replica execution; no quantized fallback |
| R08 Eager 1M | Whole-stage header budget, eager fixed GPU pools, complete cluster schedule and Windows CUDA-LUID/DXGI counters | Actual free-memory admission and full-size hardware validation |
| R09 Replicas | Independent GPU maps/scatter, exact closures/selections, derived mesh ordering and all-stage GPU/history barriers | Actual GPU/network execution and differential qualification |
| R10 mHC/CED | Actual delayed mHC/graph, full two-pass cluster scheduler and rewind preserving all encoder sources | Exact long-context replay and GPU differential checks |
| R11 Exact SWA | Adapter explicitly disables bounded replay, uses 128-row windows with prefix undo and always traverses the complete target prompt | Actual full-prefix replay/decode differential hardware fixtures; no bounded mode is substituted |
| R12 Sampling | Probability controls, separate output penalties, transactional RNG, exact correction/EOS/length and token-aligned stops before commit; unsupported optional grammar controls rejected | Actual GPU/model sampling qualification |
| R13 DSpark | Actual three-block graph, target-context/Markov sampled verifier, API coordinator/publication and explicit context-tail normal decode source | Full cluster acceptance/rollback and GPU qualification |
| R14 Vision | Actual tower/aligner, original preprocessing, exact spans and bounded original/feature storage connected to original chat preparation and two-pass resolver | SM89/SM120 and full-model/image execution |
| R15 Cancel | API request ownership, scoped cancellation, per-layer/expert/fragment observers, persistent mesh recovery and real TLS interruption checks | Every-phase full-model cancellation and hardware recovery |
| R16 Prefix reuse | Exact token/image/component negotiation, sealed encoder SWA/pair checkpoint, retained source generations and required full target replay | Actual GPU fresh-versus-reused differential qualification |
| R17 Snapshots | Bounded state-only archives, all-stage save/seal, two-pass prevalidated restore, exact RNG/stop state, administrative pause/continuation and HTTP/CLI integration source | Actual GPU/distributed save/restore/fault/continuation qualification |
| R18 OpenCode | Original encoding/tokenizer, real preserving parser audit, bounded OpenAI JSON/SSE, separate profile and actual pinned SDK text/image/tool/reasoning/cancel audits | Actual Docker client/tool/image/context qualification |
| R19 Windows/SM89/SM120 | Windows CPU CI, project ABI 2 and full EXLlama native builds pass; vLLM build in progress; private installer, hash-locked whole dependency bundle and actual compiler checks have source | Complete vLLM build, private full-runtime/compiler install qualification and actual GPU execution |
| R20 ISA/no INT8 | Before-import EXL3 policy, exact CUDA auxiliary math and actual packed adapter with shortcuts disabled | Packed GPU differential/dispatch qualification; optional ISA work later |
| R21 Auth/bounds | Mutual TLS SAN/run/plane, bounded data/control queues/RPC, actual stage/API admission, deadlines/recovery, private ACL certificate generation and exclusive listener binding | Actual installed three-PC startup/fault qualification |
| R22 Preserve Qwen | Existing repo, independent paths/profile, private Python/no PATH registration changes, explicit port-conflict refusal and no changes to working installation | Actual complete bundle and user's existing Windows/OpenCode coexistence qualification |
| R23 Honest provenance | Locks, native reports, component identities, bounded content-free logs/archives and live public diagnostics | Complete candidate archive, full runtime installation and repeated independent Astra cycles |
| R24 Speed goals | Actual tier/I/O/transfer/window/request counters and timings plus matched real-API benchmark mode | Matched measured full-model workloads, no throughput promise |
## Oracle provenance and tolerances

The official October encoding file is vendored unchanged, with SHA-256 and MIT
license in its package directory. Its upstream tests only change imports and
fixture location; missing source example files are acquired at the same pinned
revision, not regenerated to make tests pass.

tests/oracles/official_methods.py extracts Gate.forward, Expert.forward and
Block.hc_pre/hc_post from the pinned official model source, retaining their
method bodies. Unquantized float32 router linear is bound to F.linear. This
avoids importing GPU-only module initialization and does not pretend to execute
the quantized model. Exact expert IDs are required on the fixed non-tied routing
fixture. GPU tie semantics remain unqualified; CPU reference ties use lower IDs.

Router/mHC comparisons use FP32 BLAS/reduction tolerances around 1e-6 absolute
and 2e-6 relative. The actual-width unquantized expert baseline uses 2e-6 absolute
and 1e-4 relative to accommodate differing FP32 accumulation orders across the
5120/2304 projections; this is not an EXL3 or full-model tolerance. Quantized GPU
tolerances require separate justification and independent review.

All 65536 mul1 codebook windows compare binary16 bits exactly against the pinned
plugin's standalone scalar reference. This checks procedural codebook values,
not trellis layout/rotations/GEMM. Encoded Engram rows compare exactly; FP8/UE8M0
FP32 diagnostic decode compares exact bits with CPU Torch. Cache/journal/TCP
checks are state/metamorphic/integration tests, not independent model oracles.
