# Requirement traceability

Engineering status only. A protocol enum, arithmetic estimate or CPU simulation
does not complete a model capability. F/R gates from IMPLEMENTATION_SPEC.md are
retained. All full-model and GPU checks are NOT RUN; performance is NOT MEASURED.

| ID | Current implementation/evidence | Remaining gate |
| --- | --- | --- |
| R01 Exact 1017 | Complete inventory, publisher-verified selective loader and actual owned graph/bootstrap source | Actual GPU constructors/loading and full-model qualification |
| R02 Three real stages | StageExecution, typed phase ledger and TCP/TLS control/data components; separate affine three-process fixture | Production service/coordinator integration and real three-PC graph |
| R03 Top-6/shared | Bounded actual EXL3 executor retains every choice, official pre-w2 weighting and actual shared expert path | GPU routing/tie/numerical qualification |
| R04 Resident laptop | VirtualLocked preload, all-page Win32 residency tests, closed disk source and one shared MiB pinned transfer | Full 16.4 GiB preload, WDDM/device working set and worst-case hardware proof |
| R05 Desktop tiers | Bounded RAM reads, permanent packed arena, GPU expert arena and event/lease ownership | Complete service integration, prefetch performance and measured memory peaks |
| R06 Disk Engram | Original encoded rows/hash history, real learned projection/gate adapter and bounded pinned/native decode | Original image pipeline, CUDA numerical and large-file full-model qualification |
| R07 Unquantized BF16 caches | BF16-only planner/wire/native operations/actual adapter source; both architectures compiled | Actual GPU model/backend/replica execution; no quantized fallback |
| R08 Eager 1M | Whole-stage header budget, eager fixed GPU pools and Windows CUDA-LUID/DXGI counters | Complete cluster schedule, actual free-memory admission and hardware validation |
| R09 Replicas | Independent GPU maps/scatter, exact closure exports/imports and source selection validation | Production transfer ordering, all-stage barriers and GPU qualification |
| R10 mHC/CED | Actual delayed mHC and graph, complete-pass phase ledger/worker and rewind preserving full encoder sources | Cluster coordinator, exact long-context replay and GPU differential checks |
| R11 Exact SWA | Adapter explicitly disables bounded replay and uses 128-row windows with prefix undo | Actual exact/bounded differential hardware fixtures |
| R12 Sampling | Requested probability controls, separate output penalties, transactional RNG, exact correction/EOS/length and frozen-prefix grammar callback | Production API/coordinator integration and stop-string handling |
| R13 DSpark | Actual three-block graph, target-context features, Markov calls and Stage C sampled verifier source | Full cluster acceptance/rollback, output publication and GPU qualification |
| R14 Vision | Owned actual tower/aligner/native weights and image-row merge source; original encoding fixtures | Original preprocessing/cache/API and SM120 execution |
| R15 Cancel | Real TLS control cancellation during queued work, per-layer/expert/fragment observers and session invalidation | Production API wiring, every-phase cancellation and hardware recovery |
| R16 Prefix reuse | Exact original token/image/component history identity and explicit full replay | Saved CED-aware prefix checkpoints and integration |
| R17 Snapshots | Protocol/design only | Bounded three-stage save/seal/restore and fault tests |
| R18 OpenCode | Original October encoding, pinned tokenizer and frozen upstream fixtures | Streaming API, tool/reasoning parser/loops and client/context qualification |
| R19 Windows/SM89/SM120 | Windows CPU CI, project ABI 2 and full EXLlama native builds pass; vLLM build in progress | Full runtime dependency bundle, complete vLLM build and actual GPU execution |
| R20 ISA/no INT8 | Before-import EXL3 policy, exact CUDA auxiliary math and actual packed adapter with shortcuts disabled | Packed GPU differential/dispatch qualification; optional ISA work later |
| R21 Auth/bounds | Mutual TLS SAN/run/plane, bounded data/control queues/RPC, deadlines, errors, mesh arrival-order tests | Complete production dispatch/admission and certificate/port setup |
| R22 Preserve Qwen | Existing repo, independent paths and no changes to working installation | Windows launcher/package isolation and port-conflict qualification |
| R23 Honest provenance | Locks, native reports, component identities and explicit incomplete/unrun status | Complete candidate archive, diagnostics and repeated independent Astra cycles |
| R24 Speed goals | Header admission arithmetic and bounded execution source | Matched measured full-model workloads, no throughput promise |
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
