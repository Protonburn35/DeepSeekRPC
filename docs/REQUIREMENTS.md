# Requirement traceability

Engineering status only. A protocol enum, arithmetic estimate or CPU simulation
does not complete a model capability. F/R gates from IMPLEMENTATION_SPEC.md are
retained. All full-model and GPU checks are NOT RUN; performance is NOT MEASURED.

| ID | Current implementation/evidence | Remaining gate |
| --- | --- | --- |
| R01 Exact 1017 | Source/model locks, complete header inventory; test_actual_source_map_and_replica_bytes, selective/malformed file tests | Real owned loader, full local payload integrity, packed marker/rotation execution |
| R02 Three real stages | TCP/TLS/fragments and test_three_real_processes_fragmented_tcp_commit_cancel_resume (test-only affine computation) | Actual owned CUDA model stages; compiled production entry points |
| R03 Top-6/shared | CPU route/expert reference compared to extracted official methods; asymmetric clamp and actual-width tests | Quantized CUDA executor, exact GPU tie/routing behavior, all selected expert accumulation |
| R04 Resident laptop | ResidentStore structurally has no inference loader; Win32 LockedBuffer refuses failed locks | Full preload, model/device working-set and WDDM admission; native Windows and worst-case hardware proof |
| R05 Desktop tiers | RAMCache reservations/leases/in-flight dedup/failure tests | Native async SSD policy, GPU tier/events, full runtime memory ledger and integration |
| R06 Disk Engram | Exact encoded row/scale reads; FP8 decode against CPU Torch; pinned tokenizer/compressed map and original hash operation match upstream frozen odd-chunk/image-mask fixtures with partial acceptance/reset | Learned project/gate integration, real image pipeline, full graph and Windows large-file qualification |
| R07 FP16 default | Source audit identifies unsupported upstream paths; no fallback advertised | Implement/compile/test actual FP16 sparse KV and index backend on both SMs |
| R08 Eager 1M | Derived source/shadow byte plan; bounded 64-bit file offsets | Real eager GPU pools and bounded exact schedule, complete lifetime/metadata budget and hardware validation |
| R09 Replicas | Derived dependency graph; local slot generation/journal checks | Actual deltas, independent GPU slot scatter and candidate/top-k ordering |
| R10 mHC/CED | Four-stream hc_pre/hc_post against official methods | Complete exact phase scheduler, graph and decoder-state replay |
| R11 Exact SWA | Upstream bounded default identified in compatibility report | Effective adapter setting and exact/bounded differential fixtures |
| R12 Sampling | Not implemented | Requested controls, RNG/distribution/EOS/output boundaries |
| R13 DSpark | Partial/full/zero prefix journal rollback tests are state checks only | Draft execution, target probabilities, exact sampled verifier and all-rank rollback; remains off |
| R14 Vision | Official encoding and upstream image marker cases | Actual tower/aligner and downstream mask/position/SM120 execution |
| R15 Cancel | Three real processes cancel and immediately resume; persistent fixture identities | API cancellation wired through actual model and every phase/window |
| R16 Prefix reuse | Not implemented | Full token/image/component/state identity and CED-aware replay |
| R17 Snapshots | Protocol/design only | Bounded three-stage save/seal/restore and fault tests |
| R18 OpenCode | Official October encoding + frozen upstream cases | Streaming API, tool/reasoning loops, client adapter/version and context accounting |
| R19 Windows/SM89/SM120 | First Windows CI passed 129 tests including real Win32 lock and sparse-file cases, wheel build/install; installed CLI path correction awaiting repeat | Complete Windows check, build actual runtime and both CUDA targets; then hardware execution |
| R20 ISA/no INT8 | No added activation INT8 or optional AVX-512 path enabled | Native baseline/dispatch qualification; optional ISA optimization later |
| R21 Auth/bounds | Mutual TLS stage SAN/run/plane checks; length overflow, digest, deadline, bad-auth tests | Full coordinator admission/queues, control priority during model execution, real deployment cert/port setup |
| R22 Preserve Qwen | Existing repo and distinct project paths; no original installation changes | Future launcher/config/package isolation and port conflict tests |
| R23 Honest provenance | Capabilities command, manifest and CI report explicitly classify incomplete/unrun statuses | Exact candidate source/build/review archive, real runtime diagnostics |
| R24 Speed goals | Payload arithmetic only | Matched measured full-model workloads, no throughput promise |

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
