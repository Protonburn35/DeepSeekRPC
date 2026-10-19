# DeepSeekRPC protocol 1.0

Status: engineering specification. Implemented capabilities and test evidence
are recorded in HANDOFF.md; the complete model path is not yet qualified.

## Connections and authentication

Use persistent native TCP sockets with TLS 1.3 and mutual certificate validation
through Python's OpenSSL-backed ssl library. Deploy a private CA and distinct
certificates for host, middle and terminal; no default key or shared secret is
shipped. A certificate's DNS SAN must identify its stage as
`host.deepseekrpc`, `middle.deepseekrpc`, or `terminal.deepseekrpc`. Outgoing
connections verify the expected peer name. Listeners require trusted client
certificates and check the expected stage SAN as well. HELLO must agree with
the authenticated identity and configured run fingerprint. Key file paths are
configuration, not key values. Keep keys outside source/artifacts and redact
credentials from diagnostics.

Each topology edge has separate data and control TLS connections. Bulk frames
cannot occupy the cancellation/heartbeat connection. Pair channels using run,
session, peer identity and epoch; never accept a cross-run channel. Bind worker
listeners to explicit configured LAN addresses, not all interfaces. Tests use
loopback with ephemeral certificates and ports. This does not authorize a public
inference endpoint.

## Framing

Integers use network byte order. The 16-byte prefix is `!4sBBHII`:

| Field | Encoding | Rule |
| --- | --- | --- |
| Magic | 4 bytes | `DSRP` |
| Major, minor | two uint8 | 1, 0; reject unsupported versions |
| Kind | uint16 | enum table below; reject unknown values |
| Envelope length | uint32 | 1..16384 bytes |
| Payload length | uint32 | 0..1048576 bytes |

Check both lengths before reading/allocating variable buffers. The envelope is
strict UTF-8 JSON with duplicate keys and nonfinite numbers rejected, bounded
to 16 KiB. It contains `run` (SHA-256), `sender`/`receiver` (0/1/2), `session`
and `request` (128-bit lower-case hex), `epoch`, `operation`, `sequence`,
`transaction` (128-bit lower-case hex), `position`, `rows`, `slot_generation`,
`payload_sha256`, and bounded typed tensor descriptors. Control messages may
have no tensors. Each operation binds identity and SHA-256 of its semantic
inputs; same operation plus different data is an error. Frame integrity is
SHA-256 over payload inside authenticated TLS, not a home-grown authentication
scheme. Error messages carry bounded codes and never arbitrary exception dumps.

Descriptors identify field semantic, source layer, dtype, shape, logical row
origin, tensor identity/total length and fragment offset. Supported storage
dtype IDs are FP16, BF16, FP32, I32, I64 and U8. Activations/pre_mix require
FP32 (the launch setting is --layer-stage-wire fp32). Token/position/index values
stay integers; KV/index rows require unquantized BF16 storage. Validate rank, dimensions, element-byte multiplication, fragment
alignment and all intervals with checked bounds. A frame payload is at most
1 MiB; consumers stream fragments to a previously budgeted destination. Do not
buffer an entire prompt or allocate a shape announced by the peer. Reject
overlapping, missing, out-of-order or generation-incompatible fragments.

Socket deadlines cover a whole frame, not a fresh timeout for every byte.
Bound descriptor count, outstanding operations, replay history, worker queues,
serialization and staging. A failed send/receive makes that channel unusable.
Do not continue a partially received operation or infer a remote commit outcome.

## Message families

| IDs | Names | Semantics |
| --- | --- | --- |
| 1–4 | HELLO, CAPABILITIES, PLAN_ACK, READY | Authenticated stage, run/model/component/ABI identities, ownership, source maps, formats, capacity and memory admission |
| 10–13 | BEGIN, PREFIX_PROBE, PREFIX_ACCEPT, RESET | Unanimous session identity, accepted prefix and state generations |
| 20–23 | STEP_BEGIN, SOURCE_DELTA, STAGE_INPUT, STAGE_DONE | Exact CED phase, positions, activations/pre_mix, token/image metadata, source deltas and dependency completion |
| 30–34 | DRAFT, VERIFY_RESULT, COMMIT, COMMIT_ACK, ABORT | Proposal, exact target result, accepted prefix, correction/bonus semantics, tentative writes and all-rank acknowledgement |
| 40–43 | CANCEL, CANCEL_ACK, HEARTBEAT, ERROR | Safe-boundary cancellation, bounded health and typed recoverability |
| 50–55 | SNAPSHOT_PREPARE, SNAPSHOT_CHUNK, SNAPSHOT_SEAL, RESTORE_PREPARE, RESTORE_APPLY, RESTORE_ACK | Quiescent three-stage snapshot transaction and bounded authenticated streams |

Enumeration does not imply that every family already has a production handler.
Capabilities must enumerate only actually implemented/qualified operations.

## Ownership and ordering

A coordinates, computation flows A→B→C, and C returns proposed results to A.
Every target verification traverses all three stages, including A's Engram.
From config, default A→B supplies source-14 KV/index K and index-14 top-k;
B→C supplies source-20 KV/index K, index-20 top-k and candidates. Transfer
token IDs, absolute positions, image/validity metadata and compression phases
as well as `[rows,4,5120]` FP32 hidden and `[rows,4]` pre_mix. Add graph-required
state discovered in the pinned source audit; this table is not a waiver.

Shadows have independent physical slots. A delta identifies source layer,
logical row, transaction, slot-map generation, format and validity. The receiver
validates and scatters to its local mapping before any consumer executes.
Half-rate source row production follows absolute position parity across chunk
boundaries. Speculative rollback includes overwritten SWA/compression rows,
index/candidate state, Engram lookback and RNG/draft effects.

## State machine and transactions

BOOT/PREFLIGHT→CONNECTING→IDLE; IDLE admits PREFILL or DECODE/VERIFY,
SAVING or RESTORING. CANCELLING returns to coherent IDLE only after all ranks
abort or finish the agreed window. Unknown commit outcome→FAULTED_SESSION;
device failure→FAILED_DEVICE. Healthy model allocations remain loaded during
ordinary cancellation and session invalidation.

Each operation has an immutable identity, tentative state, and a single accepted
prefix. All ranks ACK the same COMMIT before output is emitted. COMMIT replay is
accepted only if identity, payload digest and accepted prefix match the cached
result. Stale epochs and conflicting duplicates fail. Replay history is bounded;
an evicted operation cannot be inferred safe to replay. Process-crash exactly-once
semantics are not claimed.

Track token history, per-source KV validity, emitted output cursor, and the
sampled token not yet forwarded separately. Empty/partial/full draft acceptance,
correction and bonus tokens, EOS, stops and output exhaustion have distinct
transitions. DSpark remains off until the full acceptance/rollback path passes
its independent fidelity gates. Temperature-1 verification must preserve the
target distribution; a confidence threshold is insufficient.

Prefix reuse binds tokens, image content and positions, complete numerical and
component identity, cache generations, Engram/SWA/draft state and CED replay
semantics. Equal token counts or matching tokens with changed images cannot
establish a hit. Reset/replay on mismatch. Exact mode explicitly disables the
upstream bounded-replay optimization.

## Snapshots and recovery

Only idle clusters save. Use one snapshot ID, per-stage inventories, lengths,
hashes, timelines, pending-token/RNG semantics and numerical/session identity.
B/C stream bounded chunks to host storage; A writes its local state. Seal a
single manifest atomically only after every stream verifies. Never include
weights or expert payload caches. Client chat/images remain client-owned.

Validate the complete manifest, paths, bounds, compatibility and checksums
before mutating live state. In-place restore uses bounded staging; no second
complete state copy is assumed. Partial apply invalidates all ranks. Reestablish
B residency before READY. Opaque IDs and generated stage filenames prevent peer
path traversal. A disconnect with uncertain state requires coordinated reset or
validated restore, not guessed positions.
