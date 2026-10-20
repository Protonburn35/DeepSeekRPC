# Diagnostics and qualification boundaries

`serve --preflight` checks small metadata and TLS files without constructing a
model or allocating payloads. `serve --self-test` adds the installed runtime,
GPU/SM inventory and bounded local checks. `selftest-cuda` runs those operation
checks without requiring a role configuration or model directory. Normal model
startup repeats required local diagnostics before full allocation.

The actual local CUDA checks cover BF16 sparse attention/index/Engram/scatter,
all EXL3 physical K values and exact trellis/fragment reconstruction, native
rotations and 5120/2304 projections, four-stream mHC, text/image-aware target and
draft routers, maximum fused vision attention shape, odd compressor closures,
independent prefix maps and ring rollback at every accepted length 0..6. CPU
equations are independent of the tested CUDA implementation. Packed projection
error bounds include prescribed FP16 operand rounding and FP32 dot/reduction
error; this does not introduce an FP16 KV mode. Windows synthetic checks cover
a sparse file offset above 4 GiB, exact Engram reads/cache and private locked
host pages. CPU feature reports use Windows OS-enabled feature APIs.

The local checks and installed-cluster probes are implemented but NOT GPU-RUN
in the CPU-only development environment. Compilation/import and mathematical
CPU fixtures are separate evidence. No full-model numerical golden or measured
throughput is available yet.

The installer separately qualifies the real bundled TinyCC extension build/load
and offline Triton SM89/SM120 compilation with GPUs hidden, bare system PATH and
inherited SDK/compiler selectors removed. The exact Triton source/binary and
standalone Python development inputs are pinned in the packaged TRITON.json.
Stage identity also binds the actual Triton compiler components. Kernel source
overrides, forced architecture and CPU interpreter fallback are refused. This
compiler check creates no GPU context and makes no GPU execution claim.

After the planned reviews, start C/B/A normally and wait for all-stage READY.
Run the installed-model probe against that existing process:

```powershell
.\runtime\Scripts\python.exe -I -m deepseekrpc.cli selftest-cluster
```

Add `--api-key-file .\credentials\api.key` if A uses API authentication. This
command does not launch or reload a model. It reserves the actual cluster and
creates synthetic diagnostic sessions, so finish a retained paused user session
first. If it needs preservation, save it, then finish a separate short chat
before probing; restore the saved session afterward. Existing saved snapshots
are not overwritten. Diagnostic snapshots use fresh opaque IDs and are removed
after their comparison. Interrupt/disconnect cancels the owned operation and
retains loaded weights while uncertain sessions/meshes recover.

The cluster probe performs actual text/image CED, normal/prefix-reused target
repeatability, greedy target-versus-DSpark token parity, sampled DSpark repeat/RNG
consistency, sampled logical BF16 source/shadow comparison, eager-pool checks,
B all-page weight residency and actual pause/save/restore/continuation. Its
fixed-length internal token probes ignore EOS to exercise verification windows;
separate deterministic verifier fixtures check EOS and output boundaries. It
does not judge the generated answer's plausibility or call split-versus-itself
an independent correctness oracle. It returns actual findings and stage budget
counters as bounded JSON without user paths, prompts, images or secrets.

Speculation defaults off. Explicit `speculative: true` requests fail clearly
until this actual loaded-cluster probe passes. The successful proof exists only
in the process and binds the exact model/component identity and mesh incarnation;
editing a saved report cannot enable it. `/readyz` reports qualification. A new
process or replacement mesh requires qualification again. Qualify a new process
before restoring a paused speculative snapshot that you intend to resume.
The implemented feature remains available for review without claiming that its
unrun hardware gate has passed.

The open Linux Torch 2.13 CPU sqrt discrepancy is DSR-AUDIT-011 in FINDINGS.md.
It is preserved as a failure, with frozen oracle and unchanged tolerances.
Windows CPU, Torch 2.11 CUDA runtime and actual GPU evidence are distinct.
