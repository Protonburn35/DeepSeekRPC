# DeepSeekRPC whole-source implementation handoff

The complete native Windows/CUDA application source is implemented. Native
Windows builds/imports and the complete private-runtime installation passed
at source 30b24d3. The final review packet must bind matching CPU/install
reports to its exact frozen source. Independent review and GPU/model execution
remain pending. The user wants whole-source Astra review/fix cycles before
testing on the three PCs; this is not a hardware-test release.

## Requirements and authorization

Reuse Protonburn35/DeepSeekRPC on main. Pushes, Windows-only CI, direct commits
and merges are authorized. Use only the neutral DeepSeekRPC Automation author
and keep the user's personal identity out of source, artifacts and logs.

All deployment PCs run Windows 11: A RTX 4090, B RTX 4060 laptop, C RTX 5070 Ti
and Ryzen 7 9800X3D. Preserve BF16-only unquantized KV/index/draft storage,
FP32 hidden/pre_mix stage wire, exact full target/SWA replay and the complete
three-PC graph. Optional FP16 cache mode is excluded. Intrinsic FP16 operands
inside the selected EXL3 weight kernels are separate.

Do not modify working Qwen StrataRPC, create another project repository,
download model/Engram payloads into cloud, rent GPUs or depend on the user's
personal cloud VM. Current user instructions override the historical attachment
authorization language; see docs/DECISIONS.md and the full implementation spec.

## Source and development environment

- Checkout: /workspace/DeepSeekRPC; obtain its exact revision with
  `git rev-parse HEAD`. The generated review manifest records the final commit.
- Read-only references: /workspace/deepseekrpc-references. Writable build copies:
  /workspace/deepseekrpc-build. External logs: /workspace/deepseekrpc-evidence.
- Primary Python: /workspace/deepseekrpc-tools/.venv/bin/python, Python 3.12
  with Torch 2.13 CPU. Independent target-version CPU environment:
  /workspace/deepseekrpc-tools/target-cpu-2.11/bin/python.
- Current primary installed audit uses the actual patched empty vLLM build and
  official no-CUDA EXL3 plugin wheel. These contain no surrogate GPU results.
- The cloud has no GPU, MSVC, nvcc or PowerShell. Native Windows CI is separate.
- The reusable cloud configuration is saved as a draft. Saving is not publishing;
  a missing publication tool is not a reason to stop repository implementation.

All source pins, archive hashes, model metadata and exact patch hashes are
under provenance/ and patches/. DeepInfer is the eighth source-only reference,
pinned at 8b16149741a4935e1dec8cea27853f273b374c5b. Read
docs/DEEPINFER_REVIEW.md; do not run examples that acquire model payloads or
adopt its FP8 draft keys, bounded replay, biased rejection epsilon or confidence
pruning.

## Implemented scope

- Selective publisher-verified pack loading, actual EXL3 dense/routed execution,
  shared experts, bounded desktop caches and B's all-page VirtualLocked canonical
  preload, closed source files and one bounded pinned transfer tile. Active Windows
  weight/Engram readers deny write/delete/replacement access until close.
- Actual owned vLLM decoder/vision/three-block draft graph and native weights;
  BF16 CUDA attention/index/source/shadow/SWA, exact undo, FP32 compressor pairs,
  learned Engram, delayed mHC and whole-stage memory admission.
- Full two-pass CED, image-safe encoder prefix checkpoints and full target
  replay; exact Markov proposal/rejection/stop handling, three transactional RNG
  domains and separate processed/pending token identities.
- Persistent Windows workers, mutual TLS, bounded data/control RPC, independent
  cancellation/heartbeat, incarnation-bound recovery and all-stage GPU/history
  commits plus Stage C RNG finalization before output publication.
- Bounded original OpenAI JSON/SSE text/image/reasoning/tool handling, early
  LOADING diagnostics, request cancellation without unloading, private PKI and
  isolated Windows launch/configuration paths.
- All-stage bounded state snapshots, prevalidated restore, administrative pause
  and exact pending-input/history/image/RNG/stop continuation.
- Content-free observability, local GPU/cluster diagnostics, real API benchmark
  commands and the separate OpenCode provider with actual pinned SDK audits.
- Five-wheel private Python installer, exact native/source admission, app-local
  CRT, private Triton/metadata caches and retained exact native dependency notices.

docs/REVIEW_GUIDE.md maps the source and review hazards. Optional grammar,
response-format and named-tool constraints fail explicitly as permitted by the
specification. They are not mandatory implementation omissions. Speculation
stays disabled until the actual loaded components and current mesh pass their
in-memory cluster qualification gate; a report file cannot enable it.

## Actual checks and current build gates

At 30b24d3b78caa538b8b18f927c27056a08c020d6, Windows CPU run 39461505378
passed 389 tests/one intentional non-Windows-refusal skip, wheel build/install,
CLI and actual OpenCode SDK checks. External runtime/compiler run 39438027672
at 3457a686c3aeee5b4306ad5cfb025fa284712846 passed the private Python 3.12.14,
TinyCC load, offline SM89/SM120 Triton compilation, full external dependencies,
pip check, CUDA-Torch imports, Gloo subgroup lifetime and unloaded API checks.
Both refreshed installed cloud environments pass seven integration checks and
pip check. The Windows wheel and current installed/source package share identity
93f1f2b7f4ded70f2596f41357b2cd1232241dd908fb40029d13f1014b8e3054
(105 files). The original parser passes twenty cases in the complete Windows
runtime. These results do not execute the model.

Use qualified EXLlama run 39345608258 and project native ABI 2 run 39410211028.
Native vLLM run 39428534610 passed source
b8d36645f60d3d4d2f83d2b57cf465306476430b with the exact five-patch profile,
actual core/MoE imports without SDK and both SM89/SM120 inspections. It excludes unused FlashKDA/Qutlass/DeepGEMM targets,
not the owned BF16/EXL3/mHC graph. The fifth patch hash is
974afb34134cedc3bdccda2153b0722c94c4982cbadd82d51835b5cfa22d7ce1.
The admitted wheel metadata version is 0.31.0+deepseekrpc2.cu132.

Automatic complete-runtime run 39462821561 passed all eight checks at 30b24d3:
private_vllm_cache, versions, application_source_identity,
private_compilers_without_sdk, native_without_sdk, actual_local_gloo,
complete_python_integration and actual_original_parser. It uses private Python
3.12.14 and retains thirteen exact native source notices. All 81 downloaded
files match the sealed manifest. Actual bounded baseline reports are committed
in evidence/whole-runtime-30b24d3/; full original artifacts/logs remain external.
DSR-AUDIT-027 file-sharing/Unicode/concurrent-reader and constructor-lifetime
checks passed on actual Windows. DSR-024/025/026 build/version/private-cache
checks passed in the native/full runtime. No native profile is pending here.

DSR-AUDIT-011 remains open: intermittent Linux Torch 2.13/MKL sqrt bytes match
an approximate AVX2 path despite the ATen high-accuracy wrapper. Preserved
synthetic captures and instruction probes are in evidence/cpu-sqrt-diagnostic/.
Four observed suites passed 384 tests/four Windows-only skips, but none captured
the failing call. No oracle, tolerance, vendor library or runtime math was
changed. A later pass is not a repair. Other findings and historical build
failures remain recorded in docs/FINDINGS.md and the append-only build ledger.
GitHub API's temporary 401 recovered; do not request a token or reconnect.

## Finish the candidate

The final capability/docs freeze and append-only evidence ledger record the
actual 30b24d3 baseline. Before delivering the first review packet, obtain
matching Windows CPU and complete-runtime reports for the frozen HEAD, then
run tools/build_review_packet.py outside the source checkout. Its REVIEW.json
binds the archive and reports to that exact commit; do not infer a final result
from an earlier baseline. Reuse unchanged qualified native wheels; do not
rebuild CUDA merely for documentation changes. Verify every downloaded bundle
file against its manifest and preserve source/evidence externally.

For a complete-runtime rerun after a Python/source fix:
`gh workflow run windows-runtime.yml --ref main -f vllm_run=39428534610
-f exllama_run=39345608258 -f native_run=39410211028` (one shell command).
Use native source/profile admission to decide whether those components remain
valid. Consult the exact matching review packet and current Actions status for
final run IDs, rather than treating this earlier baseline as current evidence.

The packet builder refuses mismatched or incomplete reports. Independent Astra
review remains pending and user-mediated. Every GPU/model execution result,
full-size B residency, 1M context, distributed recovery, vision, actual Docker
coexistence and throughput remain unrun/unmeasured. Do not ask for early hardware
testing or mark this production-proven.
