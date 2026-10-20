# DeepSeekRPC

Native Windows three-PC inference for DeepSeek-V4.1-Flash-1017 EXL3 2.9.
The complete application source is implemented for whole-source review.
Windows CPU checks, native builds/imports and complete private-runtime
installation qualification pass. GPU/model execution and independent reviews
remain pending; this is not a release candidate.

The topology is 4090 stage A → resident 4060 laptop stage B → 5070 Ti stage C,
default layers 18/5/17, full eager 1M context, BF16 KV/index and exact SWA replay.
All three deployment PCs run Windows 11; stage C uses a Ryzen 7 9800X3D.
The implementation uses the pinned Windows vLLM graph and EXL3 kernels. The working Qwen
StrataRPC deployment is kept separate.

Implemented source includes actual owned decoder/vision/draft construction,
packed dense/routed EXL3 execution, bounded expert tiers and original learned
Engram operations. BF16 CUDA attention/index/source/shadow/SWA pools use explicit
generations, causal visibility, undo and full two-pass target replay. A boot-only
Win32 locked preload verifies all canonical laptop pages and closes its weight
file source. Physical full-size residency remains unqualified.

Persistent Windows workers use mutual TLS, bounded control/data transfer,
all-stage commit barriers, cancellation and authenticated mesh replacement.
The bounded OpenAI JSON/SSE API preserves original text, reasoning, tools and
images. Exact encoder prefix reuse, all-stage state snapshots and paused
pending-input continuation have integrated source. The DSpark graph and exact
sampled verifier remain off by default until the actual loaded cluster passes
its qualification probe. Local GPU diagnostics, bounded numeric logs,
diagnostic archives and matched real-API benchmark commands are included;
their full-model GPU results and throughput remain unmeasured.

See [requirements](docs/REQUIREMENTS.md),
[compatibility](docs/COMPATIBILITY.md), [memory](docs/MEMORY_PLAN.md),
[protocol](docs/RPC_PROTOCOL.md), [Windows installation](docs/INSTALL_WINDOWS.md),
[diagnostics](docs/DIAGNOSTICS.md), [OpenCode](docs/OPENCODE.md), and
[handoff](HANDOFF.md). The eighth reference, official DeepInfer, is reviewed in
[DEEPINFER_REVIEW.md](docs/DEEPINFER_REVIEW.md); its bounded replay and biased
sampling adjustment are not adopted.
The [whole-runtime review guide](docs/REVIEW_GUIDE.md) maps the integrated
source and the remaining qualification questions.
The first complete Windows installation at `30b24d3` has retained
[build/install evidence](evidence/whole-runtime-30b24d3/summary.json).
The final review packet binds its source archive to matching Windows CPU and
installed-runtime reports for the exact frozen commit.

## CPU development

Use the existing checkout; Codex tasks already have isolated workspaces.
CPython 3.12 is required. In this prepared cloud environment:

```bash
cd /workspace/DeepSeekRPC
source /workspace/deepseekrpc-tools/.venv/bin/activate
export PYTHONUTF8=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
python -m deepseekrpc.cli plan --model-config provenance/model-config.json
```

For a fresh development environment, create a Python 3.12 virtual environment,
then install the hash-pinned tools. Use `requirements-reference-windows.lock`
instead on native Windows. CPU PyTorch is for independent reference checks; it
does not qualify the Windows CUDA runtime stack.

```bash
python -m pip install --require-hashes --only-binary=:all: \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements-dev.lock -r requirements-reference-linux.lock
python tools/ci_check.py
```

The CI helper runs the suite, builds and installs a wheel, then checks its CLI
outside the checkout. Its BUILD.json distinguishes source revision, dirty
working trees, check exits and unrun GPU/full-model validation. CI builds target
native Windows only. Repository pushes and CPU CI are authorized by the user;
local cloud checks remain development evidence. The Win32 lock test is
explicitly skipped on Linux.

## Provenance and inventory

Source references and metadata are retained outside the checkout in
`/workspace/deepseekrpc-references`. Model payloads must never be downloaded into
Codex. The compressed machine-readable inventory contains metadata only:

```bash
python tools/acquire_references.py --root /workspace/deepseekrpc-references
PYTHONPATH=src python tools/build_inventory.py \
  --metadata-root /workspace/deepseekrpc-references/metadata
```

`provenance/allocation-plan.json` records actual tensor bytes and prospective
state arithmetic separately. `provenance/tensors.jsonl.gz` maps every selected
tensor to its file, exact offset/shape/type, owner and component. Payload hashes
in the manifest are publisher metadata; no full weight integrity or hardware
fit claim is implied. Never use a simulated TCP fixture as the production model
executor. The test-only stage math is confined to tests/tcp_fixture.py.
