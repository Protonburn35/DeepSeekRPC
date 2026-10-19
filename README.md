# DeepSeekRPC

Native Windows three-PC inference for DeepSeek-V4.1-Flash-1017 EXL3 2.9 is the
project goal. This repository is currently an **incomplete engineering
checkpoint**, not a release candidate or a model-serving application.

The planned topology is 4090 stage A → resident 4060 laptop stage B → 5070 Ti stage C,
default layers 18/5/17, full eager 1M context, FP16 KV/index and exact SWA replay.
All three deployment PCs run Windows 11; stage C uses a Ryzen 7 9800X3D.
The Windows vLLM graph remains the intended foundation. The working Qwen
StrataRPC deployment is kept separate.

Implemented foundations include pinned source/model provenance, a complete
header-derived tensor ownership ledger, bounded typed framing, native TCP with
mutual TLS, tensor fragmentation, tentative-state rollback, resident-only byte
storage, a leased RAM cache, bounded original Engram row reads, the official
October text/tool/vision encoding, and independent CPU operation fixtures.
The pinned tokenizer and original Engram compressed token map/hash operation
also run on CPU with bounded transactional validity; learned Engram execution
remains unfinished.
Selective local loading verifies publisher file hashes and exposes only owned
tensor ranges. A boot-only Win32 locked preload closes the laptop disk source;
full device residency and model execution are still unqualified.
The vLLM stage adapter, CUDA/FP16 attention integration and complete application
features remain unfinished. See [requirements](docs/REQUIREMENTS.md),
[compatibility](docs/COMPATIBILITY.md), [memory](docs/MEMORY_PLAN.md),
[protocol](docs/RPC_PROTOCOL.md), and [handoff](HANDOFF.md).

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
