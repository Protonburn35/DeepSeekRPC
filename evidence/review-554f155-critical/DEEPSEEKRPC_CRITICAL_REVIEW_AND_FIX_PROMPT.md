# DeepSeekRPC critical review and fix prompt

Reviewed commit: `554f155dc88376c962074167149a77c8486fcc66`  
Review date: 2026-10-20  
Repository: https://github.com/Protonburn35/DeepSeekRPC

**Verdict: three confirmed failures need fixing before hardware trials. Two prevent normal worker startup; the third aborts the first request using each uncached image.** These are ordinary execution paths, not speculative edge cases.

The review changed no application source and pushed nothing to GitHub. Bundle checksums were verified. The enclosed Windows build/import evidence is consistent, but explicitly records no GPU execution or full-model validation. The three findings below were independently reproduced on CPU at their failing Python control-flow boundaries.

## R1 — Loader rejects the run identity produced by its own configuration

Priority: P1; blocks A, B and C startup with valid normal configurations.

Source:

- `src/deepseekrpc/deployment_config.py:106`: `DeploymentConfig.read()` sets `run = fingerprint(shared)`. `jsonutil.fingerprint()` returns a 64-character SHA-256 hex digest.
- `src/deepseekrpc/worker.py:314`: normal startup passes that value unchanged as `load_stage(..., run_id=self.config.run)`.
- `src/deepseekrpc/bootstrap.py:136`: the loader requires `hex_id(run_id, 32, "worker run")`.
- `src/deepseekrpc/protocol.py:135` and `mesh.py:143`: the wire protocol and mesh correctly require 64 characters.

A valid configuration therefore fails with `ValidationError: invalid worker run identity` before weight loading and model admission. The 32-character `cluster_id` and the 64-character derived run fingerprint are different identifiers.

Fix: make loader validation require the same complete 64-character run fingerprint as the configuration, mesh and protocol. Do not truncate the digest or change the protocol to accommodate the incorrect loader check. Prefer validating this before expensive runtime initialization.

Regression: read a real valid shared/local configuration and pass its resulting run identity through the actual loader admission boundary, with hardware initialization isolated. Confirm all three roles accept the 64-character run fingerprint and malformed or 32-character run values fail. A test that checks only configuration parsing will miss this mismatch.

## R2 — Nested GPU self-test tries to change the locked vLLM cache root

Priority: P1; independently blocks normal loading and configured `--self-test`.

Source:

- `src/deepseekrpc/bootstrap.py:114`: initializes runtime policy with `cache_root=state_root / "kernels/triton"`.
- `src/deepseekrpc/runtime_identity.py:56`: derives and locks the matching vLLM metadata root, `state_root / "kernels/vllm"`.
- `src/deepseekrpc/bootstrap.py:152`: subsequently calls `run_cuda_checks()`.
- `src/deepseekrpc/gpu_selftest.py:34`: that function calls `require_windows_runtime(device.index)` again, omitting the root.
- `src/deepseekrpc/runtime_identity.py:56`: omission selects `sys.prefix / ".deepseekrpc-vllm"` rather than the already configured worker root.
- `src/deepseekrpc/runtime_policy.py:28`: correctly rejects that ownership change.

After R1 is fixed, a normally configured worker still fails with `ValidationError: vLLM cache ownership cannot change within a worker process`. The configured preflight path has the same sequence at `worker.py:116` and `worker.py:119`, so its GPU self-test also fails. This happens before any native operation checks execute.

Fix: preserve one cache owner across the whole initialization/self-test call chain. Pass the selected root through `run_cuda_checks`, or make a repeated runtime identity check without a new root reuse the already selected owner. Preserve standalone diagnostic behavior and rejection of an explicitly different root. Do not remove the ownership guard or silently move worker caches into shared/interpreter directories.

Regression: exercise the real sequence of explicit worker-root initialization followed by the nested runtime check in the same process. Verify paths remain under the worker state directory, repeated calls succeed, and an explicitly different root still fails. Cover both normal startup and configured `--self-test`. A CPU test may isolate platform/native capability observers while retaining the actual cache policy; actual GPU execution remains a separate hardware check.

## R3 — First uncached image aborts at an undefined timing variable

Priority: P1 for the vision feature; text-only requests do not take this path.

Source: `src/deepseekrpc/vision_runtime.py:45`, particularly lines 55–56.

`ImageWindowResolver.__call__()` encodes and stores a new image, then evaluates `time.perf_counter_ns() - first` twice. `first` has never been defined in this method or at module scope. A similarly named local assignment is misplaced in the separate `can_reuse_prefix()` method at line 22; it cannot supply this value.

The first request using an uncached image therefore raises `NameError: name 'first' is not defined` after feature storage, before the resolver returns image rows. The stage service faults the request. Retrying an image whose features were already stored may skip the failing branch and hide the defect.

Fix: initialize the timer inside the uncached-image branch before the operation being timed. Remove the unrelated assignment in `can_reuse_prefix`. Keep the image output and cache behavior unchanged.

Regression: invoke the actual resolver's uncached-image path through feature storage and row return, then invoke its cached path. Verify successful row delivery, exactly one encode, and valid timing metadata. Controlled tensor/asset boundaries are sufficient for the Python regression; they do not establish actual vision GPU correctness. Add an undefined-name check so this class of failure cannot pass unnoticed again.

## Reproduction evidence

Files alongside this report:

- `reproduce_review_findings.py`
- `reproduction-results.json`

Run against an extracted source tree with Python and NumPy:

```text
python reproduce_review_findings.py /path/to/DeepSeekRPC/source
```

R1 uses the actual configuration parser and exact loader validator. R2 uses the real cache policy and runtime identity entry point, substituting only that module's platform observer to reach the guard on CPU. R3 executes the unchanged resolver class with controlled tensor/asset boundaries. All three reproduced the stated exceptions. No model data, CUDA execution, performance benchmark, or complete three-process inference was involved.

## Prompt to send to Sol

Review and fix R1, R2 and R3 above against the current branch. Confirm each issue still exists before changing it if the branch has advanced. Keep the work focused on these confirmed startup and vision failures. Preserve the agreed Windows deployment, BF16 KV/index storage, FP32 wire, exact CED replay, exact speculative sampling, privacy policy and existing numerical tolerances.

Add meaningful regressions that cross the mismatched interfaces rather than testing each helper alone. Re-run appropriate CPU tests and the Windows build/install qualification. Report the fixing commits and actual results. Distinguish GPU-free validation from hardware execution; passing compilation/import checks must not be described as successful full-model inference. Record any remaining unrun hardware checks explicitly.

The reproduction script here demonstrates the original failures; after fixes, its expected-failure assertions may need to be replaced by permanent success regressions in the project tests.

This review found no additional confirmed catastrophic numerical/state defect in the inspected paths. It does not establish full-model correctness, physical memory residency, throughput, or successful three-PC CUDA inference. Those remain unverified in the supplied bundle.
