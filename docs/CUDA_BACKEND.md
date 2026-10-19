# Native CUDA operation contract

The native extension is part of the unfinished runtime. Windows compile/link,
wheel install, CPU import and both SM89/SM120 cubins passed in run 39314752138.
The expanded checks also passed in run 39316435141. Actual GPU execution remains unrun.
The CUDA build workflow uses a CPU Windows runner and never executes kernels.
The intended deployment OS is Windows 11 on all three PCs.

The user's current requirement is BF16-only, unquantized KV/index storage. FP16
cache mode and quantized-cache fallbacks are excluded. This supersedes the
original implementation prompt's FP16-default language; FP16 inside existing
EXL3 weight kernels remains independent of this cache policy.

`native/operations.cu` implements these operations without INT8 activations or
implicit lower-precision cache storage:

| Operation | Storage and computation | State contract |
| --- | --- | --- |
| Sparse latent attention | Two BF16 KV pools, FP32 query/accumulation/output; online softmax includes the learned per-head sink as a zero-value term | Int32 per-query slots select the small SWA staging pool or persistent compressed source pool; -1 is padding, duplicates retain their multiplicity. Invalid slots/nonfinite scores fail. No full cache concatenation or dense 1M score matrix. |
| Index scores | BF16 stored keys, FP32 query/dot/rectification/head-weight reduction | A bounded tile of at most 8192 positions; explicit per-query visible counts; unreachable/padded positions are -inf. Candidate filtering and global top-k require the surrounding adapter. |
| Original Engram row decode | Exact FP8 E4M3 values multiplied by standard UE8M0 scales in FP32, then explicitly rounded to BF16 | Bounded encoded [rows,264] input, standard scale-byte-zero interpretation; NaN/overflow fails. Learned projection/gating is a separate graph operation. |
| Cache scatter | Exact BF16 copies into existing slot pools | Checks physical bounds, duplicate targets and expected int64 slot generations. Failed operations may have tentative writes; transaction rollback is mandatory before reuse/publication. |

Python wrappers inspect the native status after stream completion before
returning a result. This synchronous baseline establishes an ordering boundary;
future overlap must retain equivalent event and transaction dependencies. The
wrappers require the real compiled extension and a real CUDA device. They do not
substitute CPU model computation.

The attention equation and index ordering follow official October
`inference/kernel.py:sparse_attn_kernel` and
`inference/model.py:Indexer.forward`. Source hashes are in models.lock.json.
Independent tiny CPU equations in reference_attention.py compare against dense
Torch softmax/matrix operations; these fixtures do not validate CUDA execution.
ExLlama supplies the real packed EXL3 matrix kernels separately; implementing
these four operations does not complete EXL3 loading or the distributed graph.

The build pins Torch 2.11.0+cu130 (the selected Windows vLLM requirement) and the
official CUDA 13.2 compiler packages used by the selected ExLlama build recipe.
Their hashes and exact byte counts are recorded in the SDK lock. Compiler minor
version interoperability, both embedded cubins, extension loading and full
upstream native builds must be verified before qualification. Compilation does
not establish numerical GPU correctness or hardware residency.

The separate `windows-exllama.yml` job builds the full pinned EXLlama source,
without namespace replacements or a JIT-only wheel. Its helper verifies the
source archive hash, extracts a separate writable build tree, compiles both
architectures, imports the installed native extension and Python package, and
checks dependency consistency. This is separate from the four project kernels
and from the still-required native vLLM build. The upstream compiler math flags
are preserved; numerical packed-weight execution must be qualified separately.
