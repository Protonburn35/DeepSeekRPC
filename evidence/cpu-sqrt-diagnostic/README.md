# DSR-AUDIT-011 captured CPU sqrt path

This is a small **synthetic CPU arithmetic** failure capture, not a model
checkpoint, GPU result or desired numerical baseline. The frozen router oracle
and its tolerances remain unchanged. The intermittent selection cause is open.

`captured-failure.npz` contains the exact captured FP32 input/output and an
independent FP64 sqrt of that input, each shaped [7,384]. Load it with
`numpy.load(..., allow_pickle=False)`. Only the first 1344 results are inaccurate;
three immediate repeats on identical inputs were accurate. Checksums and scope
are recorded in PROVENANCE.json.

The project-owned `instruction-probe.c` was compiled on this development Intel
CPU with GCC, `-O2 -mavx2 -ffp-contract=off -fPIC -shared`. Variant 0 computes
`input * vrsqrtps(input)` without refinement. It reproduces **all 1344 bad
results bit for bit**. Other variants test reciprocal rsqrt, reciprocal
approximation and direct hardware sqrt. These instructions' approximate seed
bits may differ on other CPUs; this is a signature on the captured host.

The installed Torch 2.13.0+cpu library exports vmsSqrt. Explicit MKL low/high/
enhanced-performance modes were checked in a separate diagnostic process. With
`MKL_ENABLE_INSTRUCTIONS=AVX2` set only for another isolated diagnostic process,
enhanced-performance mode also reproduces all 1344 captured bad values. Its
high-accuracy mode does not. Normal AVX512 high-accuracy mode matches the
accurate second partition. See the three numeric JSON reports.

The installed ATen VML wrapper header explicitly passes high accuracy, FTZ/DAZ
off and ignore-errors flags to vmsSqrt. That header's hash is retained. A match
to a reduced-accuracy path is not proof of why the first parallel partition
selected it intermittently. Global VML mode alone has not explained it.
No environment/accuracy override is applied to production or normal tests.

Native Windows CPU reference runs, including the actual Torch 2.11+cu130 stack,
have passed independently. A passing run does not repair the captured Linux
failure, and neither these probes nor that Windows evidence establishes GPU
router or full-model numerical correctness.
