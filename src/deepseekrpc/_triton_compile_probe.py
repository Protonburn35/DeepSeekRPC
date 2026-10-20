"""Small project-owned offline compiler input, never launched on a device.

Import only after configuring the actual Windows Triton compiler policy.
"""
import triton
import triton.language as tl


@triton.jit
def copy_probe(source, destination, count: tl.constexpr):
    offsets = tl.arange(0, count)
    values = tl.load(source + offsets)
    tl.store(destination + offsets, values + 1.)
