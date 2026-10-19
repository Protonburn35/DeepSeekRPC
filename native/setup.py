"""Native CUDA extension; build explicitly with the pinned Windows CUDA stack."""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="deepseekrpc-cuda",
    version="0.0.1.dev0",
    license="AGPL-3.0-only",
    ext_modules=[CUDAExtension(
        "deepseekrpc_cuda",
        ["bindings.cpp", "operations.cu"],
        extra_compile_args={
            "cxx": ["/O2", "/std:c++17", "/fp:precise"] if os.name == "nt" else ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "--std=c++17", "--ftz=false", "--fmad=false"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
