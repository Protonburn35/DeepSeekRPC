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
            "cxx": ["/O2", "/fp:precise", "/Zc:preprocessor", "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX"]
                   if os.name == "nt" else ["-O3", "-std=c++17"],
            # Torch appends the Windows NVCC standard flag unconditionally.
            "nvcc": ["-O3", "--ftz=false", "--fmad=false"] +
                    (["-DWIN32_LEAN_AND_MEAN", "-DNOMINMAX", "-Xcompiler=/Zc:preprocessor"]
                     if os.name == "nt" else ["--std=c++17"]),
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
