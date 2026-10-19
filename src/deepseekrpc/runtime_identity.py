"""Require actual pinned native Windows components before stage construction."""
from __future__ import annotations

import hashlib
from importlib import metadata, resources
import json
from pathlib import PurePosixPath
import sys

from .jsonutil import ValidationError
from .runtime_policy import configure_exact_cuda_math


def require_windows_runtime(device=0):
    if sys.platform != "win32":
        raise RuntimeError("the model runtime requires native Windows; CPU source diagnostics are separate")
    configure_exact_cuda_math()
    lock = json.loads(resources.files("deepseekrpc._vendor.vllm_windows").joinpath("RUNTIME.json").read_text())
    for name, expected in lock["versions"].items():
        if metadata.version(name) != expected:
            raise ValidationError(f"installed {name} differs from the admitted runtime version {expected}")
    for record in lock["files"]:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValidationError("runtime source identity contains an invalid relative path")
        path = metadata.distribution(record["distribution"]).locate_file(str(relative))
        if (not path.is_file() or path.stat().st_size > 4 * 1024 * 1024 or
                hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValidationError(f"installed runtime source differs from its admitted patch: {record['path']}")
    import torch
    if not torch.cuda.is_available() or torch.version.cuda != "13.0":
        raise RuntimeError("the actual admitted Torch CUDA runtime and a working NVIDIA device are required")
    torch.cuda.set_device(device)
    architecture = torch.cuda.get_device_capability(device)
    if architecture not in ((8, 9), (12, 0)):
        raise ValidationError("this application build requires SM89 or SM120")
    import exllamav3_ext
    import deepseekrpc_cuda
    if (not callable(exllamav3_ext.exl3_gemm) or deepseekrpc_cuda.abi_version() != 2 or
            not callable(getattr(deepseekrpc_cuda, "windows_memory_budget", None))):
        raise ValidationError("the actual packed/native operation ABI does not match")
    return {"versions": lock["versions"], "source_fingerprints_verified": True,
            "architecture": f"sm_{architecture[0]}{architecture[1]}",
            "GPU_execution": "not established by this identity check"}
