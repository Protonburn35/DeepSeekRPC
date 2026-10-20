"""Require actual pinned native Windows components before stage construction."""
from __future__ import annotations

import hashlib
from importlib import metadata, resources
import json
from pathlib import Path, PurePosixPath
import sys

from .jsonutil import ValidationError, fingerprint
from .runtime_policy import configure_exact_cuda_math, configure_owned_vllm_cache


def application_source_identity():
    """Bind every shipped Python module and data manifest, independent of path."""
    entries = {}

    def visit(directory, prefix=""):
        for item in sorted(directory.iterdir(), key=lambda p: p.name):
            relative = prefix + item.name
            if item.is_dir() and item.name != "__pycache__":
                visit(item, relative + "/")
            elif item.is_file() and (item.name.endswith((".py", ".json")) or item.name == "LICENSE"):
                data = item.read_bytes()
                if len(data) > 4 * 1024 * 1024:
                    raise ValidationError("application identity file exceeds its admitted source size")
                entries[relative] = hashlib.sha256(data).hexdigest()
    visit(resources.files("deepseekrpc"))
    return {"sha256": fingerprint(entries), "files": len(entries)}


def _distribution_identity(name):
    """Hash actual used distributions, including compiled native libraries."""
    distribution = metadata.distribution(name)
    entries, total = {}, 0
    for relative in distribution.files or ():
        pure = PurePosixPath(str(relative).replace("\\", "/"))
        if (pure.is_absolute() or ".." in pure.parts or any(part.endswith(".dist-info") for part in pure.parts) or
                pure.suffix not in {".py", ".json", ".pyd", ".dll", ".exe", ".bc", ".lib", ".def", ".h", ".hpp", ".a"}):
            continue
        path = Path(distribution.locate_file(relative))
        size = path.stat().st_size
        total += size
        if not path.is_file() or size > 4 * 1024 ** 3 or total > 8 * 1024 ** 3:
            raise ValidationError("installed component identity exceeds its admitted source/native bounds")
        with path.open("rb") as stream:
            entries[pure.as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    if not entries:
        raise ValidationError("installed runtime component has no inspectable source or native files")
    return {"sha256": fingerprint(entries), "files": len(entries), "bytes": total}


def require_windows_runtime(device=0, *, cache_root=None):
    if sys.platform != "win32":
        raise RuntimeError("the model runtime requires native Windows; CPU source diagnostics are separate")
    metadata_root = Path(cache_root).absolute().parent / "vllm" if cache_root is not None else None
    configure_owned_vllm_cache(metadata_root)
    from .triton_policy import configure_windows_triton
    compiler_policy = configure_windows_triton(cache_root)
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
    components = {name: _distribution_identity(name) for name in ("vllm", "vllm-exl3", "exllamav3", "deepseekrpc-cuda", "triton-windows")}
    # Compressed extensions may extract the actual loaded .pyd on import.
    # Bind that binary as well as all files recorded by the wheel installer.
    for name, module in (("exllamav3_ext", exllamav3_ext), ("deepseekrpc_cuda", deepseekrpc_cuda)):
        path = Path(module.__file__)
        if not path.is_file() or path.stat().st_size > 4 * 1024 ** 3:
            raise ValidationError("loaded native runtime path is not a bounded real component")
        with path.open("rb") as stream:
            components[name] = {"sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    components["deepseekrpc"] = application_source_identity()
    return {"versions": lock["versions"], "source_fingerprints_verified": True,
            "components": components, "component_identity": fingerprint({"versions": lock["versions"], "files": components}),
            "architecture": f"sm_{architecture[0]}{architecture[1]}",
            "compiler_policy": compiler_policy, "GPU_execution": "not established by this identity check"}
