"""Pin Windows JIT tools and disable ambient interpreter/kernel overrides.

This policy imports no Torch/Triton code. The bundled compilers need no
developer SDK; their actual build/load and offline code generation are separate
qualification checks. Setting paths is not evidence of successful execution.
"""
import hashlib
from importlib import metadata, resources
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading

from .jsonutil import ValidationError

_mutex = threading.Lock()
_environment = None
_cache_root = None
_report = None


def compiler_qualification_environment(environment):
    """Remove inherited SDK/compiler selectors from a fresh qualification child."""
    result = dict(environment)
    selected = {"CC", "CXX", "CUDACXX", "CUDA_PATH", "CUDA_HOME", "CUDA_BIN_PATH", "NVCC", "PYTHONPATH",
        "VCINSTALLDIR", "VCTOOLSVERSION", "VCTOOLSINSTALLDIR", "VSINSTALLDIR", "INCLUDE", "LIB", "LIBPATH",
        "CL", "_CL_", "LINK", "_LINK_", "PTXAS_OPTIONS"}
    for key in list(result):
        upper = key.upper()
        if upper in selected or upper.startswith(("CUDA_PATH_V", "WINDOWSSDK", "TRITON_")):
            del result[key]
    windows = Path(environment["SystemRoot"])
    result.update(PATH=os.pathsep.join([str(windows / "System32"), str(windows)]),
        PYTHONNOUSERSITE="1", PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="-1")
    return result


def _selected_environment():
    return {key: value for key, value in os.environ.items()
            if key.upper().startswith("TRITON_") or key.upper() in {"CC", "CUDA_PATH", "CUDA_HOME", "PTXAS_OPTIONS"}}


def configure_windows_triton(cache_root=None):
    global _environment, _cache_root, _report
    if sys.platform != "win32":
        raise RuntimeError("the pinned Triton compiler policy requires native Windows")
    with _mutex:
        if _environment is not None:
            if cache_root is not None and Path(cache_root).absolute() != _cache_root:
                raise ValidationError("Triton cache ownership cannot change within a worker process")
            assert_windows_triton()
            return dict(_report)
        if any(name == "triton" or name.startswith("triton.") for name in sys.modules):
            raise ValidationError("Triton was imported before its compiler policy; start a fresh worker")
        lock = json.loads(resources.files("deepseekrpc._vendor.vllm_windows").joinpath("TRITON.json").read_text())
        distribution = metadata.distribution(lock["distribution"])
        if distribution.version != lock["version"]:
            raise ValidationError("installed Triton differs from the pinned Windows JIT toolchain")
        prefix = Path(sys.prefix).resolve()
        verified = {}
        for record in lock["files"]:
            relative = PurePosixPath(record["path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in record["path"]:
                raise ValidationError("invalid pinned compiler input path")
            path = Path(distribution.locate_file(record["path"])).resolve(strict=True)
            if (not path.is_relative_to(prefix) or not path.is_file() or
                    path.stat().st_size != record["bytes"] or record["bytes"] > 32 * 1024 ** 2):
                raise ValidationError("Triton compiler input is missing or outside the private runtime")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                    raise ValidationError("Triton compiler input differs from the admitted wheel")
            verified[record["path"]] = record["sha256"]
        package = Path(distribution.locate_file("triton")).resolve(strict=True)
        cuda = package / "backends/nvidia"
        cache = Path(cache_root).absolute() if cache_root is not None else Path(sys.prefix) / ".deepseekrpc-triton"
        if cache.is_symlink() or cache.exists() and not cache.is_dir():
            raise ValidationError("private Triton cache is redirected or is not a directory")
        cache.mkdir(parents=True, exist_ok=True)
        # The pinned wheel appends EXE before constructing env_nvidia_tool's
        # key, so Windows uses TRITON_PTXAS.EXE_PATH (including the dot).
        selected = {"CC": str(package / "runtime/tcc/tcc.exe"), "CUDA_PATH": str(cuda), "CUDA_HOME": str(cuda),
            "TRITON_PTXAS.EXE_PATH": str(cuda / "bin/ptxas.exe"),
            "TRITON_LIBDEVICE_PATH": str(cuda / "lib/libdevice.10.bc"),
            "TRITON_LIBCUDA_PATH": str(cuda / "lib/x64"),
            "TRITON_INTERPRET": "0", "TRITON_KERNEL_OVERRIDE": "0", "TRITON_KERNEL_DUMP": "0",
            "TRITON_F32_DEFAULT": "ieee", "TRITON_DISABLE_LINE_INFO": "1",
            "TRITON_HOME": str(cache), "TRITON_CACHE_DIR": str(cache / "cache"),
            "TRITON_DUMP_DIR": str(cache / "dump"), "TRITON_OVERRIDE_DIR": str(cache / "override")}
        # Inherited cache backends, architecture/mock-PTX settings, backend
        # include directories and PTXAS options are not part of this profile.
        for key in _selected_environment():
            os.environ.pop(key, None)
        os.environ.update(selected)
        _cache_root, _environment = cache, _selected_environment()
        _report = {"version": lock["version"], "verified_files": verified,
            "host_compiler": "bundled TinyCC", "device_compiler": "bundled CUDA ptxas",
            "kernel_override": False, "interpreter": False, "fp32_dot_default": "ieee",
            "compiler_execution": "not established by policy configuration"}
        return dict(_report)


def assert_windows_triton(*, required=True):
    if _environment is None:
        if required:
            raise ValidationError("Windows Triton compiler policy has not been configured")
        return
    if _selected_environment() != _environment:
        raise ValidationError("Windows Triton toolchain or numerical environment changed after configuration")
    knobs = sys.modules.get("triton.knobs")
    if knobs is not None and (knobs.runtime.interpret or knobs.runtime.override_arch is not None or
                              knobs.compilation.override or knobs.build.impl is not None):
        raise ValidationError("Triton interpreter, architecture, kernel or build override is active")
