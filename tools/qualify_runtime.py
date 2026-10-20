#!/usr/bin/env python3
"""Qualify an installed Windows bundle with no visible GPU or compiler SDK.

Checks real installed distributions, native library registration and the
original parser. No model payload, namespace stub or GPU result is used.
"""
from importlib import metadata
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def qualify_local_gloo(output):
    """Exercise real local stage-group dependencies without CUDA collectives."""
    import tempfile
    from datetime import timedelta
    import gc
    import torch
    from deepseekrpc.local_groups import initialize_local_process_group
    distributed = torch.distributed
    # Stage groups are local TP=PP=1; the three-PC protocol does not use
    # Gloo for inter-PC CUDA tensors. Exercise the same in-memory store as
    # startup, even when the output directory has spaces/non-ASCII characters.
    with tempfile.TemporaryDirectory(prefix="local Gloo \u0394 ", dir=output):
        store = initialize_local_process_group()
        try:
            # The real vLLM setup creates additional groups. Force collection
            # first to catch a Python Store owner lost behind a C++ PrefixStore.
            gc.collect()
            subgroup = distributed.new_group([0], backend="gloo", timeout=timedelta(seconds=30))
            value = torch.tensor([41017], dtype=torch.int64)
            distributed.all_reduce(value, group=subgroup)
            if value.item() != 41017 or distributed.get_world_size() != 1 or distributed.get_rank() != 0:
                raise RuntimeError("actual local Gloo collective returned an invalid rank/value")
            distributed.destroy_process_group(subgroup)
        finally:
            if distributed.is_initialized():
                distributed.destroy_process_group()
            del store
    return "actual local bounded Torch Store/Gloo group and CPU metadata collective; no inter-PC/GPU collective"


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--bundle", type=Path, required=True)
    options.add_argument("--output", type=Path, required=True)
    options.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = options.parse_args()
    root, output = args.bundle.resolve(strict=True), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "BUNDLE.json").read_text(encoding="utf-8"))
    if sys.platform != "win32" or manifest["kind"] != "runtime":
        raise RuntimeError("full runtime qualification requires the actual native Windows bundle")
    if not args.child:
        from deepseekrpc.triton_policy import compiler_qualification_environment
        env = compiler_qualification_environment(os.environ)
        # No developer toolchain, SDK, user-site package or ambient PATH entry
        # can supply a concealed DLL to the installation check.
        env.update(VLLM_TARGET_DEVICE="empty", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", VLLM_NO_USAGE_STATS="1")
        command = [sys.executable, str(Path(__file__).resolve()), "--bundle", str(root), "--output", str(output), "--child"]
        subprocess.run(command, env=env, cwd=output, check=True)
        return
    report = {"scope": "installed native Windows Python/DLL/API/parser qualification; no visible GPU or compiler SDK",
        "source_commit": manifest["source_commit"], "python": platform.python_version(),
        "CUDA_execution": "not run", "full_model_validation": "not run", "checks": [], "native": []}

    def record(name, function):
        value = function()
        report["checks"].append({"name": name, "status": "passed", "result": value})

    def versions():
        from importlib import resources
        lock = json.loads(resources.files("deepseekrpc._vendor.vllm_windows").joinpath("RUNTIME.json").read_text())
        for name, version in lock["versions"].items():
            if metadata.version(name) != version:
                raise RuntimeError(f"installed runtime version differs: {name}")
        for record in lock["files"]:
            path = Path(metadata.distribution(record["distribution"]).locate_file(record["path"]))
            if not path.is_relative_to(Path(sys.prefix)) or path.stat().st_size > 4 * 1024 ** 2:
                raise RuntimeError("runtime source is outside the isolated environment or exceeds its bound")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                    raise RuntimeError("installed runtime Python source differs from its admitted exact patch")
        if sys.version_info[:3] != (3, 12, 14):
            raise RuntimeError("runtime interpreter differs from its pinned tested bootstrap")
        return lock["versions"]

    def source_identity():
        from deepseekrpc.runtime_identity import application_source_identity
        actual = application_source_identity()
        if actual != manifest["application_source_identity"]:
            raise RuntimeError("installed application source differs from the exact bundled wheel")
        return actual

    def native_imports():
        from deepseekrpc.runtime_policy import configure_exact_cuda_math
        configure_exact_cuda_math()
        for relative in manifest["app_local_crt"]:
            expected = manifest["files"][relative]["sha256"]
            for directory in (Path(sys.base_prefix), Path(sys.executable).parent):
                path = directory / Path(relative).name
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise RuntimeError("private app-local MSVC runtime differs from its admitted bundle")
        for record in manifest["native_files"]:
            path = Path(metadata.distribution(record["distribution"]).locate_file(record["path"]))
            if not path.is_relative_to(Path(sys.prefix)):
                raise RuntimeError("native library is outside the isolated runtime")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                    raise RuntimeError("native library differs from the admitted built wheel")
        import torch
        if torch.__version__ != "2.11.0+cu130" or torch.cuda.is_available():
            raise RuntimeError("device-independent runtime check must use pinned CUDA Torch with all GPUs hidden")
        import deepseekrpc_cuda
        import exllamav3_ext
        if deepseekrpc_cuda.abi_version() != 2 or not callable(exllamav3_ext.exl3_gemm):
            raise RuntimeError("actual native ABI or EXL3 operation is absent")
        distribution = metadata.distribution("vllm")
        paths = [distribution.locate_file(path) for path in distribution.files or ()
                 if str(path).startswith("vllm/_C_stable_libtorch") and str(path).endswith(".pyd")]
        if len(paths) != 1:
            raise RuntimeError("actual compiled vLLM core is missing or duplicated")
        torch.ops.load_library(str(paths[0]))
        if not callable(torch.ops._C.rms_norm):
            raise RuntimeError("compiled vLLM RMSNorm registration is absent")
        moe_paths = [distribution.locate_file(path) for path in distribution.files or ()
                     if str(path).startswith("vllm/_moe_C_stable_libtorch") and str(path).endswith(".pyd")]
        if len(moe_paths) != 1:
            raise RuntimeError("actual compiled vLLM MoE/router library is missing or duplicated")
        torch.ops.load_library(str(moe_paths[0]))
        if not callable(torch.ops._moe_C.topk_softmax):
            raise RuntimeError("compiled vLLM router registration is absent")
        for operation in ("cutlass_scaled_mm_supports_fp8", "cutlass_scaled_mm_supports_block_fp8", "cutlass_scaled_mm_supports_fp4"):
            if getattr(torch.ops._C, operation)(120):
                raise RuntimeError("installed native vLLM differs from the admitted EXL3 profile")
        for path in [Path(deepseekrpc_cuda.__file__), Path(exllamav3_ext.__file__), *paths, *moe_paths]:
            with path.open("rb") as stream:
                report["native"].append({"file": path.name, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()})
        return "actual native exports/registration imported with a bare Windows system PATH"

    def compiler_checks():
        from deepseekrpc.compiler_selftest import run_compiler_checks
        return run_compiler_checks(output / "triton")

    def python_integration():
        import exllamav3
        import vllm
        import vllm_exl3.exl3 as plugin
        if not plugin._VLLM_AVAILABLE or not plugin._TORCH_AVAILABLE:
            raise RuntimeError("plugin selected a missing-dependency fallback")
        entries = [entry for entry in metadata.entry_points(group="vllm.general_plugins") if entry.name == "vllm_exl3"]
        if len(entries) != 1:
            raise RuntimeError("actual EXL3 plugin entry point is missing or duplicated")
        entries[0].load()()
        from vllm.usage.usage_lib import is_usage_stats_enabled
        if is_usage_stats_enabled():
            raise RuntimeError("actual upstream usage reporting is enabled despite the offline runtime policy")
        from deepseekrpc.vllm_attention import OwnedBF16Attention, OwnedDSparkAttention
        from deepseekrpc.vllm_engram import DiskEngram
        from deepseekrpc.vllm_experts import BoundedMoEMethod
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
        if (not issubclass(BoundedMoEMethod, FusedMoEMethodBase) or BoundedMoEMethod.__abstractmethods__ or OwnedBF16Attention.__abstractmethods__ or
                OwnedDSparkAttention.__abstractmethods__ or DiskEngram.__bases__[0].__name__ != "Engram"):
            raise RuntimeError("owned adapters do not bind their actual concrete upstream classes")
        from deepseekrpc.api import create_app
        from deepseekrpc.api_runtime import APIRuntime
        from fastapi.testclient import TestClient
        with TestClient(create_app(APIRuntime())) as client:
            if client.get("/v1/models").json()["data"][0]["context_length"] != 1048576 or client.get("/readyz").status_code != 503:
                raise RuntimeError("device-independent API startup advertised incorrect readiness/context")
        return "actual installed classes/API, without model construction or GPU qualification"

    try:
        record("versions", versions)
        record("application_source_identity", source_identity)
        record("private_compilers_without_sdk", compiler_checks)
        record("native_without_sdk", native_imports)
        record("actual_local_gloo", lambda: qualify_local_gloo(output))
        record("complete_python_integration", python_integration)
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
        subprocess.run([sys.executable, str(root / "tools/audit_parser_integration.py"),
                        "--output", str(output / "parser.json")], cwd=root, check=True)
        report["checks"].append({"name": "actual_original_parser", "status": "passed"})
    except BaseException as error:
        # Public installation evidence never records a private user path or
        # username from an OS/import exception.
        report["failure"] = {"type": type(error).__name__}
        raise
    finally:
        (output / "runtime.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
