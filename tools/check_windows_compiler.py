#!/usr/bin/env python3
"""Qualify pinned standalone Python/Triton compilers on a CPU-only Windows VM.

The project/Triton compiler subset runs first. Then the complete external
dependency lock and actual CUDA-Torch CPU startup are checked independently of
the still-required native vLLM/EXL3/project wheels. No namespace stubs are used.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from acquire_references import bounded_download


def main():
    if sys.platform != "win32":
        raise RuntimeError("the compiler check requires actual native Windows")
    root = Path(__file__).resolve().parents[1]
    work = Path(os.environ["RUNNER_TEMP"]) / "deepseekrpc-compiler-check"
    work.mkdir(exist_ok=False)
    report = {"scope": "actual isolated Windows compilers and external dependency stack; native application runtime not qualified",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "GPU_execution": "not run", "checks": []}

    def run(name, command, **kwargs):
        with (work / f"{name}.log").open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
        report["checks"].append({"name": name, "exit_code": process.returncode})
        if process.returncode:
            raise RuntimeError(f"actual Windows compiler check failed: {name}")

    try:
        run("bootstrap", [sys.executable, str(root / "tools/check_windows_bootstrap.py")], cwd=root)
        bundle = Path(os.environ["RUNNER_TEMP"]) / "deepseekrpc-bootstrap-check/bundle"
        python, uv = bundle / "runtime/Scripts/python.exe", bundle / ".bootstrap/uv.exe"
        wheels = work / "wheels"
        wheels.mkdir()
        run("wheel", [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(wheels)], cwd=root)
        wheel, = wheels.glob("deepseekrpc-*.whl")
        triton = wheels / "triton_windows-3.6.0.post26-cp312-cp312-win_amd64.whl"
        lock = json.loads((root / "src/deepseekrpc/_vendor/vllm_windows/TRITON.json").read_text())
        bounded_download("https://files.pythonhosted.org/packages/62/1e/8d9814e67ba3f20094cf3c69e7815a491f20beb86469a647550ba86728b0/" + triton.name,
            triton, expected_size=47402104, sha256=lock["wheel_sha256"], limit=64 * 1024 ** 2)
        run("install-subset", [str(uv), "--no-config", "pip", "install", "--python", str(python), "--no-index", "--no-deps",
            str(wheel), str(triton)], cwd=work)
        # Read the actual installed policy using -I, then strip every inherited
        # SDK/compiler selector before the new process imports any Triton code.
        child = r'''
import json, os, subprocess, sys
from pathlib import Path
from deepseekrpc.triton_policy import compiler_qualification_environment
output = Path(sys.argv[1])
code = "from deepseekrpc.compiler_selftest import run_compiler_checks; import json,sys; from pathlib import Path; output=Path(sys.argv[1]); output.write_text(json.dumps(run_compiler_checks(output.parent/'triton'),indent=2),encoding='utf-8')"
subprocess.run([sys.executable, '-I', '-c', code, str(output)],
    env=compiler_qualification_environment(os.environ), cwd=output.parent, check=True)
'''
        run("compilers-without-sdk", [str(python), "-I", "-c", child, str(work / "compilers.json")], cwd=work)
        compiled = json.loads((work / "compilers.json").read_text())
        if {item["architecture"] for item in compiled["targets"]} != {"sm_89", "sm_120"}:
            raise AssertionError("both actual device compiler targets were not qualified")
        report["compilers"] = compiled
        run("install-external-runtime", [str(uv), "--no-config", "pip", "install", "--python", str(python),
            "--require-hashes", "--only-binary", ":all:", "--torch-backend", "cu130",
            "--default-index", "https://pypi.org/simple", "-r", str(root / "requirements-runtime-windows.lock")], cwd=work)
        run("external-pip-check", [str(python), "-I", "-m", "pip", "check"], cwd=work)
        external_child = r'''
import importlib.util, json, os, subprocess, sys
from pathlib import Path
from deepseekrpc.triton_policy import compiler_qualification_environment
code = r"""
import importlib.util, json, sys
from pathlib import Path
from deepseekrpc.triton_policy import configure_windows_triton
configure_windows_triton(Path(sys.argv[1]).parent/'triton-external')
import torch, torchvision, torchaudio, tvm_ffi, winloop, portalocker
from importlib.metadata import version
from deepseekrpc.runtime_policy import configure_exact_cuda_math
configure_exact_cuda_math()
if sys.version_info[:3] != (3,12,14) or torch.__version__ != '2.11.0+cu130' or torch.cuda.is_available():
    raise RuntimeError('external qualification needs pinned private CUDA Torch with GPUs hidden')
output, tool_path = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location('actual_runtime_qualification_tool', tool_path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
gloo = tool.qualify_local_gloo(output.parent)
from deepseekrpc.api import create_app
from deepseekrpc.api_runtime import APIRuntime
from fastapi.testclient import TestClient
with TestClient(create_app(APIRuntime())) as client:
    if client.get('/readyz').status_code != 503 or client.get('/v1/models').json()['data'][0]['context_length'] != 1048576:
        raise RuntimeError('external-stack API startup advertised incorrect readiness/context')
report = {'scope':'actual private external dependencies, SDK-free Torch imports/Gloo/API; native vLLM/EXL3/project wheels not installed',
    'GPU_execution':'not run', 'full_model_validation':'not run', 'checks':[
        {'name':'actual_cuda_torch_vision_audio_tvm_imports_without_sdk','status':'passed'},
        {'name':'actual_local_gloo_subgroup_lifetime','status':'passed','result':gloo},
        {'name':'actual_unloaded_api_startup','status':'passed'}],
    'versions':{name:version(name) for name in ('torch','torchvision','torchaudio','triton-windows','apache-tvm-ffi','winloop','portalocker')}}
output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
"""
subprocess.run([sys.executable, '-I', '-c', code, sys.argv[1], sys.argv[2]],
    env=compiler_qualification_environment(os.environ), cwd=Path(sys.argv[1]).parent, check=True)
'''
        run("external-runtime-without-sdk", [str(python), "-I", "-c", external_child,
            str(work / "external-runtime.json"), str(root / "tools/qualify_runtime.py")], cwd=work)
        report["external_runtime"] = json.loads((work / "external-runtime.json").read_text())
        with wheel.open("rb") as stream:
            report["application_wheel_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__}
        raise
    finally:
        (work / "BUILD.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
