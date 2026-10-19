#!/usr/bin/env python3
"""Compile/link and inspect both native CUDA targets on a CPU Windows runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("native Windows compilation requires the native Windows builder")
    root = Path(__file__).resolve().parents[1]
    output, sdk = args.output.resolve(), args.sdk.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_PATH=str(sdk), CUDA_HOME=str(sdk), MAX_JOBS="2",
               TORCH_CUDA_ARCH_LIST="8.9;12.0", DISTUTILS_USE_SDK="1")
    env["PATH"] = os.pathsep.join([str(sdk / "bin"), str(sdk / "bin/x64"), env["PATH"]])
    report = {"scope": "native Windows CUDA compile/link/import only; no GPU execution",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "worktree_clean": not bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
              "platform": platform.platform(), "architectures": ["sm_89", "sm_120"],
              "checks": [], "GPU_execution": "not run", "full_model_validation": "not run"}

    def run(name, command):
        with (output / f"{name}.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report["checks"].append({"name": name, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed; see retained {name}.log")

    try:
        run("compiler", [str(sdk / "bin/nvcc.exe"), "--version"])
        run("torch", [sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda); assert torch.__version__ == '2.11.0+cu130'"])
        wheels = output / "wheels"
        wheels.mkdir(exist_ok=True)
        run("build", [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheels), str(root / "native")])
        wheel, = wheels.glob("deepseekrpc_cuda-*.whl")
        report["wheel"] = {"name": wheel.name, "sha256": hashlib.file_digest(wheel.open("rb"), "sha256").hexdigest()}
        run("install", [sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)])
        query = "import torch, deepseekrpc_cuda as c; assert c.abi_version() == 1; print(c.__file__)"
        run("import", [sys.executable, "-c", query])
        module = Path((output / "import.log").read_text(encoding="utf-8").strip().splitlines()[-1])
        run("fatbin", [str(sdk / "bin/cuobjdump.exe"), "--list-elf", str(module)])
        listing = (output / "fatbin.log").read_text(encoding="utf-8")
        for architecture in ("sm_89", "sm_120"):
            if architecture not in listing:
                raise RuntimeError(f"compiled module is missing target {architecture}")
        report["binary_architectures_verified"] = True
    finally:
        (output / "BUILD.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
