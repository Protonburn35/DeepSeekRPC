#!/usr/bin/env python3
"""CPU foundation evidence, never a CUDA/full-model qualification command."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir())) / "deepseekrpc-check"
    output.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=root, text=True))
    report = {"source_commit": source, "worktree_clean": not dirty, "platform": platform.platform(),
              "python": sys.version, "checks": [], "CUDA_execution": "not run",
              "full_model_validation": "not run", "release_candidate": False}
    def run(name, command, *, cwd=root, env=None):
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        report["checks"].append({"name": name, "command": command, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")
    try:
        import torch
        if torch.version.cuda is not None:
            raise RuntimeError("CPU reference job must use the explicit CPU-only Torch wheel")
        run("pytest", [sys.executable, "-m", "pytest", "-q", f"--junitxml={output / 'pytest.xml'}"])
        wheels = output / "wheels"
        wheels.mkdir(exist_ok=True)
        # Remove only prior regenerable wheels in this dedicated CI-output dir.
        for wheel in wheels.glob("deepseekrpc-*.whl"):
            wheel.unlink()
        run("wheel", [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels), str(root)])
        artifacts = list(wheels.glob("deepseekrpc-*.whl"))
        if len(artifacts) != 1:
            raise RuntimeError("expected exactly one wheel from the current build")
        wheel = artifacts[0]
        report["wheel"] = {"name": wheel.name, "sha256": hashlib.file_digest(wheel.open("rb"), "sha256").hexdigest()}
        run("install-wheel", [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)])
        installed_env = dict(os.environ)
        installed_env.pop("PYTHONPATH", None)
        command = [sys.executable, "-c", "import deepseekrpc; print(deepseekrpc.__file__)"]
        location = Path(subprocess.check_output(command, cwd=output, env=installed_env, text=True).strip()).resolve()
        if location.is_relative_to(root / "src"):
            raise RuntimeError("installed-package check accidentally imported checkout source")
        entry = Path(sys.executable).parent / ("deepseekrpc.exe" if os.name == "nt" else "deepseekrpc")
        run("installed-cli", [str(entry), "capabilities"], cwd=output, env=installed_env)
        capabilities = json.loads((output / "installed-cli.log").read_text(encoding="utf-8"))
        if capabilities["release_candidate"] is not False:
            raise RuntimeError("foundation wheel incorrectly advertises a release candidate")
    finally:
        (output / "BUILD.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
