#!/usr/bin/env python3
"""Build the exact pinned EXL3 runtime on Windows, without a GPU or model files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

from acquire_references import bounded_download, safe_relative


def extract_source(archive: Path, destination: Path):
    """Extract regular source files into a new writable build directory."""
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        roots = {safe_relative(m.name).parts[0] for m in members}
        if len(roots) != 1 or sum(m.size for m in members) > 1024 ** 3:
            raise ValueError("unexpected source archive root or extraction size")
        prefix = roots.pop()
        # All three selected source archives have no links. Reject them rather
        # than widening this build helper's extraction contract implicitly.
        for member in members:
            safe_relative(member.name).relative_to(prefix)
            if not member.isfile() and not member.isdir():
                raise ValueError("source build archive contains a non-regular entry")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            relative = safe_relative(member.name).relative_to(prefix)
            path = destination / relative
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open("xb") as target:
                while data := source.read(1024 * 1024):
                    target.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("this qualification requires native Windows/MSVC")
    root = Path(__file__).resolve().parents[1]
    work, output, sdk = args.work.resolve(), args.output.resolve(), args.sdk.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads((root / "provenance/sources.lock.json").read_text(encoding="utf-8"))["sources"]
    record, = [r for r in records if r["repository"] == "vcruz305/exllamav3"]
    env = dict(os.environ, CUDA_HOME=str(sdk), CUDA_PATH=str(sdk), MAX_JOBS="2",
               TORCH_CUDA_ARCH_LIST="8.9;12.0", DISTUTILS_USE_SDK="1",
               EXLLAMA_EXT_COMPRESS="require", EXL3_INT8_GEMV="0", EXL3_GR_INT8="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("EXLLAMA_NOCOMPILE", None)
    env["PATH"] = os.pathsep.join([str(sdk / "bin"), str(sdk / "bin/x64"), env["PATH"]])
    report = {"scope": "pinned EXLlama Windows native compile/link/import; no GPU execution",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "worktree_clean": not bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
              "upstream": {k: record[k] for k in ("repository", "commit", "url", "archive_sha256", "archive_bytes")},
              "patches": [], "platform": platform.platform(), "architectures": ["sm_89", "sm_120"],
              "checks": [], "GPU_execution": "not run", "full_model_validation": "not run"}

    def run(name, command, *, cwd=root):
        with (output / f"{name}.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report["checks"].append({"name": name, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed; see retained {name}.log")

    try:
        archive = work / "archives" / "exllamav3.tar.gz"
        bounded_download(record["url"], archive, expected_size=record["archive_bytes"],
                         sha256=record["archive_sha256"], limit=64 * 1024 * 1024)
        source = work / "exllamav3"
        extract_source(archive, source)
        run("torch", [sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda); assert torch.__version__ == '2.11.0+cu130'"])
        wheels = output / "wheels"
        wheels.mkdir(exist_ok=True)
        run("build", [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-v", "--wheel-dir", str(wheels), str(source)], cwd=source)
        wheel, = wheels.glob("exllamav3-*.whl")
        with wheel.open("rb") as stream:
            report["wheel"] = {"name": wheel.name, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
        run("install", [sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)])
        # Import outside the upstream source, against the real installed wheel.
        query = "import torch, exllamav3_ext as e; assert callable(e.exl3_gemm); print(e.__file__)"
        run("import-native", [sys.executable, "-c", query], cwd=output)
        module = Path((output / "import-native.log").read_text(encoding="utf-8").strip().splitlines()[-1])
        run("fatbin", [str(sdk / "bin/cuobjdump.exe"), "--list-elf", str(module)])
        listing = (output / "fatbin.log").read_text(encoding="utf-8")
        for architecture in ("sm_89", "sm_120"):
            if architecture not in listing:
                raise RuntimeError(f"compiled EXLlama is missing target {architecture}")
        report["binary_architectures_verified"] = True
        query = "import exllamav3; import exllamav3.modules.quant.exl3 as q; assert callable(q.LinearEXL3); print(exllamav3.__file__)"
        run("import-package", [sys.executable, "-c", query], cwd=output)
        run("dependency-check", [sys.executable, "-m", "pip", "check"], cwd=output)
    finally:
        (output / "BUILD.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
