#!/usr/bin/env python3
"""Build exact pinned upstream native wheels on Windows without model files."""
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
        omitted = []
        helper_links = {"ci-fails-buildkite", "debug-ima", "kernel-microbenchmark", "triton-kernel-writing"}
        for member in members:
            relative = safe_relative(member.name).relative_to(prefix)
            if (member.issym() and relative.parent.as_posix() == ".claude/skills" and
                    relative.name in helper_links and member.linkname == f"../../.agents/skills/{relative.name}"):
                omitted.append(member.name)
                continue
            if not member.isfile() and not member.isdir():
                raise ValueError("source build archive contains a non-regular entry")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            if member.name in omitted:
                continue
            relative = safe_relative(member.name).relative_to(prefix)
            path = destination / relative
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open("xb") as target:
                while data := source.read(1024 * 1024):
                    target.write(data)
        # Windows extraction needs no symlink privilege. Only these exact
        # upstream agent-helper aliases are omitted; no runtime/build file is.
        return [str(safe_relative(name).relative_to(prefix)) for name in omitted]


def dependency_provenance(source: Path, lock: dict):
    observed = []
    for item in lock["dependencies"]:
        directory = source / ".deps" / item["directory"]
        if not (directory / ".git").exists():
            continue
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=directory)
        observed.append({"component": item["component"], "commit": commit,
                         "expected_commit": item["commit"], "matches_lock": commit == item["commit"],
                         "upstream_build_patch_sha256": hashlib.sha256(diff).hexdigest(),
                         "submodules": subprocess.check_output(["git", "submodule", "status", "--recursive"], cwd=directory, text=True).splitlines()})
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=["exllamav3", "vllm-windows"], default="exllamav3")
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
    repository = {"exllamav3": "vcruz305/exllamav3", "vllm-windows": "SystemPanic/vllm-windows"}[args.component]
    record, = [r for r in records if r["repository"] == repository]
    env = dict(os.environ, CUDA_HOME=str(sdk), CUDA_PATH=str(sdk), MAX_JOBS="2",
               TORCH_CUDA_ARCH_LIST="8.9;12.0", DISTUTILS_USE_SDK="1",
               EXLLAMA_EXT_COMPRESS="require", EXL3_INT8_GEMV="0", EXL3_GR_INT8="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("EXLLAMA_NOCOMPILE", None)
    env["PATH"] = os.pathsep.join([str(sdk / "bin"), str(sdk / "bin/x64"), env["PATH"]])
    if args.component == "vllm-windows":
        env.update(VLLM_TARGET_DEVICE="cuda", VLLM_USE_PRECOMPILED="0", VLLM_USE_PRECOMPILED_RUST="0",
                   SETUPTOOLS_SCM_PRETEND_VERSION="0.31.0+deepseekrpc2", NVCC_THREADS="1",
                   CMAKE_ARGS="-DVLLM_DEEPSEEKRPC_EXL3=ON")
    report = {"scope": f"pinned {args.component} Windows native compile/link/import; no GPU execution",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "worktree_clean": not bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
              "upstream": {k: record[k] for k in ("repository", "commit", "url", "archive_sha256", "archive_bytes")},
              "patches": [], "platform": platform.platform(), "architectures": ["sm_89", "sm_120"],
              "checks": [], "GPU_execution": "not run", "full_model_validation": "not run"}
    dependency_lock = None

    def run(name, command, *, cwd=root, execution_env=None):
        with (output / f"{name}.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=cwd, env=env if execution_env is None else execution_env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        report["checks"].append({"name": name, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed; see retained {name}.log")

    try:
        archive = work / "archives" / f"{args.component}.tar.gz"
        bounded_download(record["url"], archive, expected_size=record["archive_bytes"],
                         sha256=record["archive_sha256"], limit=64 * 1024 * 1024)
        source = work / args.component
        report["omitted_nonbuild_agent_links"] = extract_source(archive, source)
        if args.component == "vllm-windows":
            from patch_vllm_for_rpc import apply
            report["patches"] = [apply(source), apply(source, "vllm-windows-exl3-build.lock.json"),
                                 apply(source, "vllm-windows-stage-hooks.lock.json"),
                                 apply(source, "vllm-windows-float-infinity.lock.json")]
            report["native_build_profile"] = "Windows DeepSeekRPC EXL3; SM12x FP8/NVFP4 GEMMs excluded"
            dependency_lock = json.loads((root / "provenance/vllm-native-deps.lock.json").read_text(encoding="utf-8"))
            for item in dependency_lock["dependencies"]:
                if "tag" in item:
                    lines = subprocess.check_output(["git", "ls-remote", item["repository"], f"refs/tags/{item['tag']}"], text=True).splitlines()
                    if lines != [f"{item['commit']}\trefs/tags/{item['tag']}"]:
                        raise RuntimeError(f"upstream dependency tag differs from its admitted commit: {item['component']}")
        run("torch", [sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda); assert torch.__version__ == '2.11.0+cu130'"])
        wheels = output / "wheels"
        wheels.mkdir(exist_ok=True)
        run("build", [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-v", "--wheel-dir", str(wheels), str(source)], cwd=source)
        if dependency_lock is not None:
            observed = dependency_provenance(source, dependency_lock)
            if any(not item["matches_lock"] for item in observed):
                raise RuntimeError("vLLM fetched a native dependency outside the admitted lock")
        wheel, = wheels.glob("exllamav3-*.whl" if args.component == "exllamav3" else "vllm-*.whl")
        with wheel.open("rb") as stream:
            report["wheel"] = {"name": wheel.name, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
        run("install", [sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)])
        # Import outside the upstream source, against the real installed wheel.
        if args.component == "exllamav3":
            query = "import torch, exllamav3_ext as e; assert callable(e.exl3_gemm); print(e.__file__)"
        else:
            # Load the real compiled torch library without pretending that the
            # entire serving dependency stack was installed for this build job.
            query = ("import torch; from importlib.metadata import distribution; "
                     "d=distribution('vllm'); p,=[d.locate_file(f) for f in d.files "
                     "if str(f).startswith('vllm/_C_stable_libtorch') and str(f).endswith('.pyd')]; "
                     "torch.ops.load_library(str(p)); assert callable(torch.ops._C.rms_norm); "
                     "assert not torch.ops._C.cutlass_scaled_mm_supports_fp8(120); "
                     "assert not torch.ops._C.cutlass_scaled_mm_supports_block_fp8(120); "
                     "assert not torch.ops._C.cutlass_scaled_mm_supports_fp4(120); print(p)")
        run("import-native", [sys.executable, "-c", query], cwd=output)
        module = Path((output / "import-native.log").read_text(encoding="utf-8").strip().splitlines()[-1])
        run("fatbin", [str(sdk / "bin/cuobjdump.exe"), "--list-elf", str(module)])
        listing = (output / "fatbin.log").read_text(encoding="utf-8")
        for architecture in ("sm_89", "sm_120"):
            if architecture not in listing:
                raise RuntimeError(f"compiled upstream module is missing target {architecture}")
        report["binary_architectures_verified"] = True
        runtime_env = dict(env)
        for key in list(runtime_env):
            if key.upper() in {"CUDA_HOME", "CUDA_PATH"} or key.upper().startswith("CUDA_PATH_V"):
                runtime_env.pop(key)
        runtime_env["PATH"] = os.pathsep.join(
            entry for entry in env["PATH"].split(os.pathsep)
            if entry and not Path(entry).resolve().is_relative_to(sdk))
        # A compiler-SDK DLL directory must not conceal a missing runtime
        # dependency in an otherwise installable Torch-based deployment.
        run("import-native-without-sdk", [sys.executable, "-c", query], cwd=output, execution_env=runtime_env)
        if args.component == "exllamav3":
            query = "import exllamav3; import exllamav3.modules.quant.exl3 as q; assert callable(q.LinearEXL3); print(exllamav3.__file__)"
            run("import-package", [sys.executable, "-c", query], cwd=output, execution_env=runtime_env)
            run("dependency-check", [sys.executable, "-m", "pip", "check"], cwd=output)
        else:
            report["complete_serving_dependencies"] = "not qualified by native-only build"
    except BaseException as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if dependency_lock is not None:
            try:
                report["native_dependencies"] = dependency_provenance(source, dependency_lock)
            except (OSError, subprocess.SubprocessError) as exc:
                report["dependency_provenance_error"] = repr(exc)
        (output / "BUILD.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
