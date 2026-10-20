#!/usr/bin/env python3
"""Assemble real admitted native wheels and a pinned isolated Windows installer.

External dependencies are hash-locked downloads at installation. Model payloads,
deployment credentials and a compiler SDK are never put in the application bundle.
"""
import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

from acquire_references import bounded_download
from build_upstream import extract_source

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def report_wheel(directory, *, component):
    reports = list(directory.rglob("BUILD.json"))
    if len(reports) != 1:
        raise RuntimeError("each admitted component requires one complete native build report")
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    required = {"build", "install", "fatbin", "import-without-sdk"} if component == "native" else {
        "build", "install", "fatbin", "import-native", "import-native-without-sdk"}
    if component == "vllm":
        required.add("fatbin-moe")
    checks = {entry["name"]: entry["exit_code"] for entry in report["checks"]}
    if ("failure" in report or "dependency_provenance_error" in report or report.get("worktree_clean") is not True or
            report.get("binary_architectures_verified") is not True or
            set(report.get("architectures", [])) != {"sm_89", "sm_120"} or not required <= set(checks) or
            any(checks[name] != 0 for name in checks)):
        raise RuntimeError("native component build/import/architecture qualification is incomplete")
    if component != "native":
        sources = json.loads((ROOT / "provenance/sources.lock.json").read_text())["sources"]
        repository = "SystemPanic/vllm-windows" if component == "vllm" else "vcruz305/exllamav3"
        expected, = [record for record in sources if record["repository"] == repository]
        for key in ("repository", "commit", "archive_sha256", "archive_bytes"):
            if report["upstream"][key] != expected[key]:
                raise RuntimeError("native component was not built from the current pinned upstream source")
        if component == "vllm":
            dependencies = report.get("native_dependencies")
            expected_dependencies = {item["component"]: item["commit"] for item in json.loads(
                (ROOT / "provenance/vllm-native-deps.lock.json").read_text())["dependencies"]}
            if (not isinstance(dependencies, list) or not dependencies or
                    any(item.get("matches_lock") is not True or item.get("commit") != expected_dependencies.get(item.get("component"))
                        for item in dependencies)):
                raise RuntimeError("vLLM native dependency commits differ from their admitted source pins")
            locks = ["vllm-windows-mhc.lock.json", "vllm-windows-exl3-build.lock.json", "vllm-windows-stage-hooks.lock.json",
                     "vllm-windows-float-infinity.lock.json", "vllm-windows-owned-runtime.lock.json"]
            # The first patch's actual filename is resolved below from the
            # repository's sole mhc patch lock, without trusting a report path.
            if not (ROOT / "patches" / locks[0]).exists():
                choices = [p.name for p in (ROOT / "patches").glob("*.lock.json")
                           if json.loads(p.read_text()).get("patch", "").endswith("mhc.patch")]
                if len(choices) != 1:
                    raise RuntimeError("current mHC patch lock is absent or ambiguous")
                locks[0] = choices[0]
            expected_patches = [json.loads((ROOT / "patches" / name).read_text())["patch_sha256"] for name in locks]
            if [patch["sha256"] for patch in report["patches"]] != expected_patches:
                raise RuntimeError("vLLM native component differs from the exact five-patch owned runtime")
    else:
        commit = report["source_commit"]
        found = subprocess.run(["git", "cat-file", "-e", f"{commit}:native"], cwd=ROOT, capture_output=True)
        if found.returncode:
            subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", commit], cwd=ROOT, check=True)
        old = subprocess.check_output(["git", "rev-parse", f"{commit}:native"], cwd=ROOT, text=True).strip()
        current = subprocess.check_output(["git", "rev-parse", "HEAD:native"], cwd=ROOT, text=True).strip()
        if old != current:
            raise RuntimeError("project native wheel predates a change to the actual native source tree")
    wheel = reports[0].parent / "wheels" / report["wheel"]["name"]
    if not wheel.is_file() or digest(wheel) != report["wheel"]["sha256"]:
        raise RuntimeError("native wheel failed its retained build report checksum")
    return wheel, report


def wheel_inventory(wheel):
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise RuntimeError("bundle wheel has invalid distribution metadata")
        info = BytesParser().parsebytes(archive.read(metadata_files[0]))
        native = []
        entries = {}
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename or member.file_size > 4 * 1024 ** 3:
                raise RuntimeError("bundle wheel contains an unsafe or oversized member")
            if name.suffix in {".pyd", ".dll"}:
                with archive.open(member) as stream:
                    native.append({"distribution": info["Name"], "path": name.as_posix(),
                                   "sha256": hashlib.file_digest(stream, "sha256").hexdigest()})
            if name.parts and name.parts[0] == "deepseekrpc" and (name.suffix in {".py", ".json"} or name.name == "LICENSE"):
                if member.file_size > 4 * 1024 ** 2:
                    raise RuntimeError("application identity member exceeds its source bound")
                entries[PurePosixPath(*name.parts[1:]).as_posix()] = hashlib.sha256(archive.read(member)).hexdigest()
        application = None
        if entries:
            encoded = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            application = {"sha256": hashlib.sha256(encoded).hexdigest(), "files": len(entries)}
        return info["Name"], info["Version"], native, application


def main():
    options = argparse.ArgumentParser(description=__doc__)
    for name in ("vllm", "exllama", "native", "work", "output"):
        options.add_argument(f"--{name}", type=Path, required=True)
    args = options.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("a reproducible application bundle requires a clean committed source tree")
    output, work = args.output.resolve(), args.work.resolve()
    output.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=True)
    wheels = output / "wheels"
    wheels.mkdir()
    reports, admitted = {}, []
    redist = os.environ.get("VCToolsRedistDir")
    if os.name != "nt" or not redist:
        raise RuntimeError("the Windows bundle requires the actual MSVC redistributable runtime directory")
    candidates = [path for path in (Path(redist) / "x64").glob("Microsoft.VC14*.CRT") if path.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError("MSVC app-local CRT directory is absent or ambiguous")
    crt = output / "crt"
    crt.mkdir()
    for path in sorted(candidates[0].glob("*.dll")):
        shutil.copyfile(path, crt / path.name)
    if not {"msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"} <= {path.name.lower() for path in crt.iterdir()}:
        raise RuntimeError("the actual required MSVC app-local runtime inventory is incomplete")
    (crt / "NOTICE.txt").write_text(
        "Microsoft Visual C++ Runtime app-local redistributable files.\n"
        "These system runtime libraries remain subject to Microsoft's redistribution terms.\n"
        "https://learn.microsoft.com/cpp/windows/redistributing-visual-cpp-files\n"
        "https://learn.microsoft.com/visualstudio/releases/2022/redistribution\n"
        f"Source MSVC redist version: {Path(redist).name}\n", encoding="utf-8")
    for name in ("vllm", "exllama", "native"):
        wheel, report = report_wheel(getattr(args, name).resolve(strict=True), component=name)
        shutil.copyfile(wheel, wheels / wheel.name)
        reports[name] = report
        admitted.append(wheels / wheel.name)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    build_wheels = work / "application-wheels"
    build_wheels.mkdir(exist_ok=False)
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(build_wheels)], cwd=ROOT, check=True)
    sources = json.loads((ROOT / "provenance/sources.lock.json").read_text())["sources"]
    plugin, = [record for record in sources if record["repository"] == "vcruz305/vllm-exl3"]
    archive = work / "vllm-exl3.tar.gz"
    bounded_download(plugin["url"], archive, expected_size=plugin["archive_bytes"], sha256=plugin["archive_sha256"], limit=8 * 1024 ** 2)
    plugin_source = work / "vllm-exl3"
    extract_source(archive, plugin_source)
    env = dict(os.environ, VLLM_EXL3_NO_CUDA="1")
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(build_wheels),
                    str(plugin_source)], cwd=work, env=env, check=True)
    for wheel in sorted(build_wheels.glob("*.whl")):
        if not wheel.name.startswith(("deepseekrpc-", "vllm_exl3-")):
            raise RuntimeError("application wheel build produced an unexpected distribution")
        shutil.copyfile(wheel, wheels / wheel.name)
        admitted.append(wheels / wheel.name)
    if len(admitted) != 5:
        raise RuntimeError("complete bundle requires exactly three native and two application/plugin wheels")
    native_files, application_identity = [], None
    for wheel in admitted:
        name, version, natives, identity = wheel_inventory(wheel)
        native_files.extend(natives)
        if identity is not None:
            if application_identity is not None:
                raise RuntimeError("application source distribution is duplicated")
            application_identity = identity
    for relative in ["LICENSE", "THIRD_PARTY_NOTICES.md", "requirements-runtime-windows.lock",
            "provenance/windows-bootstrap.lock.json", "provenance/python-windows-download.json",
            "provenance/token-map.lock.json", "tools/qualify_runtime.py", "tools/audit_parser_integration.py",
            "provenance/sources.lock.json", "provenance/vllm-native-deps.lock.json",
            "provenance/opencode-client.lock.json", "provenance/deepinfer-review.lock.json", "provenance/compiler-elf-reference.lock.json",
            "provenance/torch-store-reference.lock.json",
            "docs/DEEPINFER_REVIEW.md", "tests/fixtures/tokenizer.json.gz", "docs/SESSIONS.md",
            "docs/INSTALL_WINDOWS.md", "docs/DIAGNOSTICS.md", "docs/OBSERVABILITY.md", "docs/OPENCODE.md", "docs/REVIEW_GUIDE.md"]:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    shutil.copyfile(ROOT / "packaging/Install.ps1", output / "Install.ps1")
    shutil.copytree(ROOT / "patches", output / "patches")
    (output / "SOURCE.md").write_text(
        "# Corresponding source\n\n"
        f"Application, native kernels, tests and build scripts: https://github.com/Protonburn35/DeepSeekRPC/tree/{source_commit}\n\n"
        "Exact upstream revisions, source archive SHA-256 values and licenses are in provenance/sources.lock.json. "
        "All local upstream changes are supplied in patches/ with exact before/after hashes. "
        "Component native build reports are in evidence/. The wheels retain their distribution licenses.\n\n"
        "The application and EXL3 plugin are AGPL-3.0-only. Make this exact corresponding source available "
        "to users of the network service under that license. Other incorporated source retains its "
        "Apache-2.0/MIT notices. Model files and Microsoft's app-local CRT have their own terms.\n", encoding="utf-8")
    for directory, names in {"config": ["cluster.example.json", "A.example.json", "B.example.json", "C.example.json",
            "benchmark.example.json", "opencode.deepseek.example.json"],
            "launchers": ["RUN-HOST.bat", "RUN-MIDDLE.bat", "RUN-STAGE-C.bat", "Run-Stage.ps1"]}.items():
        (output / directory).mkdir()
        for name in names:
            shutil.copyfile(ROOT / directory / name, output / directory / name)
    evidence = output / "evidence"
    evidence.mkdir()
    for name, report in reports.items():
        (evidence / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (evidence / "plugin.json").write_text(json.dumps({"source": {key: plugin[key] for key in
        ("repository", "commit", "archive_sha256", "archive_bytes")}, "build_mode": "official VLLM_EXL3_NO_CUDA=1; actual EXL3 native execution is supplied by the owned executor"}, indent=2) + "\n")
    (output / "wheels.lock").write_text("\n".join(f"./wheels/{wheel.name} --hash=sha256:{digest(wheel)}" for wheel in sorted(admitted)) + "\n", encoding="utf-8")
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            files[path.relative_to(output).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    manifest = {"format": 1, "kind": "runtime", "source_commit": source_commit, "files": files,
        "application_source_identity": application_identity, "native_files": native_files,
        "architectures": ["sm_89", "sm_120"], "protocol_abi": 1, "snapshot_abi": 1, "native_abi": 2,
        "app_local_crt": [path.relative_to(output).as_posix() for path in sorted(crt.glob("*.dll"))],
        "scope": "Complete application/native wheel bundle; installed Python/DLL qualification must run before install completion",
        "GPU_execution": "not run", "full_model_validation": "not run"}
    (output / "BUNDLE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_commit": source_commit, "wheels": [wheel.name for wheel in admitted],
                      "application_source_identity": application_identity}, indent=2))


if __name__ == "__main__":
    main()
