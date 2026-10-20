#!/usr/bin/env python3
"""Archive exact source and real matching Windows evidence for independent review.

This never fabricates reviewer approval or GPU validation. It requires a clean
commit and successful CPU/full installed-runtime reports for that exact commit.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 ** 2:
        raise ValueError("review evidence must be an explicit bounded regular JSON file")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validated_evidence(bundle, cpu_path, commit):
    cpu, manifest = read_json(cpu_path), read_json(bundle / "BUNDLE.json")
    if (cpu.get("source_commit") != commit or cpu.get("worktree_clean") is not True or
            not str(cpu.get("platform", "")).startswith("Windows-") or not cpu.get("checks") or
            any(item.get("exit_code") != 0 for item in cpu["checks"]) or
            not {"pytest", "workflow-syntax", "wheel", "install-wheel", "installed-cli"}
                <= {item["name"] for item in cpu["checks"]}):
        raise ValueError("the exact clean source needs successful actual native Windows CPU evidence")
    if (manifest.get("kind") != "runtime" or manifest.get("source_commit") != commit or
            manifest.get("installed_qualification", {}).get("status") != "passed" or
            set(manifest.get("architectures", [])) != {"sm_89", "sm_120"} or
            manifest.get("GPU_execution") != "not run" or manifest.get("full_model_validation") != "not run"):
        raise ValueError("the exact source needs actual complete Windows runtime installation qualification")
    selected = {}
    for name in ("evidence/vllm.json", "evidence/exllama.json", "evidence/native.json", "evidence/plugin.json",
            "evidence/installed-runtime/runtime.json", "evidence/installed-runtime/parser.json"):
        relative = PurePosixPath(name)
        record, path = manifest["files"].get(name), bundle / Path(*relative.parts)
        if (not record or path.is_symlink() or not path.resolve(strict=True).is_relative_to(bundle) or
                path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]):
            raise ValueError("runtime evidence does not match the sealed bundle manifest")
        selected[name] = read_json(path)
    runtime = selected["evidence/installed-runtime/runtime.json"]
    required = {"versions", "application_source_identity", "private_compilers_without_sdk", "native_without_sdk",
        "actual_local_gloo", "complete_python_integration", "actual_original_parser"}
    if ("failure" in runtime or runtime.get("source_commit") != commit or runtime.get("python") != "3.12.14" or
            runtime.get("CUDA_execution") != "not run" or runtime.get("full_model_validation") != "not run" or
            not required <= {item["name"] for item in runtime["checks"]} or
            any(item.get("status") != "passed" for item in runtime["checks"])):
        raise ValueError("complete matching private-runtime qualification has not passed")
    return manifest, selected


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--runtime", type=Path, required=True, help="extracted successful complete runtime artifact")
    options.add_argument("--cpu-report", type=Path, required=True, help="actual same-commit Windows CPU BUILD.json")
    options.add_argument("--output", type=Path, required=True, help="new directory outside the source checkout")
    args = options.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT):
        raise ValueError("commit the complete tested candidate before building a review packet")
    bundle = args.runtime.resolve(strict=True)
    manifest, selected = validated_evidence(bundle, args.cpu_report.resolve(strict=True), commit)
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("the review packet belongs outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    source = output / f"DeepSeekRPC-source-{commit}.zip"
    subprocess.run(["git", "archive", "--format=zip", "--prefix=DeepSeekRPC/", f"--output={source}", commit], cwd=ROOT, check=True)
    with zipfile.ZipFile(source) as archive:
        if (len(archive.infolist()) > 10000 or sum(item.file_size for item in archive.infolist()) > 128 * 1024 ** 2 or
                any(item.filename.endswith((".key", ".pem", ".pfx", ".safetensors", ".whl")) for item in archive.infolist())):
            raise ValueError("source review archive unexpectedly contains payloads, credentials or an excessive inventory")
    evidence = output / "evidence"
    evidence.mkdir()
    shutil.copyfile(args.cpu_report, evidence / "windows-cpu.json")
    (evidence / "BUNDLE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name in selected:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / name, target)
    (output / "REVIEW_REQUEST.md").write_text(
        f"# DeepSeekRPC whole-runtime review at {commit}\n\n"
        "Review the entire source archive using docs/REVIEW_GUIDE.md, the full implementation specification, "
        "DECISIONS.md and REQUIREMENTS.md. Prioritize mandatory omissions, catastrophic runtime failures and "
        "silent numerical/state defects. Cite source locations, explain concrete failure paths and propose "
        "meaningful regressions. Preserve all user overrides, exact replay/sampling and existing tolerances.\n\n"
        "The enclosed reports establish exact-source native Windows CPU and private-runtime build/import/compiler "
        "qualification. They do not establish any GPU/model execution, full-size residency or measured performance. "
        "Read FINDINGS.md and the append-only build ledger; a later passing run does not resolve a captured failure. "
        "Independent review is pending. The user will mediate review/fix cycles before hardware trials.\n",
        encoding="utf-8")
    files = {path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(output.rglob("*")) if path.is_file()}
    report = {"format": 1, "kind": "whole-source independent-review packet", "source_commit": commit,
        "Windows_CPU": "passed", "Windows_installed_runtime": "passed", "GPU_execution": "not run",
        "full_model_validation": "not run", "physical_residency": "not run", "performance": "not measured",
        "independent_review": "pending", "release_candidate": False, "files": files}
    (output / "REVIEW.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
