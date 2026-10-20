#!/usr/bin/env python3
"""Seal actual installed runtime qualification into the distributable bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--bundle", type=Path, required=True)
    args = options.parse_args()
    root = args.bundle.resolve(strict=True)
    manifest = json.loads((root / "BUNDLE.json").read_text(encoding="utf-8"))
    report = json.loads((root / ".installation-check/runtime.json").read_text(encoding="utf-8"))
    if ("failure" in report or report["source_commit"] != manifest["source_commit"] or
            report["python"] != "3.12.14" or report.get("CUDA_execution") != "not run" or
            any(item.get("status") != "passed" for item in report["checks"]) or
            not {"versions", "application_source_identity", "private_compilers_without_sdk", "native_without_sdk", "actual_local_gloo", "complete_python_integration", "actual_original_parser"}
                <= {item["name"] for item in report["checks"]}):
        raise RuntimeError("complete actual installed runtime qualification has not passed on this source")
    destination = root / "evidence/installed-runtime"
    destination.mkdir(parents=True, exist_ok=False)
    # The qualification directory also holds private regenerable compiler
    # caches. Distribute only the explicit bounded JSON evidence inventory.
    for name in ("runtime.json", "parser.json"):
        path = root / ".installation-check" / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 ** 2:
            raise RuntimeError("installed qualification report is absent, redirected or oversized")
        shutil.copyfile(path, destination / name)
    for path in destination.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                manifest["files"][path.relative_to(root).as_posix()] = {
                    "bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    manifest["installed_qualification"] = {"status": "passed", "report": "evidence/installed-runtime/runtime.json",
        "scope": "actual private Windows Python 3.12.14, TinyCC extension build/load, offline SM89/SM120 compilation, bare-system-PATH native imports, concrete owned classes and parser",
        "GPU_execution": "not run", "full_model_validation": "not run"}
    (root / "BUNDLE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["installed_qualification"], indent=2))


if __name__ == "__main__":
    main()
