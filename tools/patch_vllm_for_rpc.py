#!/usr/bin/env python3
"""Apply admitted exact-source Windows patches to a writable build copy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from acquire_references import safe_relative


def apply(source: Path, lock_name="vllm-windows-mhc.lock.json"):
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "patches" / safe_relative(lock_name)).read_text(encoding="utf-8"))
    patch = root / "patches" / safe_relative(lock["patch"])
    if hashlib.sha256(patch.read_bytes()).hexdigest() != lock["patch_sha256"]:
        raise ValueError("vLLM source patch differs from its admitted hash")
    source = source.resolve(strict=True)
    paths = []
    states = []
    for record in lock["files"]:
        path = (source / safe_relative(record["path"])).resolve(strict=True)
        if not path.is_relative_to(source) or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("vLLM patch target escapes its bounded source tree")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in {record["sha256_before"], record["sha256_after"]}:
            raise ValueError(f"vLLM patch target is not the exact pinned source: {record['path']}")
        states.append(digest == record["sha256_after"])
        paths.append(path)
    if any(states) and not all(states):
        raise ValueError("partially patched vLLM source must be investigated")
    if not all(states):
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
    for path, record in zip(paths, lock["files"]):
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256_after"]:
            raise ValueError("vLLM patch output fingerprint mismatch")
    return {"name": lock["patch"], "sha256": lock["patch_sha256"], "files": lock["files"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", default="vllm-windows-mhc.lock.json")
    args = parser.parse_args()
    print(json.dumps(apply(args.source, args.lock), indent=2))
