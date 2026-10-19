#!/usr/bin/env python3
"""Install only pinned official CUDA compiler components on a Windows builder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import urllib.request
import zipfile


def prepare(lock: Path, destination: Path):
    if os.name != "nt":
        raise RuntimeError("the Windows SDK is installed only on the native Windows builder")
    manifest = json.loads(lock.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    downloads = destination.parent / "cuda-downloads"
    downloads.mkdir(exist_ok=True)
    for package in manifest["packages"]:
        url = package["url"]
        if not url.startswith("https://developer.download.nvidia.com/compute/cuda/redist/"):
            raise ValueError("CUDA SDK URL is outside the pinned NVIDIA distribution")
        archive = downloads / Path(url).name
        expected = package["sha256"]
        valid = (archive.is_file() and archive.stat().st_size == package["bytes"] and
                 hashlib.file_digest(archive.open("rb"), "sha256").hexdigest() == expected)
        if not valid:
            temporary = archive.with_suffix(".partial")
            digest, count = hashlib.sha256(), 0
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > package["bytes"]:
                        raise ValueError("CUDA package exceeds the locked byte length")
                    digest.update(chunk)
                    stream.write(chunk)
            if count != package["bytes"] or digest.hexdigest() != expected:
                raise ValueError("CUDA package SHA-256/byte count mismatch")
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as bundle:
            if sum(item.file_size for item in bundle.infolist()) > 4 * 1024**3:
                raise ValueError("CUDA archive uncompressed bound exceeded")
            for item in bundle.infolist():
                path = PurePosixPath(item.filename)
                if (path.is_absolute() or any(v in {"", ".", ".."} or ":" in v or "\\" in v for v in path.parts)
                        or stat.S_ISLNK(item.external_attr >> 16)):
                    raise ValueError("unsafe CUDA archive member")
                # NVIDIA packages each have a single package-name root.
                if not path.parts or not path.parts[0].endswith("-archive"):
                    raise ValueError("CUDA package root differs from the locked distribution")
                if len(path.parts) == 1:
                    continue
                target = destination.joinpath(*path.parts[1:])
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(item) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
        print(f"Verified and installed CUDA component: {package['component']}", flush=True)
    for required in ("bin/nvcc.exe", "include/cuda_runtime.h", "include/crt/host_defines.h",
                     "nvvm/libdevice/libdevice.10.bc", "lib/x64/cudart.lib", "lib/x64/cublas.lib"):
        if not (destination / required).is_file():
            raise RuntimeError(f"CUDA SDK incomplete: {required}")
    result = {"root": str(destination.resolve()), "version": manifest["version"],
              "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}
    (destination / "DEEPSEEKRPC_SDK.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().parents[1] / "provenance/cuda-sdk-windows.lock.json")
    args = parser.parse_args()
    print(json.dumps(prepare(args.lock, args.destination), indent=2))
