#!/usr/bin/env python3
"""Acquire only hash-pinned source archives, metadata and bounded shard headers.

No full model download exists in this tool. Header servers that ignore Range
are rejected before reading a response body. Existing mismatched cache files
are reported, never silently overwritten or accepted by changing a checksum.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct
import tarfile
import urllib.request

REPOSITORY = Path(__file__).resolve().parents[1]


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("unsafe reference filename")
    return Path(*path.parts)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bounded_download(url, path, *, expected_size, sha256, limit):
    if expected_size > limit:
        raise ValueError("reference exceeds explicit download budget")
    if path.exists():
        if path.stat().st_size != expected_size or digest(path) != sha256:
            raise ValueError(f"cached reference checksum mismatch: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=45) as response, partial.open("xb") as stream:
            remaining = expected_size
            while remaining:
                data = response.read(min(1024 * 1024, remaining))
                if not data:
                    raise ValueError("short reference download")
                stream.write(data)
                remaining -= len(data)
            if response.read(1):
                raise ValueError("oversized reference download")
        if digest(partial) != sha256:
            raise ValueError("reference checksum mismatch")
        partial.rename(path)
    finally:
        if partial.exists():
            partial.unlink()


def source_reference(root, record):
    name = record["repository"].split("/")[-1]
    archive = root / "archives" / f'{name}-{record["commit"]}.tar.gz'
    bounded_download(record["url"], archive, expected_size=record["archive_bytes"],
                     sha256=record["archive_sha256"], limit=256 * 1024 * 1024)
    destination = root / "source" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        prefixes = {member.name.split("/")[0] for member in members}
        if len(prefixes) != 1 or sum(m.size for m in members) > 1024 ** 3:
            raise ValueError("source archive has unexpected roots or extraction size")
        prefix = prefixes.pop()
        if not destination.exists():
            temporary = root / "source" / f".{name}.extract"
            temporary.mkdir(exist_ok=False)
            tar.extractall(temporary, filter="data")
            (temporary / prefix).rename(destination)
            temporary.rmdir()
        expected_files = set()
        for member in members:
            relative = safe_relative(member.name).relative_to(prefix)
            path = destination / relative
            if member.isdir():
                continue
            expected_files.add(relative.as_posix())
            if member.issym():
                if not path.is_symlink() or os.readlink(path) != member.linkname:
                    raise ValueError("source symlink differs from pinned archive")
                if not path.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("source symlink escapes reference root")
            elif member.isfile():
                source = tar.extractfile(member)
                if path.is_symlink() or not path.is_file() or path.stat().st_size != member.size:
                    raise ValueError("source file differs from pinned archive")
                with source:
                    expected = hashlib.file_digest(source, "sha256").hexdigest()
                if digest(path) != expected:
                    raise ValueError(f"modified source reference: {relative}")
            else:
                raise ValueError("unexpected source archive entry type")
        observed = {p.relative_to(destination).as_posix() for p in destination.rglob("*")
                    if p.is_file() or p.is_symlink()}
        if observed != expected_files:
            raise ValueError("unrecognized files in source reference; preserve and investigate")
    for base, directories, files in os.walk(destination, topdown=False):
        for filename in files:
            path = Path(base) / filename
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
        Path(base).chmod(Path(base).stat().st_mode & ~0o222)
    return name


def header_range(url, begin, end, file_size):
    request = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end}"})
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{end}/{file_size}":
            raise ValueError("server ignored bounded header Range; full payload will not be read")
        data = response.read(end - begin + 2)
        if len(data) != end - begin + 1:
            raise ValueError("incorrect header range length")
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/deepseekrpc-references"))
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = json.loads((REPOSITORY / "provenance/sources.lock.json").read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name in pool.map(lambda row: source_reference(root, row), sources["sources"]):
            print("Verified source:", name, flush=True)
    models = json.loads((REPOSITORY / "provenance/models.lock.json").read_text())
    for model in models["models"]:
        for entry in model["files"]:
            name = entry["name"]
            if name.endswith(".safetensors"):
                raise ValueError("full model payload download is forbidden")
            path = root / "metadata" / model["label"] / safe_relative(name)
            url = f'https://huggingface.co/{model["repository"]}/resolve/{model["revision"]}/{name}'
            bounded_download(url, path, expected_size=entry["bytes"], sha256=entry["sha256"], limit=64 * 1024 * 1024)
    headers = json.loads((REPOSITORY / "provenance/headers.lock.json").read_text())
    model_map = {m["label"]: m for m in models["models"]}
    def header(entry):
        model = model_map[entry["model"]]
        if entry["revision"] != model["revision"] or not 2 <= entry["header_bytes"] <= 8 * 1024 * 1024:
            raise ValueError("incompatible header identity or length")
        path = root / "metadata" / entry["model"] / "headers" / (safe_relative(entry["name"]).name + ".json")
        if path.exists():
            if digest(path) != entry["header_sha256"] or path.stat().st_size != entry["header_bytes"]:
                raise ValueError("cached header checksum mismatch")
            return
        url = f'https://huggingface.co/{model["repository"]}/resolve/{model["revision"]}/{entry["name"]}'
        length = struct.unpack("<Q", header_range(url, 0, 7, entry["file_bytes"]))[0]
        if length != entry["header_bytes"]:
            raise ValueError("remote header length differs from pinned metadata")
        data = header_range(url, 8, 7 + length, entry["file_bytes"])
        if hashlib.sha256(data).hexdigest() != entry["header_sha256"]:
            raise ValueError("remote header checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(header, headers["files"]))
    (root / "metadata/headers-manifest.json").write_text(json.dumps(headers, indent=2) + "\n")
    print("Verified all pinned metadata and 45 bounded headers; no weight payloads downloaded.")


if __name__ == "__main__":
    main()
