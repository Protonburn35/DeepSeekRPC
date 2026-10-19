#!/usr/bin/env python3
"""Generate complete ownership/byte inventory from pinned, header-only metadata.

Usage: PYTHONPATH=src python tools/build_inventory.py --metadata-root PATH
No network access or weight payload reads occur in this command.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from deepseekrpc.inventory import ownership, parse_header
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.planner import Geometry, Role, StagePlan


def build(root: Path, output: Path):
    manifest = json.loads((root / "headers-manifest.json").read_text())
    config = json.loads((root / "pack/config.json").read_text())
    plan = StagePlan(Geometry.from_config(config), (18, 5, 17))
    weight_map = json.loads((root / "pack/model.safetensors.index.json").read_text())["weight_map"]
    quantization = json.loads((root / "pack/quantization_config.json").read_text())["tensor_storage"]
    described = {name: info for module in quantization.values()
                 for name, info in module.get("stored_tensors", {}).items()}
    bits = Counter()
    k_map = json.loads((root / "pack/exl3_k_map.json").read_text())
    for module, entry in quantization.items():
        if "bits_per_weight" in entry:
            k = entry["bits_per_weight"]
            for name, spec in entry["stored_tensors"].items():
                if name.endswith(".trellis"):
                    if spec["shape"][-1] != 16 * k:
                        raise ValidationError(f"physical K disagrees with quant metadata: {module}")
    seen = set()
    sums = {r.label: Counter() for r in Role}
    entries, pack_payload = 0, 0
    external_tables = Counter()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "tensors.jsonl.gz"
    with destination.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for file in manifest["files"]:
            label, name = file["model"], file["name"]
            data = (root / label / "headers" / f"{name}.json").read_bytes()
            if hashlib.sha256(data).hexdigest() != file["header_sha256"]:
                raise ValidationError(f"header checksum mismatch: {name}")
            if len(data) != file["header_bytes"]:
                raise ValidationError("header byte count mismatch")
            tensors = parse_header(data, file["file_bytes"])
            for tensor in tensors.values():
                # Only original table/scale tensors are required externally;
                # Engram projections already exist in the EXL3 pack.
                if label == "official" and ".engram.embed." not in tensor.name:
                    continue
                role, component = ownership(tensor.name, plan)
                physical_k = None
                if label == "pack":
                    if tensor.name in seen or weight_map.get(tensor.name) != name:
                        raise ValidationError(f"index/header duplicate or placement mismatch: {tensor.name}")
                    seen.add(tensor.name)
                    pack_payload += tensor.nbytes
                    spec = described.get(tensor.name)
                    if spec:
                        dtype_names = {"torch.float16": "F16", "torch.bfloat16": "BF16", "torch.float32": "F32",
                                       "torch.int16": "I16", "torch.int32": "I32"}
                        if (spec["n_bytes"] != tensor.nbytes or tuple(spec["shape"]) != tensor.shape or
                                dtype_names.get(spec["dtype"]) != tensor.dtype):
                            raise ValidationError("quant metadata disagrees with physical header")
                    sums[role.label][component] += tensor.nbytes
                    if tensor.name.endswith(".trellis"):
                        if tensor.dtype != "I16" or len(tensor.shape) != 3 or tensor.shape[-1] % 16:
                            raise ValidationError("unsupported physical EXL3 trellis layout")
                        physical_k = tensor.shape[-1] // 16
                        if not 1 <= physical_k <= 8:
                            raise ValidationError("unsupported EXL3 precision")
                        bits[physical_k] += 1
                        if component in {"routed_experts", "shared_expert"}:
                            layer = tensor.name.split(".")[1]
                            category = "routed" if component == "routed_experts" else "shared"
                            expected = k_map[category][layer]
                            allowed = {4, 5} if category == "shared" and layer == "29" else {expected}
                            if physical_k not in allowed:
                                raise ValidationError("physical expert K disagrees with selected K-map")
                else:
                    if role != Role.HOST:
                        raise ValidationError("Engram table has non-host owner")
                    external_tables[component] += tensor.nbytes
                record = dict(tensor.as_dict(), file=name, model=label,
                              revision=file["revision"], publisher_file_sha256=file["file_sha256"],
                              owner=role.label, component=component,
                              physical_k=physical_k,
                              file_hash_status="publisher metadata; payload not downloaded")
                gz.write(json.dumps(record, separators=(",", ":")).encode() + b"\n")
                entries += 1
    if seen != set(weight_map):
        raise ValidationError(f"unaccounted indexed tensors: {len(set(weight_map) - seen)}")
    summary = {
        "schema": 1, "status": "header-derived weight bytes; prospective state allocations",
        "plan": plan.as_dict(), "tensor_records": entries,
        "pack_payload_bytes": pack_payload,
        "pack_owned_component_bytes": sums,
        "pack_owned_total_bytes": {r: sum(s.values()) for r, s in sums.items()},
        "external_host_table_payload_bytes": dict(external_tables),
        "quantized_module_count_by_k": dict(sorted(bits.items())),
        "quant_metadata_missing_tensor_descriptors": len(seen - set(described)),
        "manifest_sha256": hashlib.file_digest(destination.open("rb"), "sha256").hexdigest(),
        "hardware_allocations_measured": False,
        "not_included_in_weight_totals": [
            "host/device duplicate copies", "source/shadow/SWA state", "workspaces/staging/allocator overhead",
            "OS/display/runtime reserve", "resident locking/WDDM admission", "draft embedding duplicate on C",
        ],
    }
    (output / "allocation-plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("provenance"))
    args = parser.parse_args()
    build(args.metadata_root, args.output)
