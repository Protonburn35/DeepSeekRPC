"""Strict local deployment configuration with one shared cluster fingerprint."""
from dataclasses import dataclass
import ipaddress
from pathlib import Path
import re

from .jsonutil import ValidationError, fingerprint, integer, loads
from .planner import Role
from .protocol import hex_id
from .transport import TLSFiles

_ROLES = {"A": Role.HOST, "B": Role.MIDDLE, "C": Role.TERMINAL}


def _object(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValidationError("deployment configuration has missing or unknown fields")


def _read(path, limit=32 * 1024):
    with path.open("rb") as stream:
        return loads(stream.read(limit + 1), limit=limit)


def _path(base, value):
    if not isinstance(value, str) or not value or len(value) > 2048 or "\x00" in value or re.search(r"[<>]|REPLACE_|CHOOSE_", value):
        raise ValidationError("deployment path is missing or still contains a setup placeholder")
    result = Path(value).expanduser()
    return (base / result).resolve() if not result.is_absolute() else result.resolve()


def _endpoint(value):
    _object(value, ("address", "port"))
    try:
        ip = ipaddress.IPv4Address(value["address"])
    except (ValueError, TypeError):
        raise ValidationError("replace every worker address placeholder with its actual private IPv4 address") from None
    if ip.is_unspecified or ip.is_multicast or not (ip.is_private or ip.is_loopback):
        raise ValidationError("deployment listeners need explicit private IPv4 addresses")
    return str(ip), integer(value["port"], "deployment listener port", 1, 65535)


@dataclass(frozen=True)
class DeploymentConfig:
    role: Role
    run: str
    shared: dict
    model_root: Path
    engram_root: Path | None
    state_root: Path
    tls: TLSFiles
    endpoints: dict
    api_endpoint: tuple[str, int]
    api_key_file: Path | None
    device: int
    ram_tier_bytes: int
    default_output: int

    @classmethod
    def read(cls, filename: Path):
        filename = filename.resolve(strict=True)
        local = _read(filename)
        _object(local, ("format", "stage", "shared", "model_root", "state_root", "tls"),
                ("engram_root", "device", "ram_tier_bytes", "api_key_file", "default_output"))
        if (type(local["format"]) is not int or local["format"] != 1 or
                not isinstance(local["stage"], str) or local["stage"] not in _ROLES):
            raise ValidationError("deployment format must be 1 and stage must be A, B or C")
        role = _ROLES[local["stage"]]
        shared_path = _path(filename.parent, local["shared"])
        shared = _read(shared_path)
        _object(shared, ("format", "cluster_id", "split", "context", "kv_dtype", "index_dtype", "layer_stage_wire", "replay", "workers", "api"))
        if (type(shared["format"]) is not int or shared["format"] != 1 or
                shared["context"] != 1048576 or type(shared["context"]) is not int or
                shared["kv_dtype"] != "bf16" or shared["index_dtype"] != "bf16" or
                shared["layer_stage_wire"] != "fp32" or shared["replay"] != "exact"):
            raise ValidationError("this runtime requires 1M context, unquantized BF16 caches, FP32 wire and exact replay")
        hex_id(shared["cluster_id"], 32, "configured cluster")
        split = shared["split"]
        if not isinstance(split, list) or len(split) != 3 or any(type(n) is not int or not 1 <= n <= 38 for n in split) or sum(split) != 40:
            raise ValidationError("split must assign all 40 layers to three nonempty contiguous stages")
        if split[0] <= 14 or sum(split[:2]) > 37:
            raise ValidationError("A must own both Engrams and C must own target layers 37/38/39")
        _object(shared["workers"], ("B", "C"))
        endpoints = {_ROLES[key]: _endpoint(shared["workers"][key]) for key in ("B", "C")}
        if len(set(endpoints.values())) != 2:
            raise ValidationError("B and C require distinct worker endpoints")
        api = _endpoint(shared["api"])
        tls = local["tls"]
        _object(tls, ("ca", "certificate", "private_key"))
        tls = TLSFiles(*(_path(filename.parent, tls[key]) for key in ("ca", "certificate", "private_key")))
        model = _path(filename.parent, local["model_root"])
        engram = _path(filename.parent, local["engram_root"]) if "engram_root" in local else None
        if (role == Role.HOST) != (engram is not None):
            raise ValidationError("only A must configure the original Engram directory")
        state = _path(filename.parent, local["state_root"])
        # Generated files can never be placed in an existing model/reference
        # directory, including by resolving a relative path through a symlink.
        for protected in (model, engram):
            if protected is not None and (state == protected or state.is_relative_to(protected) or protected.is_relative_to(state)):
                raise ValidationError("runtime state requires a separate directory outside model/Engram storage")
        key = _path(filename.parent, local["api_key_file"]) if "api_key_file" in local else None
        if role != Role.HOST and key is not None:
            raise ValidationError("only A serves HTTP and may configure its API key file")
        if role == Role.HOST and not ipaddress.IPv4Address(api[0]).is_loopback and key is None:
            raise ValidationError("private LAN HTTP binding requires an API key file")
        return cls(role, fingerprint(shared), shared, model, engram, state, tls, endpoints, api, key,
                   integer(local.get("device", 0), "CUDA device", 0, 31),
                   integer(local.get("ram_tier_bytes", 8 * 1024 ** 3), "RAM tier bytes", 0, 256 * 1024 ** 3),
                   integer(local.get("default_output", 8192), "default output tokens", 1, 1048576))

    def api_key(self):
        if self.api_key_file is None:
            return None
        # Never include the path or contents in diagnostic output.
        if self.api_key_file.stat().st_size > 258:
            raise ValidationError("API key file exceeds its bounded credential size")
        key = self.api_key_file.read_text(encoding="ascii").rstrip("\r\n")
        if not 32 <= len(key) <= 256 or not key.isascii() or any(c.isspace() for c in key):
            raise ValidationError("API key file must contain one 32..256 character ASCII token")
        return key

    def effective(self):
        return {"format": 1, "stage": "ABC"[int(self.role)], "run": self.run, "shared": self.shared,
                "device": self.device, "ram_tier_bytes": self.ram_tier_bytes,
                "default_output": self.default_output, "credentials": "local files; redacted",
                "model_directory": "local; redacted", "state_directory": "local; redacted"}
