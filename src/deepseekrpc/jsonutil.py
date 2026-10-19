"""Strict bounded JSON for local manifests and authenticated wire metadata."""
from __future__ import annotations

import hashlib
import json
import math


class ValidationError(ValueError):
    pass


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValidationError("nonfinite JSON number")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError("JSON floating-point overflow")
    return result


def loads(data: bytes | str, *, limit: int):
    if len(data.encode("utf-8") if isinstance(data, str) else data) > limit:
        raise ValidationError("JSON byte limit exceeded")
    try:
        return json.loads(data, object_pairs_hook=_pairs, parse_constant=_nonfinite, parse_float=_float)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError("invalid or excessively nested JSON") from exc


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def integer(value, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValidationError(f"{name} must be an integer in [{low}, {high}]")
    return value
