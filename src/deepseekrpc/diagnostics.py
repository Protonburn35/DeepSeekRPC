"""Confined diagnostic archives: explicit public state, no credentials or content."""
import os
from pathlib import Path
import time
import zipfile

from .jsonutil import ValidationError, canonical, loads
from .runtime_identity import application_source_identity


def offline_report(config):
    from .worker import _OWNER
    root = config.state_root
    marker = root / ".owner"
    if root.is_symlink() or marker.is_symlink() or marker.stat().st_size != len(_OWNER) or marker.read_bytes() != _OWNER:
        raise ValidationError("offline diagnostics need an owned DeepSeekRPC runtime directory")
    path = root / "diagnostics/metrics.json"
    if path.is_symlink() or path.stat().st_size > 256 * 1024:
        raise ValidationError("stored metrics are redirected or exceed their fixed inventory")
    with path.open("rb") as stream:
        value = loads(stream.read(256 * 1024 + 1), limit=256 * 1024)
    return {"format": 1, "scope": "last saved local metrics; current process readiness is not observed",
        "configuration": config.effective(), "metrics": value,
        "sample_age_seconds": max(0., time.time() - path.stat().st_mtime),
        "collector_application": application_source_identity()}


def write_bundle(report, destination, *, logs=None):
    raw = canonical(report) + b"\n"
    if len(raw) > 1024 * 1024:
        raise ValidationError("diagnostic report exceeds its 1-MiB public metadata bound")
    destination = Path(destination)
    if destination.is_symlink() or destination.exists():
        raise ValidationError("diagnostic output must be a new file")
    # Read only the three known content-free rotated log files. Never walk the
    # state tree, model directory, environment, home, credential or image store.
    with destination.open("xb") as output:
        try:
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("diagnostics.json", raw)
                if logs is not None:
                    root = Path(logs)
                    if root.is_symlink():
                        raise ValidationError("diagnostic log directory cannot be redirected")
                    for name in ("events.jsonl", "events.1.jsonl", "events.2.jsonl"):
                        path = root / name
                        if not path.exists():
                            continue
                        if path.is_symlink() or path.stat().st_nlink != 1:
                            raise ValidationError("diagnostic log file cannot be redirected")
                        with path.open("rb") as stream:
                            value = stream.read(4 * 1024 ** 2 + 1)
                        if len(value) > 4 * 1024 ** 2:
                            raise ValidationError("diagnostic log exceeds its rotation bound")
                        archive.writestr("logs/" + name, value)
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            output.close()
            destination.unlink(missing_ok=True)
            raise
    return {"created": True, "bytes": destination.stat().st_size,
            "scope": "bounded public diagnostics and selected numeric event logs; no credentials/prompts/images"}
