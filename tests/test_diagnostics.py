import json
import zipfile

import pytest

from deepseekrpc.diagnostics import write_bundle
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.metrics import Metrics


def test_diagnostic_archive_copies_only_bounded_explicit_public_files(tmp_path):
    root = tmp_path / "diagnostics"
    metrics = Metrics(root=root, stage="C")
    metrics.event("loaded")
    (root / "credentials.key").write_text("must never be copied")
    output = tmp_path / "report.zip"
    result = write_bundle({"metrics": metrics.snapshot()}, output, logs=root)
    assert result["created"]
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"diagnostics.json", "logs/events.jsonl"}
        assert json.loads(archive.read("diagnostics.json"))["metrics"]["stage"] == "C"
        assert b"must never be copied" not in b"".join(archive.read(name) for name in archive.namelist())
    with pytest.raises(ValidationError, match="new file"):
        write_bundle({}, output)


def test_oversized_or_redirected_diagnostic_log_is_rejected_and_new_partial_removed(tmp_path):
    root = tmp_path / "diagnostics"
    root.mkdir()
    with (root / "events.jsonl").open("wb") as stream:
        stream.truncate(4 * 1024 ** 2 + 1)
    output = tmp_path / "bad.zip"
    with pytest.raises(ValidationError, match="bound"):
        write_bundle({}, output, logs=root)
    assert not output.exists()
