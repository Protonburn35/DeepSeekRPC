"""Real one-rank Gloo initialization/metadata collective; no model or CUDA."""
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.reference
def test_actual_qualification_gloo_function_repeats_and_releases_files(tmp_path):
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("qualify_runtime_tool", Path(__file__).parents[1] / "tools/qualify_runtime.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    for _ in range(2):
        assert "actual local FileStore/Gloo" in tool.qualify_local_gloo(tmp_path)
        assert not torch.distributed.is_initialized()
        assert list(tmp_path.iterdir()) == []
