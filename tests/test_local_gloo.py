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
        assert "actual local HashStore/Gloo" in tool.qualify_local_gloo(tmp_path)
        assert not torch.distributed.is_initialized()
        assert list(tmp_path.iterdir()) == []


@pytest.mark.reference
def test_actual_local_group_refuses_an_existing_process_group():
    torch = pytest.importorskip("torch")
    from deepseekrpc.jsonutil import ValidationError
    from deepseekrpc.local_groups import initialize_local_process_group

    initialize_local_process_group()
    try:
        with pytest.raises(ValidationError, match="fresh worker"):
            initialize_local_process_group()
        assert torch.distributed.get_world_size() == 1
    finally:
        torch.distributed.destroy_process_group()
