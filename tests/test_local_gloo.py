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
        assert "actual local bounded Torch Store/Gloo" in tool.qualify_local_gloo(tmp_path)
        assert not torch.distributed.is_initialized()
        assert list(tmp_path.iterdir()) == []


@pytest.mark.reference
def test_actual_local_group_refuses_an_existing_process_group():
    torch = pytest.importorskip("torch")
    from deepseekrpc.jsonutil import ValidationError
    from deepseekrpc.local_groups import initialize_local_process_group

    store = initialize_local_process_group()
    try:
        with pytest.raises(ValidationError, match="fresh worker"):
            initialize_local_process_group()
        assert torch.distributed.get_world_size() == 1
    finally:
        torch.distributed.destroy_process_group()
        del store


@pytest.mark.reference
def test_real_torch_prefix_store_trampoline_and_limits():
    torch = pytest.importorskip("torch")
    from datetime import timedelta
    from concurrent.futures import ThreadPoolExecutor
    from deepseekrpc.jsonutil import ValidationError
    from deepseekrpc.local_groups import LocalRendezvousStore

    local = LocalRendezvousStore()
    store = torch.distributed.PrefixStore("actual C++ prefix", local)
    store.set_timeout(timedelta(seconds=2))
    store.set("key", b"first")
    assert store.get("key") == b"first"
    assert store.compare_set("key", b"wrong", b"unwritten") == b"first"
    assert store.compare_set("key", b"first", b"next") == b"next"
    assert store.compare_set("new", b"wrong", b"unwritten") == b""
    assert not store.check(["new"])
    assert store.add("counter", 2) == 2
    assert store.clone().add("counter", 3) == 5
    with ThreadPoolExecutor(max_workers=2) as executor:
        waiting = executor.submit(store.get, "later")
        store.set("later", b"arrived")
        assert waiting.result(timeout=5) == b"arrived"
        assert len(list(executor.map(lambda _: store.add("counter", 1), range(32)))) == 32
    assert store.get("counter") == b"37"
    assert store.num_keys() == 3
    assert store.delete_key("later") and not store.delete_key("later")
    with pytest.raises(ValidationError, match="value exceeds"):
        store.set("oversized", b"x" * (local.MAX_VALUE + 1))
    assert not store.check(["oversized"])
    with pytest.raises(torch.distributed.DistStoreError):
        store.wait(["missing"], timedelta(milliseconds=1))
    # Exercise admission through the actual C++ wrapper; a refused addition
    # must leave both the previous values and missing-key state intact.
    local.MAX_KEYS = 2
    with pytest.raises(ValidationError, match="inventory exceeds"):
        store.set("overflow", b"unwritten")
    assert store.get("key") == b"next" and not store.check(["overflow"])
