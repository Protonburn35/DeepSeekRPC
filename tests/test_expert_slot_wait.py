"""Control-wait behavior with real host events; these are not GPU results."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from deepseekrpc.cuda_cache import _wait_for_expert_event


def test_expert_event_wait_keeps_work_pending_until_completion():
    queried, complete = threading.Event(), threading.Event()
    def ready():
        queried.set()
        return complete.is_set()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_wait_for_expert_event, ready, timeout=5)
        assert queried.wait(2)
        assert not future.done()
        complete.set()
        future.result(timeout=5)


def test_expert_event_wait_can_be_cancelled_while_completion_is_pending():
    queried, cancelled = threading.Event(), threading.Event()
    def ready():
        queried.set()
        return False
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_wait_for_expert_event, ready, cancelled=cancelled.is_set, timeout=5)
        assert queried.wait(2)
        cancelled.set()
        with pytest.raises(InterruptedError, match="slot wait cancelled"):
            future.result(timeout=5)


def test_expert_event_timeout_does_not_invent_completion():
    incomplete = threading.Event()
    with pytest.raises(TimeoutError, match="CUDA completion deadline"):
        _wait_for_expert_event(incomplete.is_set, timeout=.001)
    assert not incomplete.is_set()
