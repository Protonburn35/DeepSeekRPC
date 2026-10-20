"""Real thread scheduling/queue checks; this fixture never emulates a model."""
from concurrent.futures import TimeoutError
import threading

import pytest

from deepseekrpc.actor import ActorBusy, ActorInvalidated, SerializedActor


def test_bounded_actor_interrupts_immediately_and_discards_old_queued_work():
    entered, release, interrupted, faulted = (threading.Event() for _ in range(4))
    calls, owners = [], []
    def dispatch(value):
        owners.append(threading.get_ident())
        calls.append(value)
        if value == "blocking":
            entered.set()
            assert release.wait(3)
        return value
    def fault():
        owners.append(threading.get_ident())
        faulted.set()
    actor = SerializedActor(dispatch, interrupt=interrupted.set, fault=fault, capacity=1)
    try:
        first = actor.submit("blocking")
        assert entered.wait(2)
        queued = actor.submit("stale")
        with pytest.raises(ActorBusy):
            actor.submit("excess")
        actor.invalidate()
        assert interrupted.is_set() and not faulted.is_set()
        with pytest.raises(TimeoutError):
            first.result(timeout=.01)
        release.set()
        with pytest.raises(ActorInvalidated):
            first.result(timeout=2)
        with pytest.raises(ActorInvalidated):
            queued.result(timeout=2)
        assert faulted.wait(2)
        assert actor.submit("new epoch reset").result(timeout=2) == "new epoch reset"
        assert calls == ["blocking", "new epoch reset"]
        assert len(set(owners)) == 1 and owners[0] != threading.get_ident()
    finally:
        release.set()
        actor.close()


def test_actor_cancellation_and_dispatch_exception_leave_owner_responsive():
    entered, release = threading.Event(), threading.Event()
    def dispatch(value):
        if value == 1:
            entered.set()
            assert release.wait(3)
        if value == 3:
            raise ValueError("fixture failure")
        return value
    actor = SerializedActor(dispatch, interrupt=release.set, fault=lambda: None)
    try:
        first = actor.submit(1)
        assert entered.wait(2)
        cancelled = actor.submit(2)
        assert cancelled.cancel()
        release.set()
        assert first.result(timeout=2) == 1
        with pytest.raises(ValueError, match="fixture failure"):
            actor.submit(3).result(timeout=2)
        assert actor.submit(4).result(timeout=2) == 4
    finally:
        actor.close()


def test_serialized_lifecycle_install_follows_fault_on_the_same_owner():
    calls = []
    actor = SerializedActor(lambda: None, interrupt=lambda: calls.append("interrupt"),
                            fault=lambda: calls.append(("fault", threading.get_ident())))
    try:
        actor.invalidate()
        result = actor.call(lambda: calls.append(("install", threading.get_ident())))
        result.result(timeout=2)
        assert calls[0] == "interrupt"
        assert calls[1][0] == "fault" and calls[2][0] == "install"
        assert calls[1][1] == calls[2][1] != threading.get_ident()
    finally:
        actor.close()
