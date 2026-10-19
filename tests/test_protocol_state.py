from dataclasses import replace
import hashlib
import json

import pytest

from deepseekrpc.fragments import FragmentReceiver, fragments
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.planner import Role
from deepseekrpc.protocol import MAX_ENVELOPE, MAX_PAYLOAD, PREFIX, Frame, Kind, TensorDescriptor, decode, parse_prefix
from deepseekrpc.state import Lifecycle, Operation, Session


def control(**changes):
    return replace(Frame(Kind.BEGIN, "a" * 64, Role.HOST, Role.MIDDLE, "b" * 32,
                         "c" * 32, 1, 1, 0, "d" * 32), **changes)


def operation(number=1, start=0, rows=3, epoch=1):
    return Operation("b" * 32, "c" * 32, epoch, number, f"{number:032x}", "a" * 64, start, rows)


def session(**kwargs):
    result = Session(capacity=32, max_journal_bytes=4096, sources={"kv14": (2, 1)}, **kwargs)
    result.reset("b" * 32, 1)
    return result


def test_frame_round_trip_and_integrity():
    original = control(payload=b"bounded control metadata")
    prefix, metadata, payload = original.encode()
    kind, headers, length = parse_prefix(prefix)
    assert (headers, length) == (len(metadata), len(payload))
    assert decode(kind, metadata, payload) == original
    with pytest.raises(ValidationError, match="digest"):
        decode(kind, metadata, payload + b"!")


@pytest.mark.parametrize("prefix", [
    PREFIX.pack(b"BAD!", 1, 0, 1, 1, 0), PREFIX.pack(b"DSRP", 2, 0, 1, 1, 0),
    PREFIX.pack(b"DSRP", 1, 0, 65535, 1, 0),
    PREFIX.pack(b"DSRP", 1, 0, 1, MAX_ENVELOPE + 1, 0),
    PREFIX.pack(b"DSRP", 1, 0, 1, 1, MAX_PAYLOAD + 1), b"short",
])
def test_reject_prefix_before_payload_allocation(prefix):
    with pytest.raises(ValidationError):
        parse_prefix(prefix)


def test_invalid_wire_field_types_and_extra_keys():
    _, metadata, payload = control().encode()
    raw = json.loads(metadata)
    for change in ({"sender": True}, {"epoch": -1}, {"run": "invalid"}, {"extra": 1}):
        with pytest.raises(ValidationError):
            decode(Kind.BEGIN, json.dumps(dict(raw, **change)).encode(), payload)


def test_lossless_fragmented_transfer_and_wrong_epoch():
    descriptor = TensorDescriptor("hidden_states", -1, "fp32", (17, 4, 5120), 0, "1" * 32, 0, 0)
    template = control(kind=Kind.STAGE_INPUT, tensor=descriptor, rows=17)
    original = b"abcd" * (descriptor.nbytes // 4)
    result = bytearray(len(original))
    receiver = FragmentReceiver(template, descriptor, result)
    pieces = list(fragments(template, descriptor, original))
    assert len(pieces) == 2
    with pytest.raises(ValidationError):
        receiver.accept(replace(pieces[0], epoch=0))
    assert not receiver.accept(pieces[0])
    with pytest.raises(ValidationError):
        receiver.accept(pieces[0])
    assert receiver.accept(pieces[1])
    assert bytes(result) == original
    with pytest.raises(ValidationError):
        receiver.accept(pieces[1])


@pytest.mark.parametrize("dtype,shape", [("fp16", (1, 4, 5120)), ("fp32", (1048576, 4, 5120)), ("fp32", (True,))])
def test_invalid_activation_layouts(dtype, shape):
    with pytest.raises(ValidationError):
        TensorDescriptor("hidden_states", -1, dtype, shape, 0, "a" * 32, 0, 0)


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_partial_commit_restores_overwritten_ring_rows(accepted):
    s = session()
    op = operation()
    s.begin(op)
    for i, data in enumerate((b"a", b"b", b"c")):
        s.write("kv14", i % 2, token_position=i, logical_position=i, generation=1, data=data)
    s.commit(op, accepted)
    expected = ({}, {0: b"a"}, {0: b"a", 1: b"b"}, {0: b"c", 1: b"b"})[accepted]
    assert {key[1]: row.data for key, row in s.slots.items()} == expected
    assert s.position == accepted and s.state == Lifecycle.IDLE


def test_replay_returns_original_boundary_after_later_operation():
    s = session()
    first = operation(rows=1)
    s.begin(first)
    assert s.commit(first, 1) == 1
    second = operation(number=2, start=1, rows=1)
    s.begin(second)
    s.commit(second, 1)
    assert s.begin(first) == 1
    assert s.commit(first, 1) == 1
    assert s.position == 2
    with pytest.raises(ValidationError):
        s.begin(replace(first, digest="e" * 64))
    with pytest.raises(ValidationError):
        s.commit(first, 0)


def test_cancel_then_immediate_next_request_retains_state_container():
    s = session()
    container = id(s)
    first = operation()
    s.begin(first)
    s.write("kv14", 0, token_position=0, logical_position=0, generation=0, data=b"a")
    s.abort(first)
    assert not s.slots
    second = replace(operation(number=2, rows=1), request="d" * 32)
    s.begin(second)
    s.commit(second, 1)
    assert id(s) == container and s.position == 1


def test_fault_stale_epoch_order_and_old_replay_refused():
    s = session(replay_limit=1)
    with pytest.raises(ValidationError):
        s.begin(operation(epoch=0))
    with pytest.raises(ValidationError):
        s.begin(operation(number=2))
    first = operation(rows=1)
    s.begin(first)
    s.commit(first, 1)
    second = operation(number=2, start=1, rows=1)
    s.begin(second)
    s.commit(second, 1)
    with pytest.raises(ValidationError):
        s.begin(first)
    s.fault()
    with pytest.raises(ValidationError):
        s.commit(second, 1)
    s.reset("b" * 32, 2)
    with pytest.raises(ValidationError):
        s.begin(operation())


def test_unregistered_state_and_journal_budget():
    s = Session(capacity=32, max_journal_bytes=129, sources={"kv14": (2, 1)})
    s.reset("b" * 32, 1)
    s.begin(operation())
    with pytest.raises(ValidationError):
        s.write("invented", 0, token_position=0, logical_position=0, generation=0, data=b"a")
    s.write("kv14", 0, token_position=0, logical_position=0, generation=0, data=b"a")
    with pytest.raises(ValidationError, match="budget"):
        s.write("kv14", 1, token_position=1, logical_position=1, generation=0, data=b"b")
