"""Test-only three-process affine/state fixture. Never an inference backend.

Production code provides transport, framing, fragments and state journals. The
deliberately simple stage math here only checks process/socket/state integration.
"""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from deepseekrpc.fragments import FragmentReceiver, fragments
from deepseekrpc.jsonutil import canonical, loads
from deepseekrpc.planner import Role
from deepseekrpc.protocol import Frame, Kind, TensorDescriptor
from deepseekrpc.state import Operation, Session
from deepseekrpc.transport import TLSFiles, accept, connect, listen

RUN = "a" * 64
SID = "b" * 32


def make_frame(kind, sender, receiver, op, rows, start=0, payload=b"", tensor=None):
    return Frame(kind, RUN, sender, receiver, SID, f"{op + 100:032x}", 1, op, 0, f"{op:032x}",
                 start, rows, 1, tensor=tensor, payload=payload)


def operation(frame):
    payload = loads(frame.payload, limit=1024)
    return Operation(frame.session, frame.request, frame.epoch, frame.operation,
                     frame.transaction, payload["digest"], frame.position, frame.rows)


def tensor_info(rows, start):
    return TensorDescriptor("hidden_states", -1, "fp32", (rows, 4, 5120), start, "1" * 32, 0, 0)


def send_tensor(connection, template, array):
    descriptor = tensor_info(template.rows, template.position)
    data_frame = replace(template, kind=Kind.STAGE_INPUT, tensor=descriptor, payload=b"")
    for frame in fragments(data_frame, descriptor, array):
        connection.send(frame)


def receive_tensor(connection, template):
    if not 1 <= template.rows <= 17:
        raise ValueError("fixture rows outside local admission budget")
    descriptor = tensor_info(template.rows, template.position)
    storage = bytearray(descriptor.nbytes)
    data_frame = replace(template, kind=Kind.STAGE_INPUT, tensor=descriptor, payload=b"")
    receiver = FragmentReceiver(data_frame, descriptor, storage)
    while not receiver.complete:
        receiver.accept(connection.receive())
    return np.frombuffer(storage, dtype=np.float32).reshape(descriptor.shape)


def touch(session, op, array):
    for i in range(op.rows):
        session.write("fixture", op.start + i, token_position=op.start + i,
                      logical_position=op.start + i, generation=1,
                      data=array[i, 0, 0].tobytes())


def run(role, directory, ready, result, downstream_port=None):
    tls = TLSFiles(directory / "ca.pem", directory / f"{role.label}.pem", directory / f"{role.label}.key")
    session = Session(capacity=32, max_journal_bytes=65536, sources={"fixture": (32, 4)})
    session.reset(SID, 1)
    loaded_id = uuid.uuid4().hex
    positions = []
    identities = []
    connections = []
    listener = None
    try:
        if role != Role.HOST:
            listener = listen("127.0.0.1", 0)
            ready.write_text(json.dumps({"port": listener.getsockname()[1]}))
        if role != Role.TERMINAL:
            peer = Role(int(role) + 1)
            down_control = connect(("127.0.0.1", downstream_port), tls, local=role, peer=peer,
                                   run=RUN, plane="control", timeout=10)
            down_data = connect(("127.0.0.1", downstream_port), tls, local=role, peer=peer,
                                run=RUN, plane="data", timeout=10)
            connections.extend((down_control, down_data))
        if role != Role.HOST:
            peer = Role(int(role) - 1)
            up_control = accept(listener, tls, local=role, peer=peer, run=RUN, plane="control", timeout=10)
            up_data = accept(listener, tls, local=role, peer=peer, run=RUN, plane="data", timeout=10)
            connections.extend((up_control, up_data))
        for number, (rows, cancel) in enumerate(((17, False), (3, True), (2, False)), start=1):
            if role == Role.HOST:
                original = np.arange(rows * 4 * 5120, dtype=np.float32).reshape(rows, 4, 5120) / 16
                begin = make_frame(Kind.STEP_BEGIN, role, Role.MIDDLE, number, rows, session.position,
                                   canonical({"digest": hashlib.sha256(original.tobytes()).hexdigest()}))
            else:
                begin = up_control.receive()
                if begin.kind != Kind.STEP_BEGIN:
                    raise AssertionError("unexpected control operation")
            op = operation(begin)
            session.begin(op)
            if role == Role.HOST:
                output = original * 2 + 1
            else:
                input_values = receive_tensor(up_data, begin)
                output = input_values * 3 - 2 if role == Role.MIDDLE else input_values * 0.5 + 4
            touch(session, op, output)
            if role != Role.TERMINAL:
                outgoing = replace(begin, sender=role, receiver=Role(int(role) + 1))
                down_control.send(outgoing)
                send_tensor(down_data, outgoing, output)
                reply = replace(begin, sender=Role(int(role) + 1), receiver=role)
                output = receive_tensor(down_data, reply)
                complete = down_control.receive()
                assert complete.kind == Kind.STAGE_DONE
                records = loads(complete.payload, limit=4096)
            else:
                records = []
            records.append({"role": role.label, "loaded_id": loaded_id})
            if role != Role.HOST:
                reply = replace(begin, sender=role, receiver=Role(int(role) - 1))
                send_tensor(up_data, reply, output)
                up_control.send(replace(reply, kind=Kind.STAGE_DONE, payload=canonical(records)))
                decision = up_control.receive()
            else:
                # Independent closed-form affine reference, no calls into stage helpers.
                np.testing.assert_array_equal(output, original * 3 + np.float32(4.5))
                identities.append(records)
                decision = replace(begin, kind=Kind.CANCEL if cancel else Kind.COMMIT,
                                   payload=canonical({"accepted": 0 if cancel else rows}))
            if role != Role.TERMINAL:
                down_control.send(replace(decision, sender=role, receiver=Role(int(role) + 1)))
                ack = down_control.receive()
                assert ack.kind == (Kind.CANCEL_ACK if cancel else Kind.COMMIT_ACK)
            if decision.kind == Kind.CANCEL:
                session.abort(op)
            elif decision.kind == Kind.COMMIT:
                session.commit(op, loads(decision.payload, limit=1024)["accepted"])
            else:
                raise AssertionError("unexpected transaction decision")
            positions.append(session.position)
            if role != Role.HOST:
                reply = replace(decision, sender=role, receiver=Role(int(role) - 1),
                                kind=Kind.CANCEL_ACK if cancel else Kind.COMMIT_ACK,
                                payload=canonical({"position": session.position}))
                up_control.send(reply)
        result.write_text(json.dumps({"role": role.label, "positions": positions,
                                     "loaded_id": loaded_id, "identities": identities}))
    finally:
        for connection in connections:
            connection.close()
        if listener:
            listener.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=[r.label for r in Role], required=True)
    parser.add_argument("--cert-dir", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--downstream-port", type=int)
    args = parser.parse_args()
    run(Role[args.role.upper()], args.cert_dir, args.ready, args.result, args.downstream_port)
