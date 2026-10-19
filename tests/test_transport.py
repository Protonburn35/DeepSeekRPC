from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import socket
import ssl
import time
import threading

import pytest

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.planner import Role
from deepseekrpc.protocol import MAX_PAYLOAD, PREFIX, Frame, Kind
from deepseekrpc.transport import accept, accept_authenticated, connect, listen
from deepseekrpc.control import ControlPump
from deepseekrpc.control_rpc import ControlRPC, RemoteError

RUN = "a" * 64
ZERO = "0" * 32


def frame(sender, receiver, kind=Kind.HEARTBEAT):
    return Frame(kind, RUN, sender, receiver, ZERO, ZERO, 1, 1, 0, ZERO)


def test_persistent_authenticated_channel(certificates):
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        def server():
            with accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                        run=RUN, plane="control", timeout=3) as connection:
                for _ in range(3):
                    request = connection.receive()
                    connection.send(replace(request, sender=Role.MIDDLE, receiver=Role.HOST))
                return connection.bytes_received
        result = pool.submit(server)
        with connect(listener.getsockname(), certificates[Role.HOST], local=Role.HOST, peer=Role.MIDDLE,
                     run=RUN, plane="control", timeout=3) as client:
            for _ in range(3):
                client.send(frame(Role.HOST, Role.MIDDLE))
                assert client.receive().kind == Kind.HEARTBEAT
        assert result.result(timeout=5) > 0


def test_wrong_trusted_stage_certificate_rejected(certificates):
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        def server():
            with pytest.raises(ValidationError, match="certificate"):
                accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                       run=RUN, plane="control", timeout=2)
        result = pool.submit(server)
        with pytest.raises((OSError, EOFError, ValidationError)):
            connect(listener.getsockname(), certificates[Role.TERMINAL], local=Role.HOST, peer=Role.MIDDLE,
                    run=RUN, plane="control", timeout=2)
        result.result(timeout=4)


def test_client_without_certificate_rejected(certificates):
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        def server():
            with pytest.raises(ssl.SSLError):
                accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                       run=RUN, plane="control", timeout=2)
        result = pool.submit(server)
        context = ssl.create_default_context(cafile=certificates[Role.HOST].ca)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        with socket.create_connection(listener.getsockname(), timeout=2) as raw:
            with pytest.raises((ssl.SSLError, EOFError)):
                with context.wrap_socket(raw, server_hostname=Role.MIDDLE.dns_name) as client:
                    data = client.recv(1)
                    if not data:
                        raise EOFError()
        result.result(timeout=4)


def test_untrusted_ca_rejected(certificates, tmp_path):
    from conftest import make_certificates
    other = make_certificates(tmp_path / "other-ca")
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        def server():
            # Windows may deliver the rejecting client's TLS close as WSAECONNRESET.
            with pytest.raises((ssl.SSLError, ConnectionResetError)):
                accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                       run=RUN, plane="control", timeout=2)
        result = pool.submit(server)
        with pytest.raises(ssl.SSLError):
            connect(listener.getsockname(), other[Role.HOST], local=Role.HOST, peer=Role.MIDDLE,
                    run=RUN, plane="control", timeout=2)
        result.result(timeout=4)


@pytest.mark.parametrize("wrong", ["run", "plane"])
def test_channel_binding_rejected(certificates, wrong):
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        def server():
            with pytest.raises(ValidationError):
                accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                       run=RUN, plane="control", timeout=2)
        result = pool.submit(server)
        with pytest.raises((OSError, EOFError, ValidationError)):
            connect(listener.getsockname(), certificates[Role.HOST], local=Role.HOST, peer=Role.MIDDLE,
                    run="b" * 64 if wrong == "run" else RUN,
                    plane="data" if wrong == "plane" else "control", timeout=2)
        result.result(timeout=4)


def test_oversized_encrypted_frame_closes_channel_before_body(certificates):
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        def server():
            connection = accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                                run=RUN, plane="control", timeout=2)
            with pytest.raises(ValidationError):
                connection.receive()
            assert connection.closed
        result = pool.submit(server)
        with connect(listener.getsockname(), certificates[Role.HOST], local=Role.HOST, peer=Role.MIDDLE,
                     run=RUN, plane="control", timeout=2) as client:
            client.socket.sendall(PREFIX.pack(b"DSRP", 1, 0, Kind.BEGIN, 10, MAX_PAYLOAD + 1))
            result.result(timeout=4)


def test_partial_frame_deadline_invalidates_channel(certificates):
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        def server():
            connection = accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                                run=RUN, plane="control", timeout=2)
            connection.timeout = 0.15
            with pytest.raises(TimeoutError):
                connection.receive()
            assert connection.closed
        result = pool.submit(server)
        with connect(listener.getsockname(), certificates[Role.HOST], local=Role.HOST, peer=Role.MIDDLE,
                     run=RUN, plane="control", timeout=2) as client:
            client.socket.sendall(b"DS")
            result.result(timeout=4)


@pytest.mark.parametrize("address", ["0.0.0.0", "224.0.0.1", "8.8.8.8"])
def test_listener_refuses_unconfined_addresses(address):
    with pytest.raises(ValidationError):
        listen(address, 0)


def test_control_owner_sends_execution_reply_and_cancellation_while_a_task_is_pending(certificates):
    """Real TLS and a blocked work fixture; no GPU/model emulation."""
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        cancelled, release_task, started_task = threading.Event(), threading.Event(), threading.Event()
        def server():
            connection = accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                                run=RUN, plane="control", timeout=2)
            tasks = []
            def handle(message):
                if message.kind == Kind.STEP_BEGIN:
                    def task():
                        started_task.set()
                        assert release_task.wait(3)
                        pump.send(replace(message, kind=Kind.STAGE_DONE, sender=Role.MIDDLE,
                                          receiver=Role.HOST)).result(timeout=2)
                    thread = threading.Thread(target=task)
                    tasks.append(thread)
                    thread.start()
                elif message.kind == Kind.CANCEL:
                    cancelled.set()
                    pump.send(replace(message, kind=Kind.CANCEL_ACK, sender=Role.MIDDLE, receiver=Role.HOST))
                else:
                    raise AssertionError("unexpected control fixture message")
            pump = ControlPump(connection, handle).start()
            assert cancelled.wait(4)
            release_task.set()
            for task in tasks:
                task.join(timeout=3)
                assert not task.is_alive()
            pump.close()
        result = pool.submit(server)
        with connect(listener.getsockname(), certificates[Role.HOST], local=Role.HOST, peer=Role.MIDDLE,
                     run=RUN, plane="control", timeout=3) as client:
            client.send(frame(Role.HOST, Role.MIDDLE, Kind.STEP_BEGIN))
            assert started_task.wait(3)
            client.send(frame(Role.HOST, Role.MIDDLE, Kind.CANCEL))
            kinds = {client.receive().kind, client.receive().kind}
            assert kinds == {Kind.CANCEL_ACK, Kind.STAGE_DONE}
        result.result(timeout=5)


def test_control_queue_has_bounded_backpressure_and_close_resolves_unsent_messages(certificates):
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        def server():
            connection = accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                                run=RUN, plane="control", timeout=2)
            pump = ControlPump(connection, lambda _: None, queue_size=1)
            queued = pump.send(frame(Role.MIDDLE, Role.HOST))
            with pytest.raises(ValidationError, match="queue is full"):
                pump.send(frame(Role.MIDDLE, Role.HOST))
            pump.close()
            with pytest.raises(ConnectionError, match="before startup"):
                queued.result()
        result = pool.submit(server)
        with connect(listener.getsockname(), certificates[Role.HOST], local=Role.HOST, peer=Role.MIDDLE,
                     run=RUN, plane="control", timeout=2):
            result.result(timeout=4)


def test_correlated_rpc_keeps_cancel_and_error_responses_independent_of_queued_work(certificates):
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        release, started = threading.Event(), threading.Event()
        tasks = []
        def dispatch(message, body, reply):
            if body["command"] == "task":
                def task():
                    started.set()
                    assert release.wait(3)
                    reply({"result": 17})
                thread = threading.Thread(target=task)
                tasks.append(thread)
                thread.start()
            elif message.kind == Kind.CANCEL:
                reply({"signalled": True})
                release.set()
            else:
                reply(error="SESSION_FAULT")
        def server():
            connection = accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                                run=RUN, plane="control", timeout=2)
            return ControlRPC(connection, dispatch).start()
        connected = pool.submit(server)
        client_connection = connect(listener.getsockname(), certificates[Role.HOST], local=Role.HOST,
                                    peer=Role.MIDDLE, run=RUN, plane="control", timeout=2)
        remote = connected.result(timeout=3)
        client = ControlRPC(client_connection, lambda *_: None).start()
        try:
            result = client.submit(frame(Role.HOST, Role.MIDDLE, Kind.STEP_BEGIN), {"command": "task"})
            assert started.wait(3)
            assert client.call(frame(Role.HOST, Role.MIDDLE, Kind.CANCEL), {"command": "cancel"}, timeout=2) == {"signalled": True}
            assert client.wait(result, timeout=2) == {"result": 17}
            with pytest.raises(RemoteError, match="SESSION_FAULT"):
                client.call(frame(Role.HOST, Role.MIDDLE), {"command": "fault"}, timeout=2)
        finally:
            release.set()
            for task in tasks:
                task.join(timeout=3)
            client.close()
            remote.close()


def test_rpc_timeout_closes_uncertain_channel_and_resolves_other_pending_requests(certificates):
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        received = threading.Event()
        def server():
            connection = accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                                run=RUN, plane="control", timeout=2)
            return ControlRPC(connection, lambda *_: received.set()).start()
        connected = pool.submit(server)
        client_connection = connect(listener.getsockname(), certificates[Role.HOST], local=Role.HOST,
                                    peer=Role.MIDDLE, run=RUN, plane="control", timeout=2)
        remote = connected.result(timeout=3)
        client = ControlRPC(client_connection, lambda *_: None).start()
        try:
            first = client.submit(frame(Role.HOST, Role.MIDDLE), {"command": "no_reply"})
            second = client.submit(frame(Role.HOST, Role.MIDDLE), {"command": "also_waiting"})
            assert received.wait(3)
            with pytest.raises(TimeoutError, match="outcome is unknown"):
                client.wait(first, timeout=.05)
            with pytest.raises(ConnectionError, match="outcome is unknown"):
                second.result(timeout=1)
            assert client_connection.closed
            with pytest.raises(ConnectionError, match="closed"):
                client.submit(frame(Role.HOST, Role.MIDDLE), {"command": "cannot_continue"})
        finally:
            client.close()
            remote.close()


def test_mesh_listener_selects_certificate_role_and_plane_without_assuming_peer_start_order(certificates):
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        def server():
            observed = []
            for _ in range(4):
                with accept_authenticated(listener, certificates[Role.TERMINAL], local=Role.TERMINAL,
                                          allowed_peers={Role.HOST, Role.MIDDLE}, run=RUN, timeout=2) as channel:
                    observed.append((channel.peer, channel.plane))
            return observed
        result = pool.submit(server)
        expected = [(Role.HOST, "data"), (Role.MIDDLE, "control"), (Role.HOST, "control"), (Role.MIDDLE, "data")]
        for role, plane in expected:
            with connect(listener.getsockname(), certificates[role], local=role, peer=Role.TERMINAL,
                         run=RUN, plane=plane, timeout=2):
                pass
        assert result.result(timeout=4) == expected
