from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import socket
import ssl
import time

import pytest

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.planner import Role
from deepseekrpc.protocol import MAX_PAYLOAD, PREFIX, Frame, Kind
from deepseekrpc.transport import accept, connect, listen

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
