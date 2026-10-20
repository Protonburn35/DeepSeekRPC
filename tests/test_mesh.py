"""Actual three-role TLS startup/recovery; no model or GPU emulation."""
import socket
import time

import pytest

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.mesh import MeshNetwork
from deepseekrpc.planner import Role
from deepseekrpc.protocol import Frame, Kind
from deepseekrpc.transport import connect

RUN, ZERO = "b" * 64, "0" * 32


def free_endpoint():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()


def until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.025)
    raise AssertionError("real TLS mesh did not reach the expected bounded state")


def test_three_role_loading_controls_and_new_incarnation_after_interruption(certificates):
    endpoints = {role: free_endpoint() for role in (Role.MIDDLE, Role.TERMINAL)}
    networks = {}
    try:
        for role in (Role.TERMINAL, Role.MIDDLE, Role.HOST):
            def status(role=role):
                return {"stage": int(role), "run": RUN, "status": "LOADING"}
            networks[role] = MeshNetwork(role=role, run=RUN, tls=certificates[role], endpoints=endpoints,
                                         status=status, timeout=1, retry=.05).start()
        until(lambda: all(n.current is not None and n.current.complete for n in networks.values()))
        old = {role: n.current for role, n in networks.items()}
        assert len({mesh.nonce for mesh in old.values()}) == 1
        for network in networks.values():
            network.heartbeat(network.current, timeout=2)
        rpc = old[Role.HOST].controls[Role.MIDDLE]
        frame = Frame(Kind.CAPABILITIES, RUN, Role.HOST, Role.MIDDLE, ZERO, ZERO, 0, 0, 0, ZERO)
        assert rpc.call(frame, {"action": "status"}, timeout=2) == {
            "stage": int(Role.MIDDLE), "run": RUN, "status": "LOADING", "transport_ready": False}
        # A request cancellation closes data to interrupt a blocked transfer.
        data, _ = old[Role.HOST].resources()
        data[Role.MIDDLE].close()
        until(lambda: all(n.current is not old[role] and n.current.complete for role, n in networks.items()))
        assert len({n.current.nonce for n in networks.values()}) == 1
        assert all(not mesh.healthy for mesh in old.values())
        for network in networks.values():
            network.heartbeat(network.current, timeout=2)
        current = networks[Role.TERMINAL].current
        # A trusted but stale incarnation must not replace the live one.
        stale = connect(endpoints[Role.TERMINAL], certificates[Role.HOST], local=Role.HOST,
                        peer=Role.TERMINAL, run=RUN, plane="data", mesh=old[Role.HOST].nonce, timeout=1)
        try:
            assert stale.socket.recv(1) == b""
        finally:
            stale.close()
        assert networks[Role.TERMINAL].current is current and current.complete
    finally:
        for network in reversed(list(networks.values())):
            network.close()


def test_worker_cannot_select_an_incarnation_and_duplicate_channel_does_not_replace_it(certificates):
    endpoints = {role: free_endpoint() for role in (Role.MIDDLE, Role.TERMINAL)}
    network = MeshNetwork(role=Role.TERMINAL, run=RUN, tls=certificates[Role.TERMINAL], endpoints=endpoints,
                          status=lambda: {"stage": 2, "run": RUN, "status": "LOADING"}, timeout=1, retry=.05).start()
    channels = []
    try:
        for role, mesh in ((Role.MIDDLE, "a" * 32), (Role.HOST, "c" * 32), (Role.HOST, "c" * 32)):
            connection = connect(endpoints[Role.TERMINAL], certificates[role], local=role, peer=Role.TERMINAL,
                                 run=RUN, plane="data", mesh=mesh, timeout=1)
            channels.append(connection)
            if role == Role.MIDDLE or len(channels) == 3:
                assert connection.socket.recv(1) == b""
            else:
                until(lambda: network.current is not None and network.current.has(Role.HOST, "data"))
        assert network.current.nonce == "c" * 32
        assert network.current.healthy
    finally:
        for connection in channels:
            connection.close()
        network.close()


def test_mesh_rejects_unconfined_endpoint_before_opening_a_listener(certificates):
    with pytest.raises(ValidationError, match="explicit private"):
        MeshNetwork(role=Role.MIDDLE, run=RUN, tls=certificates[Role.MIDDLE],
                    endpoints={Role.MIDDLE: ("0.0.0.0", 39011), Role.TERMINAL: ("127.0.0.1", 39012)},
                    status=lambda: {})
