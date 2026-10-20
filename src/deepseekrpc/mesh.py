"""Persistent authenticated mesh incarnations, independent of model loading.

A chooses an incarnation and connects C before B. B then connects C using
that same incarnation. Every edge has separate control/data TLS connections.
An interrupted edge retires the entire incarnation; partial data streams are
never reused. This layer owns sockets, and contains no model implementation.
"""
from __future__ import annotations

import socket
import threading
import uuid

from .control_rpc import ControlRPC
from .jsonutil import ValidationError, integer
from .planner import Role
from .protocol import Frame, Kind, hex_id
from .transport import accept_authenticated, connect, listen


class ControlGate:
    """One control pump from early diagnostics through loaded-stage serving."""
    def __init__(self, status, failed, connection):
        self._status, self._failed, self.connection = status, failed, connection
        self._lock, self._service = threading.Lock(), None

    def attach(self, service):
        with self._lock:
            self._service = service

    def detach(self):
        with self._lock:
            self._service = None

    def failure(self, _):
        self._failed.set()
        with self._lock:
            service = self._service
        if service is not None and service.mesh_nonce == self.connection.mesh:
            service.actor.invalidate()

    def handle(self, frame, body, reply):
        with self._lock:
            service = self._service
        if service is not None:
            service._handle(frame, body, reply, mesh=self.connection.mesh)
        elif (frame.kind, body) == (Kind.CAPABILITIES, {"action": "status"}):
            reply({**self._status(), "transport_ready": False})
        elif (frame.kind, body) == (Kind.HEARTBEAT, {"action": "heartbeat"}):
            reply({"alive": True, **self._status(), "transport_ready": False})
        else:
            reply(error="NOT_READY")


class MeshChannels:
    def __init__(self, role, run, nonce, status):
        self.role, self.run = role, run
        self.nonce = hex_id(nonce, 32, "mesh incarnation")
        self.failed = threading.Event()
        self._lock, self._closed = threading.RLock(), False
        self._connections, self.controls, self.gates = {}, {}, {}
        self._status = status

    def add(self, connection):
        with self._lock:
            key = connection.peer, connection.plane
            if (self._closed or self.failed.is_set() or connection.closed or
                    (connection.local, connection.run, connection.mesh) != (self.role, self.run, self.nonce) or
                    connection.peer == self.role or connection.plane not in {"control", "data"}):
                connection.close()
                raise ValidationError("channel differs from the active authenticated mesh incarnation")
            if key in self._connections:
                connection.close()
                raise ValidationError("duplicate peer/plane in an authenticated mesh")
            self._connections[key] = connection
            if connection.plane == "data":
                # A peer may spend time in a bounded GPU/disk operation before
                # the next tile. Cancellation still closes this socket
                # immediately and has a separate responsive control plane.
                connection.timeout = 120
            if connection.plane == "control":
                gate = ControlGate(self._status, self.failed, connection)
                rpc = ControlRPC(connection, gate.handle, on_failure=gate.failure)
                self.gates[connection.peer], self.controls[connection.peer] = gate, rpc
                rpc.start()

    @property
    def healthy(self):
        with self._lock:
            return not (self._closed or self.failed.is_set() or any(c.closed for c in self._connections.values()))

    @property
    def complete(self):
        with self._lock:
            expected = {(peer, plane) for peer in set(Role) - {self.role} for plane in ("control", "data")}
            return self.healthy and set(self._connections) == expected

    def has(self, peer, plane):
        with self._lock:
            return (peer, plane) in self._connections

    def resources(self):
        with self._lock:
            if not self.complete:
                raise ConnectionError("all four authenticated channels must be healthy before model binding")
            return ({peer: self._connections[peer, "data"] for peer in set(Role) - {self.role}},
                    dict(self.controls))

    def attach(self, service):
        with self._lock:
            if not self.complete:
                raise ConnectionError("mesh was interrupted before loaded-stage attachment")
            for gate in self.gates.values():
                gate.attach(service)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Detach before close so deliberately retiring old controls cannot
            # invalidate a newly rebound model actor through a late callback.
            for gate in self.gates.values():
                gate.detach()
            connections, controls = list(self._connections.values()), list(self.controls.values())
        for connection in connections:
            connection.close()
        for rpc in controls:
            rpc.close()


class MeshNetwork:
    """Listeners and retry loops survive ordinary request cancellation.

    Production requires a 128-bit incarnation on all planes. Retired nonces
    are remembered for this process lifetime, with a hard bound; exhausting
    that bound fails explicitly instead of accepting a replayed incarnation.
    """
    def __init__(self, *, role, run, tls, endpoints, status, timeout=3, retry=.5):
        if (not isinstance(role, Role) or set(endpoints) != {Role.MIDDLE, Role.TERMINAL} or
                not callable(status) or not 0 < timeout <= 30 or not .05 <= retry <= 10):
            raise ValidationError("mesh requires explicit B/C endpoints and bounded diagnostics/retry policy")
        hex_id(run, 64, "mesh configured run")
        for address, port in endpoints.values():
            # Reuse the listener address policy without opening a socket.
            import ipaddress
            ip = ipaddress.IPv4Address(address)
            if ip.is_unspecified or ip.is_multicast or not (ip.is_private or ip.is_loopback):
                raise ValidationError("mesh endpoints must be explicit private IPv4 addresses")
            integer(port, "mesh endpoint port", 1, 65535)
        self.role, self.run, self.tls, self.endpoints = role, run, tls, dict(endpoints)
        self.status, self.timeout, self.retry = status, timeout, retry
        self._lock, self._stop = threading.RLock(), threading.Event()
        self._current, self._retired, self._threads = None, set(), []
        self.listener, self.failure = None, None
        self._started = False

    @property
    def current(self):
        with self._lock:
            return self._current

    def _replace(self, nonce):
        # Caller holds the network lock; close happens outside that lock.
        if nonce in self._retired:
            raise ValidationError("retired mesh incarnation cannot be reused")
        old = self._current
        if old is not None:
            if len(self._retired) >= 4096:
                raise RuntimeError("mesh incarnation lifetime limit reached")
            self._retired.add(old.nonce)
        self._current = MeshChannels(self.role, self.run, nonce, self.status)
        return self._current, old

    def _adopt(self, connection):
        old = None
        try:
            with self._lock:
                if self._stop.is_set() or connection.mesh is None:
                    raise ValidationError("production mesh requires an active incarnation")
                current = self._current
                if current is None or current.nonce != connection.mesh:
                    if connection.peer != Role.HOST:
                        raise ValidationError("only authenticated A may select a mesh incarnation")
                    current, old = self._replace(connection.mesh)
                current.add(connection)
        except BaseException:
            connection.close()
            raise
        finally:
            if old is not None:
                old.close()

    def start(self):
        if self._started or self._stop.is_set():
            raise ValidationError("mesh listener is already started or stopped")
        if self.role != Role.HOST:
            self.listener = listen(*self.endpoints[self.role])
            self._threads.append(threading.Thread(target=self._accept_loop, name="deepseekrpc-listener", daemon=True))
        self._threads.append(threading.Thread(target=self._connect_loop, name="deepseekrpc-connect", daemon=True))
        self._started = True
        for thread in self._threads:
            thread.start()
        return self

    def _accept_loop(self):
        allowed = {Role.HOST} if self.role == Role.MIDDLE else {Role.HOST, Role.MIDDLE}
        while not self._stop.is_set():
            try:
                connection = accept_authenticated(self.listener, self.tls, local=self.role, allowed_peers=allowed,
                                                   run=self.run, timeout=self.timeout)
                self._adopt(connection)
            except (OSError, EOFError, TimeoutError, ValidationError):
                # Bad certificates/duplicate/stale peers cannot replace a
                # healthy incarnation. The accepted socket is always closed.
                continue
            except BaseException as exc:
                self.failure = exc
                self._stop.set()

    def _dial(self, mesh, peer, plane):
        connection = connect(self.endpoints[peer], self.tls, local=self.role, peer=peer,
                             run=self.run, plane=plane, mesh=mesh.nonce, timeout=self.timeout)
        with self._lock:
            if self._stop.is_set() or self._current is not mesh:
                connection.close()
                raise ConnectionError("mesh was superseded during connection")
            mesh.add(connection)

    def _connect_loop(self):
        while not self._stop.is_set():
            try:
                if self.role == Role.HOST:
                    mesh = self.current
                    if mesh is None or not mesh.healthy:
                        with self._lock:
                            mesh, old = self._replace(uuid.uuid4().hex)
                        if old is not None:
                            old.close()
                    # C must learn A's incarnation before B connects to C.
                    for peer in (Role.TERMINAL, Role.MIDDLE):
                        for plane in ("control", "data"):
                            if not mesh.has(peer, plane):
                                self._dial(mesh, peer, plane)
                else:
                    mesh = self.current
                    if mesh is not None and not mesh.healthy:
                        mesh.close()  # Closing A's control requests a fresh incarnation.
                    elif mesh is not None and self.role == Role.MIDDLE:
                        for plane in ("control", "data"):
                            if not mesh.has(Role.TERMINAL, plane):
                                self._dial(mesh, Role.TERMINAL, plane)
            except (OSError, EOFError, TimeoutError, ValidationError):
                mesh = self.current
                if self.role == Role.HOST and mesh is not None:
                    mesh.close()
            except BaseException as exc:
                self.failure = exc
                self._stop.set()
            self._stop.wait(self.retry)

    def heartbeat(self, mesh, *, timeout=3):
        """Bounded live-peer probe; silence cannot leave readiness stuck true."""
        if mesh is not self.current or not mesh.complete:
            raise ConnectionError("mesh is incomplete or superseded")
        zero = "0" * 32
        futures = {}
        for peer, rpc in mesh.controls.items():
            frame = Frame(Kind.HEARTBEAT, self.run, self.role, peer, zero, zero, 0, 0, 0, zero)
            futures[peer] = rpc.submit(frame, {"action": "heartbeat"})
        for peer, future in futures.items():
            result = mesh.controls[peer].wait(future, timeout=timeout)
            if result.get("alive") is not True or result.get("stage") != int(peer) or result.get("run") != self.run:
                mesh.failed.set()
                raise ValidationError("heartbeat differs from the authenticated peer/run")

    def close(self):
        self._stop.set()
        if self.listener is not None:
            self.listener.close()
        mesh = self.current
        if mesh is not None:
            mesh.close()
        for thread in self._threads:
            if threading.current_thread() is not thread:
                thread.join(self.timeout * 2 + 1)
                if thread.is_alive():
                    raise TimeoutError("mesh network owner did not stop before its bounded deadline")
