"""Persistent native TCP, standard mutual TLS, whole-frame deadlines."""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import socket
import ssl
import threading
import time
from . import metrics

from .jsonutil import ValidationError, canonical, loads
from .planner import Role
from .protocol import DATA_KINDS, Frame, Kind, PREFIX, decode, hex_id, parse_prefix


@dataclass(frozen=True)
class TLSFiles:
    ca: Path
    certificate: Path
    private_key: Path

    def context(self, *, server: bool) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if server else ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_verify_locations(cafile=self.ca)
        context.load_cert_chain(certfile=self.certificate, keyfile=self.private_key)
        return context


def check_peer(sock: ssl.SSLSocket, expected: Role):
    stage_names = {r.dns_name for r in Role}
    offered = {value for kind, value in sock.getpeercert().get("subjectAltName", ())
               if kind == "DNS" and value in stage_names}
    if offered != {expected.dns_name}:
        raise ValidationError("authenticated peer certificate has the wrong stage identity")


def _hello_metadata(frame):
    zeros = "0" * 32
    if (frame.kind != Kind.HELLO or (frame.session, frame.request, frame.transaction) != (zeros, zeros, zeros) or
            any(getattr(frame, key) for key in ("epoch", "operation", "sequence", "position", "rows", "slot_generation"))):
        raise ValidationError("channel HELLO has an invalid initial timeline")
    message = loads(frame.payload, limit=256)
    if (not isinstance(message, dict) or set(message) not in ({"plane"}, {"plane", "mesh"}) or not isinstance(message["plane"], str) or
            message["plane"] not in {"control", "data"}):
        raise ValidationError("invalid channel HELLO plane")
    mesh = message.get("mesh")
    if "mesh" in message:
        hex_id(mesh, 32, "channel mesh incarnation")
    return message["plane"], mesh


def listen(address: str, port: int) -> socket.socket:
    ip = ipaddress.IPv4Address(address)
    if ip.is_unspecified or ip.is_multicast or not (ip.is_private or ip.is_loopback):
        raise ValidationError("worker listener needs an explicit private or loopback IPv4 address")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Windows must not permit another process to reuse this bound endpoint.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((address, port))
        sock.listen(4)
        return sock
    except BaseException:
        sock.close()
        raise


class Connection:
    def __init__(self, sock: ssl.SSLSocket, *, local: Role, peer: Role, run: str,
                 plane: str, timeout: float = 10, mesh: str | None = None):
        if plane not in {"data", "control"} or not 0 < timeout <= 300:
            raise ValidationError("invalid channel plane or deadline")
        check_peer(sock, peer)
        if mesh is not None:
            hex_id(mesh, 32, "channel mesh incarnation")
        self.socket, self.local, self.peer, self.run = sock, local, peer, run
        self.plane, self.timeout, self.mesh = plane, timeout, mesh
        self.closed = False
        # Blocking channel I/O has one owner at a time. Control and data use
        # separate connections so a bulk read never owns the control lock.
        self._io_lock = threading.RLock()
        self.bytes_sent = self.bytes_received = 0

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.socket.close()

    def _remaining(self, deadline: float):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("RPC frame deadline exceeded")
        self.socket.settimeout(remaining)

    def _read(self, count: int, deadline: float) -> bytes:
        data = bytearray()
        while len(data) < count:
            self._remaining(deadline)
            part = self.socket.recv(count - len(data))
            if not part:
                raise EOFError("peer disconnected during RPC frame")
            data.extend(part)
            metrics.current.add("rpc_receive_bytes", len(part))
        self.bytes_received += len(data)
        return bytes(data)

    def send(self, frame: Frame):
        if self.closed:
            raise ConnectionError("RPC channel is closed")
        if (frame.run, frame.sender, frame.receiver) != (self.run, self.local, self.peer):
            raise ValidationError("outgoing frame identity differs from authenticated channel")
        if (frame.kind in DATA_KINDS) != (self.plane == "data") and frame.kind != Kind.HELLO:
            raise ValidationError("message is on the wrong channel plane")
        started = time.perf_counter_ns()
        try:
            with metrics.current.measure("rpc_encode"):
                chunks = frame.encode()
            with self._io_lock:
                deadline = time.monotonic() + self.timeout
                for chunk in chunks:
                    self._remaining(deadline)
                    self.socket.sendall(chunk)
                    self.bytes_sent += len(chunk)
                    metrics.current.add("rpc_send_bytes", len(chunk))
            metrics.current.add("rpc_send_frames")
            metrics.current.event("frame", peer=int(self.peer), kind=int(frame.kind), rows=frame.rows,
                bytes=sum(len(chunk) for chunk in chunks), elapsed_ns=time.perf_counter_ns() - started)
        except BaseException:
            self.close()
            raise
        finally:
            metrics.current.observe("rpc_send", time.perf_counter_ns() - started)

    def receive(self) -> Frame:
        with metrics.current.measure("rpc_receive"), self._io_lock:
            return self._receive_locked()

    def _receive_locked(self) -> Frame:
        if self.closed:
            raise ConnectionError("RPC channel is closed")
        deadline = time.monotonic() + self.timeout
        try:
            kind, header_bytes, payload_bytes = parse_prefix(self._read(PREFIX.size, deadline))
            header, payload = self._read(header_bytes, deadline), self._read(payload_bytes, deadline)
            with metrics.current.measure("rpc_decode"):
                frame = decode(kind, header, payload)
            if (frame.run, frame.sender, frame.receiver) != (self.run, self.peer, self.local):
                raise ValidationError("incoming frame identity differs from authenticated channel")
            if (frame.kind in DATA_KINDS) != (self.plane == "data") and frame.kind != Kind.HELLO:
                raise ValidationError("message is on the wrong channel plane")
            metrics.current.add("rpc_receive_frames")
            return frame
        except BaseException:
            self.close()
            raise

    def hello(self):
        zeros = "0" * 32
        try:
            metadata = {"plane": self.plane}
            if self.mesh is not None:
                metadata["mesh"] = self.mesh
            self.send(Frame(Kind.HELLO, self.run, self.local, self.peer, zeros, zeros,
                            0, 0, 0, zeros, payload=canonical(metadata)))
            if _hello_metadata(self.receive()) != (self.plane, self.mesh):
                raise ValidationError("channel HELLO mismatch")
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def connect(address: tuple[str, int], tls: TLSFiles, *, local: Role, peer: Role,
            run: str, plane: str, timeout: float = 10, mesh: str | None = None) -> Connection:
    raw = socket.create_connection(address, timeout=timeout)
    try:
        sock = tls.context(server=False).wrap_socket(raw, server_hostname=peer.dns_name)
        connection = Connection(sock, local=local, peer=peer, run=run, plane=plane, timeout=timeout, mesh=mesh)
        connection.hello()
        return connection
    except BaseException:
        raw.close()
        if "sock" in locals():
            sock.close()
        raise


def accept(listener: socket.socket, tls: TLSFiles, *, local: Role, peer: Role,
           run: str, plane: str, timeout: float = 10, mesh: str | None = None) -> Connection:
    listener.settimeout(timeout)
    raw, _ = listener.accept()
    try:
        raw.settimeout(timeout)
        sock = tls.context(server=True).wrap_socket(raw, server_side=True)
        connection = Connection(sock, local=local, peer=peer, run=run, plane=plane, timeout=timeout, mesh=mesh)
        connection.hello()
        return connection
    except BaseException:
        raw.close()
        if "sock" in locals():
            sock.close()
        raise


def accept_authenticated(listener: socket.socket, tls: TLSFiles, *, local: Role,
                         allowed_peers: set[Role], run: str, timeout: float = 10) -> Connection:
    """Accept either plane from an allowed certificate, independent of startup order.

    The caller must pair and deduplicate (peer, plane) within this configured
    run. No stage identity is accepted from plaintext or caller-supplied JSON.
    """
    if (not isinstance(local, Role) or not isinstance(allowed_peers, set) or not allowed_peers or
            any(not isinstance(role, Role) or role == local for role in allowed_peers) or not 0 < timeout <= 300):
        raise ValidationError("listener requires distinct explicit allowed stage certificates")
    listener.settimeout(timeout)
    raw, _ = listener.accept()
    try:
        raw.settimeout(timeout)
        sock = tls.context(server=True).wrap_socket(raw, server_side=True)
        names = {value for kind, value in sock.getpeercert().get("subjectAltName", ())
                 if kind == "DNS" and value in {role.dns_name for role in Role}}
        if len(names) != 1:
            raise ValidationError("authenticated certificate must identify exactly one stage")
        peer, = [role for role in Role if role.dns_name in names]
        if peer not in allowed_peers:
            raise ValidationError("authenticated stage is not allowed on this worker listener")
        # HELLO is exempt from plane dispatch until its authenticated content
        # selects this channel's immutable control/data plane.
        connection = Connection(sock, local=local, peer=peer, run=run, plane="control", timeout=timeout)
        connection.plane, connection.mesh = _hello_metadata(connection.receive())
        zeros = "0" * 32
        metadata = {"plane": connection.plane}
        if connection.mesh is not None:
            metadata["mesh"] = connection.mesh
        connection.send(Frame(Kind.HELLO, run, local, peer, zeros, zeros, 0, 0, 0, zeros,
                              payload=canonical(metadata)))
        return connection
    except BaseException:
        raw.close()
        if "sock" in locals():
            sock.close()
        raise
