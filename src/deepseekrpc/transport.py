"""Persistent native TCP, standard mutual TLS, whole-frame deadlines."""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import socket
import ssl
import threading
import time

from .jsonutil import ValidationError, canonical, loads
from .planner import Role
from .protocol import DATA_KINDS, Frame, Kind, PREFIX, decode, parse_prefix


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
                 plane: str, timeout: float = 10):
        if plane not in {"data", "control"} or not 0 < timeout <= 300:
            raise ValidationError("invalid channel plane or deadline")
        check_peer(sock, peer)
        self.socket, self.local, self.peer, self.run = sock, local, peer, run
        self.plane, self.timeout = plane, timeout
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
        self.bytes_received += len(data)
        return bytes(data)

    def send(self, frame: Frame):
        if self.closed:
            raise ConnectionError("RPC channel is closed")
        if (frame.run, frame.sender, frame.receiver) != (self.run, self.local, self.peer):
            raise ValidationError("outgoing frame identity differs from authenticated channel")
        if (frame.kind in DATA_KINDS) != (self.plane == "data") and frame.kind != Kind.HELLO:
            raise ValidationError("message is on the wrong channel plane")
        try:
            chunks = frame.encode()
            with self._io_lock:
                deadline = time.monotonic() + self.timeout
                for chunk in chunks:
                    self._remaining(deadline)
                    self.socket.sendall(chunk)
                    self.bytes_sent += len(chunk)
        except BaseException:
            self.close()
            raise

    def receive(self) -> Frame:
        with self._io_lock:
            return self._receive_locked()

    def _receive_locked(self) -> Frame:
        if self.closed:
            raise ConnectionError("RPC channel is closed")
        deadline = time.monotonic() + self.timeout
        try:
            kind, header_bytes, payload_bytes = parse_prefix(self._read(PREFIX.size, deadline))
            frame = decode(kind, self._read(header_bytes, deadline), self._read(payload_bytes, deadline))
            if (frame.run, frame.sender, frame.receiver) != (self.run, self.peer, self.local):
                raise ValidationError("incoming frame identity differs from authenticated channel")
            if (frame.kind in DATA_KINDS) != (self.plane == "data") and frame.kind != Kind.HELLO:
                raise ValidationError("message is on the wrong channel plane")
            return frame
        except BaseException:
            self.close()
            raise

    def hello(self):
        zeros = "0" * 32
        self.send(Frame(Kind.HELLO, self.run, self.local, self.peer, zeros, zeros,
                        0, 0, 0, zeros, payload=canonical({"plane": self.plane})))
        frame = self.receive()
        if frame.kind != Kind.HELLO or loads(frame.payload, limit=128) != {"plane": self.plane}:
            self.close()
            raise ValidationError("channel HELLO mismatch")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def connect(address: tuple[str, int], tls: TLSFiles, *, local: Role, peer: Role,
            run: str, plane: str, timeout: float = 10) -> Connection:
    raw = socket.create_connection(address, timeout=timeout)
    try:
        sock = tls.context(server=False).wrap_socket(raw, server_hostname=peer.dns_name)
        connection = Connection(sock, local=local, peer=peer, run=run, plane=plane, timeout=timeout)
        connection.hello()
        return connection
    except BaseException:
        raw.close()
        if "sock" in locals():
            sock.close()
        raise


def accept(listener: socket.socket, tls: TLSFiles, *, local: Role, peer: Role,
           run: str, plane: str, timeout: float = 10) -> Connection:
    listener.settimeout(timeout)
    raw, _ = listener.accept()
    try:
        raw.settimeout(timeout)
        sock = tls.context(server=True).wrap_socket(raw, server_side=True)
        connection = Connection(sock, local=local, peer=peer, run=run, plane=plane, timeout=timeout)
        connection.hello()
        return connection
    except BaseException:
        raw.close()
        if "sock" in locals():
            sock.close()
        raise
