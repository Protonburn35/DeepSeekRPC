"""Bounded state-only snapshot streaming on the separate authenticated data plane."""
from dataclasses import replace
import hashlib

from .jsonutil import ValidationError, fingerprint, integer
from .protocol import Frame, Kind, MAX_PAYLOAD, TensorDescriptor, hex_id
from .snapshot_archive import MAX_ARCHIVE


def template(connection, identifier, *, sender, session, request, epoch):
    hex_id(identifier, 32, "snapshot transfer")
    if sender not in {connection.local, connection.peer}:
        raise ValidationError("snapshot sender differs from its authenticated channel")
    receiver = connection.peer if sender == connection.local else connection.local
    return Frame(Kind.SNAPSHOT_CHUNK, connection.run, sender, receiver, session, request,
                 epoch, 0, 0, identifier, tensor=TensorDescriptor("snapshot", -1, "u8", (0,), 0, identifier, 0, 0))


def _descriptor(frame, sequence, count):
    identity = fingerprint({"snapshot": frame.transaction, "sender": int(frame.sender),
        "receiver": int(frame.receiver), "session": frame.session, "request": frame.request,
        "epoch": frame.epoch, "sequence": sequence})[:32]
    return TensorDescriptor("snapshot", -1, "u8", (count,), 0, identity, 0, count)


def send_archive(connection, frame, chunks, *, total, cancelled=lambda: False):
    integer(total, "snapshot transfer length", 1, MAX_ARCHIVE)
    if connection.plane != "data" or frame.sender != connection.local:
        raise ValidationError("snapshot export differs from its authenticated data owner")
    digest, written = hashlib.sha256(), 0
    try:
        for sequence, data in enumerate(chunks):
            if cancelled():
                raise InterruptedError("snapshot transfer cancelled")
            count = min(MAX_PAYLOAD, total - written)
            if not count or not isinstance(data, bytes) or len(data) != count:
                raise ValidationError("snapshot source differs from its admitted exact tile schedule")
            connection.send(replace(frame, sequence=sequence, tensor=_descriptor(frame, sequence, count), payload=data))
            digest.update(data)
            written += count
        if written != total:
            raise ValidationError("snapshot source ended before its admitted complete length")
        return {"bytes": written, "sha256": digest.hexdigest()}
    except BaseException:
        connection.close()
        raise


def receive_archive(connection, frame, *, total, sha256=None, cancelled=lambda: False):
    integer(total, "snapshot transfer length", 1, MAX_ARCHIVE)
    if sha256 is not None:
        hex_id(sha256, 64, "snapshot complete transfer checksum")
    if connection.plane != "data" or frame.receiver != connection.local:
        raise ValidationError("snapshot import differs from its authenticated data owner")
    digest, received, sequence, complete = hashlib.sha256(), 0, 0, False
    try:
        while received < total:
            if cancelled():
                raise InterruptedError("snapshot transfer cancelled")
            count = min(MAX_PAYLOAD, total - received)
            actual = connection.receive()
            expected = replace(frame, sequence=sequence, tensor=_descriptor(frame, sequence, count), payload=actual.payload)
            if actual != expected or len(actual.payload) != count:
                raise ValidationError("snapshot tile changed identity, sequence, length or inventory")
            received += count
            sequence += 1
            digest.update(actual.payload)
            yield actual.payload
        if sha256 is not None and digest.hexdigest() != sha256:
            raise ValidationError("snapshot complete stream checksum differs before restore acceptance")
        complete = True
    finally:
        if not complete:
            connection.close()


class TileReader:
    """Read a bounded archive iterator without accumulating its complete state."""
    def __init__(self, chunks):
        self.chunks, self.current, self.offset = iter(chunks), b"", 0

    def read(self, count):
        integer(count, "snapshot buffered read", 0, 8 * 1024 ** 2)
        result = bytearray()
        while len(result) < count:
            if self.offset == len(self.current):
                try:
                    self.current = next(self.chunks)
                except StopIteration:
                    break
                if not isinstance(self.current, bytes) or not 0 < len(self.current) <= MAX_PAYLOAD:
                    raise ValidationError("snapshot reader received an invalid bounded tile")
                self.offset = 0
            length = min(count - len(result), len(self.current) - self.offset)
            result.extend(self.current[self.offset:self.offset + length])
            self.offset += length
        return bytes(result)

    def finish(self):
        if self.offset != len(self.current) or next(self.chunks, None) is not None:
            raise ValidationError("snapshot archive contains unconsumed or trailing bytes")
