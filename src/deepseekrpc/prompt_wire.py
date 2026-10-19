"""Full prompt metadata on the data plane, with locally bounded allocations."""
from dataclasses import asdict, dataclass
from functools import wraps
import hashlib

import numpy as np

from .fragments import FragmentReceiver, fragments
from .history import ImageSpan, TokenHistory
from .jsonutil import ValidationError, canonical, fingerprint, integer, loads
from .planner import Role
from .protocol import Frame, Kind, TensorDescriptor, hex_id
from .runtime_messages import fields

MAX_IMAGE_REFS = 512 * 1024


def _close_on_failure(function):
    @wraps(function)
    def transfer(connection, *args, **kwargs):
        try:
            return function(connection, *args, **kwargs)
        except BaseException:
            # No next command may interpret an unread old fragment as a new
            # prompt/window. The persistent model can survive mesh recovery.
            connection.close()
            raise
    return transfer


@dataclass(frozen=True)
class PromptSpec:
    session: str
    request: str
    epoch: int
    transaction: str
    identity: str
    history_digest: str
    tokens: int
    image_count: int
    image_bytes: int
    image_sha256: str

    def __post_init__(self):
        for key in ("session", "request", "transaction"):
            hex_id(getattr(self, key), 32, key)
        for key in ("identity", "history_digest", "image_sha256"):
            hex_id(getattr(self, key), 64, key)
        integer(self.epoch, "new prompt epoch", 0, (1 << 63) - 1)
        integer(self.tokens, "full prompt token count", 1, 1048576)
        integer(self.image_count, "full prompt image count", 0, 4096)
        integer(self.image_bytes, "full prompt image reference bytes", 2, MAX_IMAGE_REFS)

    @classmethod
    def from_wire(cls, value):
        return cls(**fields(value, cls.__dataclass_fields__))

    @classmethod
    def from_history(cls, history, *, session, request, epoch, transaction):
        metadata = history.image_manifest()
        return cls(session, request, epoch, transaction, history.identity, history.digest,
                   history.length, len(history.images), len(metadata), hashlib.sha256(metadata).hexdigest())

    def descriptors(self):
        return tuple(TensorDescriptor(semantic, -1, dtype, (count,), 0,
                                      fingerprint({"prompt": asdict(self), "field": semantic})[:32], 0, 0)
                     for semantic, dtype, count in (("token_ids", "i32", self.tokens),
                                                   ("image_mask", "u8", self.tokens),
                                                   ("image_refs", "u8", self.image_bytes)))

    def frame(self, run, sender, receiver, descriptor):
        return Frame(Kind.STAGE_INPUT, run, sender, receiver, self.session, self.request,
                     self.epoch, 0, 0, self.transaction, tensor=descriptor)


@_close_on_failure
def send_prompt(connection, spec, history, *, cancelled=lambda: False):
    if (connection.local != Role.HOST or connection.peer == Role.HOST or connection.plane != "data" or
            history.identity != spec.identity or history.digest != spec.history_digest or
            history.length != spec.tokens or len(history.images) != spec.image_count):
        raise ValidationError("prompt sender differs from its admitted full token/image history")
    tokens, mask, _ = history.export_prompt()
    metadata = history.image_manifest()
    if len(metadata) != spec.image_bytes or hashlib.sha256(metadata).hexdigest() != spec.image_sha256:
        raise ValidationError("prompt image metadata differs from its prepared descriptor")
    storage = (tokens.astype("<i4", copy=False), mask.view(np.uint8), metadata)
    for descriptor, value in zip(spec.descriptors(), storage):
        template = spec.frame(connection.run, connection.local, connection.peer, descriptor)
        for fragment in fragments(template, descriptor, value):
            if cancelled():
                raise InterruptedError("full prompt transfer cancelled")
            connection.send(fragment)


@_close_on_failure
def receive_prompt(connection, spec, *, identity, cancelled=lambda: False):
    if (connection.peer != Role.HOST or connection.local == Role.HOST or
            connection.plane != "data" or spec.identity != identity):
        raise ValidationError("prompt receiver differs from the admitted model/stage identity")
    # These sizes were admitted from strict PromptSpec hard bounds, before a
    # peer frame is read. At most 5 MiB token/mask plus 512 KiB image metadata.
    tokens = np.empty(spec.tokens, dtype="<i4")
    mask = np.empty(spec.tokens, dtype=np.uint8)
    metadata = bytearray(spec.image_bytes)
    for descriptor, value in zip(spec.descriptors(), (tokens, mask, metadata)):
        template = spec.frame(connection.run, connection.peer, connection.local, descriptor)
        receiver = FragmentReceiver(template, descriptor, value)
        while not receiver.complete:
            if cancelled():
                raise InterruptedError("full prompt transfer cancelled")
            receiver.accept(connection.receive())
    if np.any(mask > 1) or hashlib.sha256(metadata).hexdigest() != spec.image_sha256:
        raise ValidationError("prompt mask or image reference digest is invalid")
    raw = loads(metadata, limit=MAX_IMAGE_REFS)
    if not isinstance(raw, list) or len(raw) != spec.image_count:
        raise ValidationError("prompt image reference count differs from its admitted inventory")
    images = tuple(ImageSpan(**fields(value, ImageSpan.__dataclass_fields__)) for value in raw)
    # This performs complete token, span/mask and rolling history validation
    # before reset can mutate live GPU state.
    history = TokenHistory(identity)
    if history.set_prompt(tokens, mask.view(np.bool_), images) != spec.history_digest:
        raise ValidationError("received full prompt differs from the exact history digest")
    return history
