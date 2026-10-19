"""Bounded exact token/image identity and published sampling counts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import struct

import numpy as np

from .jsonutil import ValidationError, canonical, integer
from .protocol import hex_id

_RECORD = np.dtype([("tag", "u1"), ("token", "<u4"), ("text", "u1")])


@dataclass(frozen=True)
class ImageSpan:
    start: int
    length: int
    key: str

    def __post_init__(self):
        integer(self.start, "image span start", 0, 1048575)
        integer(self.length, "image span length", 1, 1048576 - self.start)
        # The key binds original image bytes and exact processor/grid options.
        # Paths, URLs and caller display names are not image identity.
        hex_id(self.key, 64, "original image/processor identity")


def _tokens(tokens, *, maximum=1048576):
    values = np.asarray(tokens)
    if (values.ndim != 1 or not 1 <= len(values) <= maximum or
            values.dtype not in (np.int32, np.int64, np.uint32) or
            np.any(values < 0) or np.any(values >= 129280)):
        raise ValidationError("history needs bounded exact integer token IDs")
    return values.astype(np.int32, copy=False)


class TokenHistory:
    def __init__(self, identity, *, capacity=1048576):
        hex_id(identity, 64, "complete model/tokenizer/precision identity")
        self.identity = identity
        self.capacity = integer(capacity, "history capacity", 1, 1048576)
        self._tokens = np.empty(capacity, dtype=np.int32)
        self._text = np.empty(capacity, dtype=np.bool_)
        self._counts = np.zeros(129280, dtype=np.int32)
        self._generated_counts = np.zeros(129280, dtype=np.int32)
        self.length, self.images = 0, ()
        self._hasher = self._empty_hash()

    def _empty_hash(self):
        return hashlib.sha256(b"DeepSeekRPC/token-image-history/v1\x00" + bytes.fromhex(self.identity))

    def _update(self, hasher, tokens, mask, start, images):
        end = start + len(tokens)
        boundaries = [image for image in images if start <= image.start < end]
        cursor = start
        for image in (*boundaries, None):
            next_position = end if image is None else image.start
            for offset in range(cursor, next_position, 8192):
                count = min(8192, next_position - offset)
                rows = np.empty(count, dtype=_RECORD)
                rows["tag"], rows["token"], rows["text"] = 84, tokens[offset - start:offset - start + count], mask[offset - start:offset - start + count]
                hasher.update(rows.tobytes())
            if image is not None:
                hasher.update(b"I" + struct.pack("<QI", image.start, image.length) + bytes.fromhex(image.key))
            cursor = next_position

    @property
    def digest(self):
        return self._hasher.hexdigest()

    def set_prompt(self, tokens, text_mask=None, images=()):
        values = _tokens(tokens, maximum=self.capacity)
        mask = np.ones(len(values), dtype=np.bool_) if text_mask is None else np.asarray(text_mask)
        if mask.dtype != np.bool_ or mask.shape != values.shape:
            raise ValidationError("token/image history mask must exactly cover the prompt")
        if not isinstance(images, (tuple, list)) or len(images) > 4096 or any(not isinstance(i, ImageSpan) for i in images):
            raise ValidationError("prompt images require a bounded exact span inventory")
        expected, previous = np.ones(len(values), dtype=np.bool_), 0
        for image in images:
            if image.start < previous or image.start + image.length > len(values):
                raise ValidationError("prompt image spans overlap, are unsorted or exceed token history")
            expected[image.start:image.start + image.length] = False
            previous = image.start + image.length
        if not np.array_equal(mask, expected):
            raise ValidationError("prompt image spans do not cover exactly the original image mask")
        if np.any((~mask) & ((values < 129264) | (values > 129268))):
            raise ValidationError("image-span IDs differ from the original image sentinel range")
        hasher = self._empty_hash()
        self._update(hasher, values, mask, 0, images)
        counts = np.bincount(values, minlength=129280).astype(np.int32)
        self._tokens[:len(values)], self._text[:len(values)] = values, mask
        self.length, self.images, self._hasher = len(values), tuple(images), hasher
        self._counts[:] = counts
        self._generated_counts.fill(0)
        return self.digest

    def rows(self, start, rows):
        integer(start, "history start", 0, self.length)
        integer(rows, "history rows", 1, min(64, self.length - start))
        return (tuple(int(v) for v in self._tokens[start:start + rows]),
                tuple(bool(v) for v in self._text[start:start + rows]))

    def counts(self):
        return self._counts.copy()

    def generated_counts(self):
        return self._generated_counts.copy()

    def prefix_digest(self, length):
        integer(length, "exact history prefix", 0, self.length)
        # A prefix ending inside an image cannot claim to contain that image's
        # whole identity/span. Reuse must select an earlier complete boundary.
        if any(image.start < length < image.start + image.length for image in self.images):
            raise ValidationError("an exact prefix cannot end inside an original image span")
        hasher = self._empty_hash()
        self._update(hasher, self._tokens[:length], self._text[:length], 0, self.images)
        return hasher.hexdigest()

    def preview_append(self, tokens):
        values = _tokens(tokens, maximum=min(64, self.capacity - self.length))
        hasher = self._hasher.copy()
        self._update(hasher, values, np.ones(len(values), dtype=np.bool_), self.length, ())
        return hasher.hexdigest()

    def append(self, tokens, *, expected_digest):
        hex_id(expected_digest, 64, "published append")
        values = _tokens(tokens, maximum=min(64, self.capacity - self.length))
        hasher = self._hasher.copy()
        self._update(hasher, values, np.ones(len(values), dtype=np.bool_), self.length, ())
        if hasher.hexdigest() != expected_digest:
            raise ValidationError("published token append differs from the all-stage history decision")
        end = self.length + len(values)
        self._tokens[self.length:end], self._text[self.length:end] = values, True
        np.add.at(self._counts, values, 1)
        np.add.at(self._generated_counts, values, 1)
        self.length, self._hasher = end, hasher
        return self.digest

    def image_manifest(self):
        return canonical([asdict(image) for image in self.images])

    def export_prompt(self):
        """Bounded CPU token/mask metadata; no hidden states or model bytes."""
        return self._tokens[:self.length].copy(), self._text[:self.length].copy(), self.images
