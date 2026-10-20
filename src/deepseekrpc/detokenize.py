"""Exact pinned ByteLevel bytes, incremental UTF-8 and transactional stops."""
import codecs
from dataclasses import dataclass
from functools import lru_cache

from .jsonutil import ValidationError, fingerprint, integer, loads


def _byte_alphabet():
    # GPT-2/Tokenizers ByteLevel's reversible 256-character alphabet.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = visible.copy()
    missing = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            characters.append(256 + missing)
            missing += 1
    return {chr(character): byte for byte, character in zip(visible, characters)}


class TokenBytes:
    def __init__(self, tokenizer):
        from tokenizers import Tokenizer
        backend = tokenizer.backend_tokenizer
        if not isinstance(backend, Tokenizer):
            raise ValidationError("detokenization requires the actual admitted tokenizer")
        settings = loads(backend.decoder.__getstate__(), limit=1024)
        if settings.get("type") != "ByteLevel" or len(tokenizer) != 129280:
            raise ValidationError("streaming requires the pinned complete ByteLevel tokenizer")
        self.tokenizer, self._inverse = tokenizer, _byte_alphabet()
        # Cache ownership is per tokenizer, with bounded entries and bytes per
        # token. No unbounded request/output history is retained here.
        self._lookup = lru_cache(maxsize=4096)(self._token)

    def _token(self, token):
        text = self.tokenizer.backend_tokenizer.id_to_token(token)
        if text is None or len(text) > 65536:
            raise ValidationError("token is absent or exceeds the admitted token byte bound")
        result = bytearray()
        for character in text:
            byte = self._inverse.get(character)
            result.extend(character.encode("utf-8") if byte is None else bytes((byte,)))
        if len(result) > 65536:
            raise ValidationError("decoded token exceeds the admitted byte bound")
        return bytes(result)

    def __call__(self, token):
        integer(token, "detokenized token ID", 0, 129279)
        return self._lookup(token)


def _prefix_table(pattern):
    table = [0] * len(pattern)
    matched = 0
    for index in range(1, len(pattern)):
        while matched and pattern[index] != pattern[matched]:
            matched = table[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
        table[index] = matched
    return table


@dataclass(frozen=True)
class StopPreview:
    before: str
    after: dict
    token_count: int
    matched: str | None
    text: str


class StopTracker:
    """Preview token-aligned stop decisions; commit only after publication.

    UTF-8 fragments and suffixes that could begin a stop remain buffered. The
    visible stream never has to retract a partial stop. Exact token IDs still
    enter history when the string ends inside that token.
    """
    def __init__(self, token_bytes: TokenBytes, stops=()):
        if not isinstance(token_bytes, TokenBytes) or not isinstance(stops, tuple) or len(stops) > 16:
            raise ValidationError("stop tracking needs the actual token bytes and at most 16 immutable strings")
        if (any(not isinstance(s, str) or not 1 <= len(s.encode("utf-8")) <= 4096 for s in stops) or
                len(set(stops)) != len(stops) or sum(len(s.encode("utf-8")) for s in stops) > 16384):
            raise ValidationError("stop strings must be unique bounded nonempty UTF-8 text")
        self.token_bytes, self.stops = token_bytes, stops
        self._tables = tuple(_prefix_table(s) for s in stops)
        self._state = {"pending": "", "tail": "", "prefixes": [0] * len(stops), "ended": False}

    @property
    def identity(self):
        return fingerprint({"algorithm": "ByteLevel-UTF8-token-stops-v1", "tokenizer": self.token_bytes.tokenizer.sha256,
                            "stops": self.stops, "state": self._state})

    def snapshot(self):
        return {**self._state, "prefixes": list(self._state["prefixes"])}

    def restore(self, state):
        self._state = self.validate_snapshot(state, self.stops)

    @staticmethod
    def validate_snapshot(state, stops):
        if (not isinstance(state, dict) or set(state) != {"pending", "tail", "prefixes", "ended"} or
                not isinstance(state["pending"], str) or len(state["pending"]) > 6 or
                not isinstance(state["tail"], str) or len(state["tail"].encode("utf-8")) > 4096 or
                not isinstance(state["prefixes"], list) or len(state["prefixes"]) != len(stops) or
                type(state["ended"]) is not bool):
            raise ValidationError("invalid bounded stop/UTF-8 snapshot")
        try:
            pending = bytes.fromhex(state["pending"])
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            text = decoder.decode(pending, final=False)
        except (ValueError, UnicodeError):
            raise ValidationError("invalid pending UTF-8 snapshot bytes") from None
        if text or decoder.getstate()[0] != pending or pending.hex() != state["pending"]:
            raise ValidationError("stop snapshot does not contain an incomplete canonical UTF-8 suffix")
        expected = []
        for stop, table in ((stop, _prefix_table(stop)) for stop in stops):
            prefix = 0
            for character in state["tail"]:
                while prefix and character != stop[prefix]:
                    prefix = table[prefix - 1]
                if character == stop[prefix]:
                    prefix += 1
                if prefix == len(stop):
                    raise ValidationError("stop snapshot contains an already complete stop in its held tail")
            expected.append(prefix)
        if (any(type(v) is not int for v in state["prefixes"]) or state["prefixes"] != expected or
                len(state["tail"]) != max(expected, default=0) or
                state["ended"] and (pending or state["tail"] or any(expected))):
            raise ValidationError("stop snapshot prefixes disagree with their exact buffered text")
        return {**state, "prefixes": list(expected)}

    def preview(self, tokens):
        if not isinstance(tokens, (tuple, list)) or not 1 <= len(tokens) <= 6 or self._state["ended"]:
            raise ValidationError("stop preview requires one bounded open output token batch")
        state = self.snapshot()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        decoder.setstate((bytes.fromhex(state["pending"]), 0))
        emitted, matched, count = [], None, 0
        for count, token in enumerate(tokens, 1):
            text = decoder.decode(self.token_bytes(token), final=False)
            buffer = state["tail"] + text
            matches = [(buffer.find(stop), index) for index, stop in enumerate(self.stops)]
            match = min((pair for pair in matches if pair[0] >= 0), default=None)
            if match is not None:
                offset, index = match
                emitted.append(buffer[:offset])
                matched = self.stops[index]
                state["tail"], state["ended"] = "", True
                state["prefixes"] = [0] * len(self.stops)
                break
            for index, (stop, table) in enumerate(zip(self.stops, self._tables)):
                prefix = state["prefixes"][index]
                for character in text:
                    while prefix and character != stop[prefix]:
                        prefix = table[prefix - 1]
                    if character == stop[prefix]:
                        prefix += 1
                    # A full match would have been found in the exact buffer
                    # above. Keeping this guard also catches bad restore state.
                    if prefix == len(stop):
                        raise ValidationError("stop prefix state differs from its buffered exact text")
                state["prefixes"][index] = prefix
            keep = max(state["prefixes"], default=0)
            emitted.append(buffer[:-keep] if keep else buffer)
            state["tail"] = buffer[-keep:] if keep else ""
        state["pending"] = "" if state["ended"] else decoder.getstate()[0].hex()
        return StopPreview(self.identity, state, count, matched, "".join(emitted))

    def commit(self, preview: StopPreview):
        if not isinstance(preview, StopPreview) or preview.before != self.identity:
            raise ValidationError("stop/UTF-8 publication belongs to stale or different committed output")
        self._state = {**preview.after, "prefixes": list(preview.after["prefixes"])}
        return preview.text

    def finish(self):
        """Flush a completed non-stop stream's remaining valid text/UTF-8 tail."""
        if self._state["ended"]:
            return ""
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        decoder.setstate((bytes.fromhex(self._state["pending"]), 0))
        result = self._state["tail"] + decoder.decode(b"", final=True)
        self._state = {"pending": "", "tail": "", "prefixes": [0] * len(self.stops), "ended": True}
        return result
