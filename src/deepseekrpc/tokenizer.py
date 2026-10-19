"""Pinned local tokenizer input; never downloads a checkpoint or truncates text."""
from pathlib import Path
import hashlib

from .jsonutil import ValidationError
from .protocol import hex_id


class TokenizerAdapter:
    def __init__(self, path: Path, *, expected_sha256: str, expected_vocab: int = 129280):
        from tokenizers import Tokenizer
        hex_id(expected_sha256, 64, "tokenizer")
        with path.open("rb") as stream:
            data = stream.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValidationError("tokenizer metadata size or SHA-256 mismatch")
        self.backend_tokenizer = Tokenizer.from_str(data.decode("utf-8"))
        if self.backend_tokenizer.truncation is not None or self.backend_tokenizer.padding is not None:
            raise ValidationError("checkpoint tokenizer must not silently truncate or pad")
        self.vocab_size = self.backend_tokenizer.get_vocab_size(with_added_tokens=True)
        if self.vocab_size != expected_vocab:
            raise ValidationError("tokenizer vocabulary differs from the model")
        self.sha256 = expected_sha256

    def __len__(self):
        return self.vocab_size

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str) or len(text.encode("utf-8")) > 64 * 1024 * 1024:
            raise ValidationError("tokenizer input is not bounded UTF-8 text")
        # The official renderer supplies BOS and other special markers itself.
        return self.backend_tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: list[int]) -> str:
        return self.backend_tokenizer.decode(tokens, skip_special_tokens=False)
