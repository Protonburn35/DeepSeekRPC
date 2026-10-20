"""Real Rust tokenizer comparisons and stop/verification metadata checks."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from deepseekrpc.detokenize import StopTracker, TokenBytes
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.sampling import SamplingPolicy
from deepseekrpc.terminal_sampling import TerminalSampler
from deepseekrpc.tokenizer import TokenizerAdapter


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    root = Path(__file__).parents[1]
    path = tmp_path_factory.mktemp("stream-tokenizer") / "tokenizer.json"
    path.write_bytes(gzip.decompress((root / "tests/fixtures/tokenizer.json.gz").read_bytes()))
    pin = json.loads((root / "provenance/token-map.lock.json").read_bytes())
    return TokenizerAdapter(path, expected_sha256=pin["tokenizer_sha256"])


def test_all_vocabulary_byte_decoding_matches_actual_rust_tokenizer(tokenizer):
    token_bytes = TokenBytes(tokenizer)
    tokens = list(range(len(tokenizer)))
    # Includes every special marker and byte token. Decode the whole byte
    # stream so incomplete UTF-8 bytes join across vocabulary-token boundaries.
    actual = b"".join(token_bytes(token) for token in tokens).decode("utf-8", errors="replace")
    assert actual == tokenizer.decode(tokens)
    assert token_bytes._lookup.cache_info().currsize <= 4096


def test_stop_snapshot_preserves_held_prefix_and_rejects_inconsistent_restore(tokenizer):
    first = StopTracker(TokenBytes(tokenizer), ("STOP", "TOPIC"))
    for token in tokenizer.encode("hello ST"):
        first.commit(first.preview((token,)))
    saved = first.snapshot()
    restored = StopTracker(TokenBytes(tokenizer), first.stops)
    restored.restore(saved)
    assert restored.identity == first.identity
    for token in tokenizer.encode("OP trailing"):
        a, b = first.preview((token,)), restored.preview((token,))
        assert a == b
        first.commit(a)
        restored.commit(b)
        if a.matched:
            break
    broken = {**saved, "prefixes": [0, 0]}
    with pytest.raises(ValidationError, match="prefixes"):
        restored.restore(broken)
    with pytest.raises(ValidationError, match="UTF-8"):
        restored.restore({**saved, "pending": "ff"})


@pytest.mark.parametrize("text", ["привет 中文 🌍 end", "<｜begin▁of▁sentence｜>a\n\x00b<｜end▁of▁sentence｜>",
                                  "Combining é العربية 👩‍💻 and C++ $() `x`"])
def test_incremental_utf8_never_inserts_partial_byte_replacements(tokenizer, text):
    tokens = tokenizer.encode(text)
    tracker = StopTracker(TokenBytes(tokenizer))
    output = ""
    for token in tokens:
        preview = tracker.preview([token])
        output += tracker.commit(preview)
    output += tracker.finish()
    assert output == tokenizer.decode(tokens) == text


def test_stop_preview_is_transactional_and_never_streams_partial_stop(tokenizer):
    tracker = StopTracker(TokenBytes(tokenizer), ("STOP",))
    tokens = tokenizer.encode("hello STOP tail more text")
    before = tracker.identity
    preview = tracker.preview(tokens)
    assert tracker.identity == before and preview.token_count == 2
    assert preview.matched == "STOP" and preview.text == "hello "
    assert tracker.commit(preview) == "hello "
    assert tracker.finish() == ""
    with pytest.raises(ValidationError):
        tracker.commit(preview)
    with pytest.raises(ValidationError):
        tracker.preview([tokens[-1]])


def test_stop_across_tokens_preserves_prefix_on_abort_and_flushes_nonmatches(tokenizer):
    tracker = StopTracker(TokenBytes(tokenizer), ("STOP", "STATION"))
    # Force byte-token boundaries using the actual vocabulary's singleton
    # ByteLevel IDs, independent of the BPE merge decisions for this text.
    token_bytes = tracker.token_bytes
    by_byte = {token_bytes(token): token for token in range(129280) if len(token_bytes(token)) == 1}
    def ids(text):
        return [by_byte[bytes((byte,))] for byte in text.encode()]
    visible = tracker.commit(tracker.preview(ids("a ST")))
    assert visible == "a " and tracker.snapshot()["tail"] == "ST"
    identity = tracker.identity
    abandoned = tracker.preview(ids("OP!"))
    assert abandoned.matched == "STOP" and tracker.identity == identity
    continued = tracker.preview(ids("UFF!"))
    assert continued.matched is None
    assert tracker.commit(continued) + tracker.finish() == "STUFF!"


def test_terminal_stop_inside_accepted_drafts_trims_target_prefix_before_commit(tokenizer):
    sampler = TerminalSampler(1017)
    tracker = StopTracker(TokenBytes(tokenizer), ("STOP",))
    counts = np.zeros(129280, np.int32)
    counts[0] = 1
    work = sampler.begin_verification(seed_token=0, policy=SamplingPolicy(temperature=0), counts=counts,
        generated_counts=counts, remaining=100, stop_identity=tracker.identity)
    proposals = tokenizer.encode("hello STOP tail more text")
    assert len(proposals) == 5
    # Complete probability/logit metadata only; no graph or model is mocked.
    work.proposal_ids = proposals
    work.proposal_rows = [np.eye(1, 129280, token, dtype=np.float64).reshape(-1) for token in proposals]
    work._proposal_ready = True
    logits = np.full((6, 129280), -80., np.float32)
    for row, token in enumerate([*proposals, 2]):
        logits[row, token] = 80.
    original = work.verify(logits)
    assert original.accepted_target_rows == 6 and original.pending_token == 2
    with pytest.raises(ValidationError):
        work.prepared()  # A frozen stop policy must run before preparation.
    before = tracker.identity
    decision = work.apply_stop(tracker)
    assert decision.accepted_drafts == 2 and decision.accepted_target_rows == 3
    assert decision.output_ids == tuple(proposals[:2]) and decision.pending_token is None and decision.stop_reason == "stop"
    assert tracker.identity == before
    prepared = work.prepared()
    tracker.commit(tracker.preview(decision.output_ids))
    work.commit(prepared["digest"])


def test_seed_stop_remains_an_unforwarded_pending_token(tokenizer):
    sampler = TerminalSampler(1017)
    tracker = StopTracker(TokenBytes(tokenizer), ("STOP",))
    token = tokenizer.encode(" STOP")[0]
    logits = np.full((1, 129280), -80., np.float32)
    logits[0, token] = 80.
    work = sampler.begin_seed(logits, policy=SamplingPolicy(temperature=0), counts=np.zeros(129280, np.int32),
                              remaining=8, stop_identity=tracker.identity)
    decision = work.apply_stop(tracker)
    assert decision.token == token and decision.pending_token == token
    assert decision.accepted_target_rows == 0 and decision.stop_reason == "stop"
    work.abort()
