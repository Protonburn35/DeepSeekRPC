"""Control-state checks only; actual three-stage model execution is separate."""
from dataclasses import replace

import pytest

from deepseekrpc.attention_state import AttentionWindow
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.phases import PhaseLedger, unanimous_receipts
from deepseekrpc.planner import Role
from deepseekrpc.state import Operation
from deepseekrpc.execution import StepInput, source_closures


def ledgers(tokens=129, encoder_start=0):
    result = [PhaseLedger(role) for role in Role]
    for ledger in result:
        ledger.reset(session="a" * 32, request="b" * 32, epoch=0,
                     history_digest="c" * 64, encoder_tokens=tokens, encoder_start=encoder_start)
    return result


def window(number, start, rows, phase, extent=129, history="c" * 64):
    op = Operation("a" * 32, "b" * 32, 0, number, f"{number:032x}", f"{number:064x}", start, rows)
    return AttentionWindow(op, history, phase, extent)


def finish_all(cluster, step, accepted, pending=None):
    digests = []
    for ledger in cluster:
        assert ledger.admit(step) is None
        digests.append(ledger.prepare(step, accepted=accepted, result_digest="d" * 64,
                                      pending_token=pending))
    assert len(set(digests)) == 1
    receipts = [ledger.finish(step, commit_digest=digests[0]) for ledger in cluster]
    for ledger in cluster:
        ledger.seal(receipts, commit_digest=digests[0])
    return receipts, digests[0]


def prefill(cluster):
    number = 1
    for start, rows in ((0, 64), (64, 64), (128, 1)):
        receipts, digest = finish_all(cluster, window(number, start, rows, "encoder"), rows)
        number += 1
    for ledger in cluster:
        ledger.encoder_barrier(receipts, commit_digest=digest)
        # Actual worker must rewind CUDA SWA/pair state successfully here.
        ledger.activate_target()
    for start, rows in ((0, 64), (64, 64), (128, 1)):
        finish_all(cluster, window(number, start, rows, "target"), rows)
        number += 1
    return number


def test_ced_requires_complete_identical_chunks_and_unanimous_full_encoder_barrier():
    cluster = ledgers()
    with pytest.raises(ValidationError, match="whole prompt"):
        cluster[0].admit(window(1, 0, 63, "encoder"))
    with pytest.raises(ValidationError, match="whole prompt"):
        cluster[0].admit(window(1, 1, 64, "encoder"))
    with pytest.raises(ValidationError, match="barrier"):
        cluster[0].activate_target()
    next_number = prefill(cluster)
    assert next_number == 7
    for ledger in cluster:
        assert (ledger.phase, ledger.encoder_cursor, ledger.target_cursor, ledger.processed) == (
            "decode", 129, 129, 129)
        ledger.publish_seed(token=17, history_digest="e" * 64)
        assert ledger.pending_token == 17
        assert ledger.processed == 129  # Seed emission is not a forward pass.


def test_encoder_prefix_reuse_still_requires_full_target_replay_and_all_stage_barrier():
    cluster = ledgers(encoder_start=64)
    for number, start, rows in ((1, 64, 64), (2, 128, 1)):
        receipts, digest = finish_all(cluster, window(number, start, rows, "encoder"), rows)
    for ledger in cluster:
        ledger.encoder_barrier(receipts, commit_digest=digest)
        ledger.activate_target()
        with pytest.raises(ValidationError, match="whole prompt"):
            ledger.admit(window(3, 64, 64, "target"))
    for number, start, rows in ((3, 0, 64), (4, 64, 64), (5, 128, 1)):
        finish_all(cluster, window(number, start, rows, "target"), rows)
    assert all((ledger.phase, ledger.encoder_cursor, ledger.target_cursor, ledger.processed) ==
               ("decode", 129, 129, 129) for ledger in cluster)


def test_restored_timeline_retains_pending_input_and_operation_cursor_under_new_epoch():
    ledger = ledgers()[0]
    value = {"session": "c" * 32, "request": "d" * 32, "epoch": 7,
             "history": "e" * 64, "encoder_tokens": 129, "processed": 153,
             "pending_token": 11, "stop_reason": None, "next_operation": 42}
    ledger.restore_committed(value, session="f" * 32, request="1" * 32, epoch=8, history_digest="e" * 64)
    assert ledger.processed == 153 and ledger.pending_token == 11 and ledger._next == 42
    assert ledger.history_digest == "e" * 64 and ledger.encoder_cursor == ledger.target_cursor == 129
    step = AttentionWindow(Operation("f" * 32, "1" * 32, 8, 42, "2" * 32, "3" * 64, 153, 6),
                           "e" * 64, "decode", 159)
    assert ledger.admit(step) is None
    with pytest.raises(ValidationError):
        ledger.admit(replace(step, operation=replace(step.operation, epoch=7)))


def test_local_finish_alone_cannot_allow_more_work_and_duplicate_commit_is_idempotent():
    cluster = ledgers()
    step = window(1, 0, 64, "encoder")
    for ledger in cluster:
        ledger.admit(step)
    digest = cluster[0].prepare(step, accepted=64, result_digest="d" * 64)
    first = cluster[0].finish(step, commit_digest=digest)
    assert cluster[0].finish(step, commit_digest=digest) == first
    assert cluster[0].admit(step) == first
    with pytest.raises(ValidationError, match="active, stale"):
        cluster[0].admit(window(2, 64, 64, "encoder"))
    with pytest.raises(ValidationError, match="every actual stage"):
        cluster[0].seal([first], commit_digest=digest)
    receipts = [first]
    for ledger in cluster[1:]:
        assert ledger.prepare(step, accepted=64, result_digest="d" * 64) == digest
        receipts.append(ledger.finish(step, commit_digest=digest))
    divergent = [*receipts[:2], replace(receipts[2], processed=1)]
    with pytest.raises(ValidationError, match="disagree"):
        unanimous_receipts(divergent, expected_digest=digest)
    for ledger in cluster:
        ledger.seal(receipts, commit_digest=digest)
        ledger.seal(receipts, commit_digest=digest)  # Lost acknowledgment retry.
    changed = replace(step, operation=replace(step.operation, digest="f" * 64))
    with pytest.raises(ValidationError, match="semantic inputs"):
        cluster[0].admit(changed)
    with pytest.raises(ValidationError, match="accepted prefix"):
        cluster[0].prepare(step, accepted=0, result_digest="d" * 64)


@pytest.mark.parametrize("accepted_drafts", range(6))
def test_every_verification_prefix_advances_seed_plus_accepted_inputs_only(accepted_drafts):
    cluster = ledgers()
    next_number = prefill(cluster)
    for ledger in cluster:
        ledger.publish_seed(token=17, history_digest="e" * 64)
    draft = window(next_number, 129, 5, "draft", extent=134, history="e" * 64)
    finish_all(cluster, draft, 0)
    for ledger in cluster:
        assert (ledger.processed, ledger.pending_token) == (129, 17)
    verify = window(next_number + 1, 129, 6, "decode", extent=135, history="e" * 64)
    finish_all(cluster, verify, 1 + accepted_drafts, pending=29)
    for ledger in cluster:
        assert ledger.processed == 130 + accepted_drafts
        assert ledger.pending_token == 29  # Correction/bonus has not run.
        ledger.publish_history(history_digest="f" * 64)


def test_aborted_window_retains_cursor_seed_and_full_prefill_chunk_partition():
    cluster = ledgers()
    finish_all(cluster, window(1, 0, 64, "encoder"), 0)
    finish_all(cluster, window(2, 0, 64, "encoder"), 64)
    assert all(ledger.encoder_cursor == 64 for ledger in cluster)
    step = window(3, 64, 64, "encoder")
    cluster[0].admit(step)
    with pytest.raises(ValidationError, match="whole fixed chunks"):
        cluster[0].prepare(step, accepted=32, result_digest="d" * 64)
    cluster[0].fault()
    with pytest.raises(ValidationError, match="faulted"):
        cluster[0].admit(step)
    with pytest.raises(ValidationError, match="newer epoch"):
        cluster[0].reset(session="a" * 32, request="b" * 32, epoch=0,
                         history_digest="c" * 64, encoder_tokens=129)
    cluster[0].reset(session="a" * 32, request="b" * 32, epoch=1,
                     history_digest="c" * 64, encoder_tokens=129)
    assert cluster[0].phase == "encoder"


def test_execution_identity_binds_original_ids_masks_and_absolute_compression_closures():
    inputs = StepInput((7, 129264, 9), (True, False, True))
    step = inputs.make_window(session="a" * 32, request="b" * 32, epoch=0, number=1,
                              transaction="c" * 32, start=7, phase="decode",
                              history_digest="d" * 64, encoder_tokens=10)
    assert inputs.digest(step) == step.operation.digest
    assert StepInput((7, 129264, 8), inputs.text_mask).digest(step) != step.operation.digest
    assert StepInput(inputs.tokens, (True, True, True)).digest(step) != step.operation.digest
    assert source_closures(step, 2) == ((3, 4), (7, 9))
    assert source_closures(step, 1) == ((7, 8, 9), (7, 8, 9))
    assert source_closures(replace(step, phase="target"), 2) == ((), ())
