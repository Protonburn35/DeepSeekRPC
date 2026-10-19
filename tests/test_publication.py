"""History/RNG transactions only. No model or GPU execution is emulated."""
from dataclasses import replace

import numpy as np
import pytest

from deepseekrpc.execution import StepInput
from deepseekrpc.history import TokenHistory
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.phases import PhaseLedger
from deepseekrpc.planner import Role
from deepseekrpc.publication import OutputPlan, PublicationLedger
from deepseekrpc.sampling import SamplingPolicy
from deepseekrpc.terminal_sampling import TerminalSampler


def commit_window(phases, number, start, rows, phase, history, accepted, pending=None):
    inputs = StepInput(tuple(range(rows)), (True,) * rows)
    window = inputs.make_window(session="a" * 32, request="b" * 32, epoch=0, number=number,
                                transaction=f"{number:032x}", start=start, phase=phase,
                                history_digest=history, encoder_tokens=3 if phase != "decode" else start + rows)
    for ledger in phases:
        ledger.admit(window)
    digests = [ledger.prepare(window, accepted=accepted, result_digest="d" * 64, pending_token=pending) for ledger in phases]
    assert len(set(digests)) == 1
    receipts = [ledger.finish(window, commit_digest=digests[0]) for ledger in phases]
    for ledger in phases:
        ledger.seal(receipts, commit_digest=digests[0])
    return receipts, digests[0]


def setup_cluster():
    histories = [TokenHistory("e" * 64, capacity=32) for _ in Role]
    phases = [PhaseLedger(role) for role in Role]
    for phase, history in zip(phases, histories):
        digest = history.set_prompt([0, 3, 4])
        phase.reset(session="a" * 32, request="b" * 32, epoch=0, history_digest=digest, encoder_tokens=3)
    receipts, digest = commit_window(phases, 1, 0, 3, "encoder", histories[0].digest, 3)
    for phase in phases:
        phase.encoder_barrier(receipts, commit_digest=digest)
        phase.activate_target()
    commit_window(phases, 2, 0, 3, "target", histories[0].digest, 3)
    return phases, histories


def output_plan(phase, history, ids, pending, *, initial=False, reason=None, sampling="f" * 64, number=1):
    return OutputPlan(f"{number:032x}", phase.session, phase.request, phase.epoch, phase.processed,
                      history.digest, history.preview_append(ids), sampling, tuple(ids), pending, initial, reason)


def commit_publication(owners, plan):
    prepared = [owner.prepare(plan) for owner in owners]
    committed = [owner.commit(plan, prepared) for owner in owners]
    finalized = [owner.finalize(plan, committed) for owner in owners]
    return prepared, committed, finalized


def test_real_sampling_rng_finalizes_only_after_all_three_published_history_commits():
    phases, histories = setup_cluster()
    sampler = TerminalSampler(1017)
    before = sampler.random.snapshot()
    logits = np.full((1, 129280), -80., np.float32)
    logits[0, 7:9] = [1., 0.]
    work = sampler.begin_seed(logits, policy=SamplingPolicy(), counts=histories[2].counts(),
                              generated_counts=histories[2].generated_counts(), remaining=10)
    decision = work.decision
    plan = output_plan(phases[2], histories[2], decision.output_ids, decision.pending_token,
                        initial=True, sampling=work.prepared()["digest"])
    finalized = []
    def validate(value):
        return (value.sampling_digest == work.prepared()["digest"] and
                value.output_ids == decision.output_ids and value.initial_seed)
    def finalize(value):
        work.commit(value.sampling_digest)
        finalized.append(value.digest)
    owners = [PublicationLedger(phases[0], histories[0]), PublicationLedger(phases[1], histories[1]),
              PublicationLedger(phases[2], histories[2], validate_sampling=validate, finalize_sampling=finalize)]
    prepared = [owner.prepare(plan) for owner in owners]
    assert not any(phase.quiescent for phase in phases)
    assert sampler.random.active and not finalized
    next_step = StepInput((decision.token,), (True,)).make_window(
        session="a" * 32, request="b" * 32, epoch=0, number=3, transaction="c" * 32,
        start=3, phase="decode", history_digest=histories[0].digest, encoder_tokens=4)
    with pytest.raises(ValidationError, match="active, stale"):
        phases[0].admit(next_step)
    committed = [owner.commit(plan, prepared) for owner in owners]
    assert all(history.length == 4 for history in histories)
    assert sampler.random.active and not finalized
    with pytest.raises(ValidationError, match="every stage"):
        owners[2].finalize(plan, committed[:2])
    with pytest.raises(ValidationError, match="exact publication plan"):
        owners[2].finalize(plan, prepared)
    for owner in owners:
        owner.finalize(plan, committed)
    assert finalized == [plan.digest]
    assert all(phase.quiescent and phase.processed == 3 and phase.pending_token == decision.token for phase in phases)
    after = sampler.random.snapshot()
    assert after != before
    for owner in owners:
        owner.prepare(plan)
        owner.commit(plan, prepared)
        owner.finalize(plan, committed)
    assert sampler.random.snapshot() == after and finalized == [plan.digest]
    assert all(history.generated_counts()[decision.token] == 1 for history in histories)


def test_rejection_and_accepted_eos_publish_only_the_verified_prefix_and_actual_pending_token():
    phases, histories = setup_cluster()
    owners = [PublicationLedger(phases[0], histories[0]), PublicationLedger(phases[1], histories[1]),
              PublicationLedger(phases[2], histories[2], validate_sampling=lambda _: True, finalize_sampling=lambda _: None)]
    seed = output_plan(phases[0], histories[0], [7], 7, initial=True)
    commit_publication(owners, seed)
    # Metadata represents seed 7 + two proposals, rejecting the second.
    commit_window(phases, 3, 3, 3, "decode", histories[0].digest, 2, pending=11)
    correction = output_plan(phases[0], histories[0], [8, 11], 11, number=2)
    commit_publication(owners, correction)
    for phase, history in zip(phases, histories):
        assert (phase.processed, phase.pending_token, history.length) == (5, 11, 6)
        assert history.rows(3, 3)[0] == (7, 8, 11)
    # Stop at an accepted EOS: seed 11 + EOS were actually processed, no bonus.
    commit_window(phases, 4, 5, 3, "decode", histories[0].digest, 2, pending=None)
    eos = output_plan(phases[0], histories[0], [1], None, number=3, reason="eos")
    commit_publication(owners, eos)
    for phase, history in zip(phases, histories):
        assert (phase.processed, phase.pending_token, history.length) == (7, None, 7)
        assert history.generated_counts()[1] == 1


def test_publication_rejects_wrong_history_pending_prefix_and_reset_epoch():
    phases, histories = setup_cluster()
    owner = PublicationLedger(phases[0], histories[0])
    plan = output_plan(phases[0], histories[0], [7], 7, initial=True)
    with pytest.raises(ValidationError, match="processed-versus-pending"):
        owner.prepare(replace(plan, history_digest="0" * 64))
    owner.prepare(plan)
    phases[0].fault()
    phases[0].reset(session="a" * 32, request="b" * 32, epoch=1,
                    history_digest=histories[0].digest, encoder_tokens=3)
    with pytest.raises(ValidationError, match="invalidated"):
        owner.prepare(plan)
