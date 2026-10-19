from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.sampling import (
    RandomStreams, SamplingPolicy, categorical, correction_distribution, verify_drafts,
)


class ControlledUniforms:
    """Deterministic mathematical inputs; no model or native backend is simulated."""
    def __init__(self, acceptance):
        self.acceptance = iter(acceptance)

    def random(self, domain):
        return next(self.acceptance) if domain == "accept" else .5


def test_temperature_one_filters_and_extremely_small_positive_temperature():
    logits = np.log(np.array([.5, .3, .2], np.float64))
    np.testing.assert_allclose(SamplingPolicy().probabilities(logits), [.5, .3, .2], rtol=1e-15)
    np.testing.assert_array_equal(SamplingPolicy(temperature=0).probabilities(logits), [1, 0, 0])
    np.testing.assert_array_equal(SamplingPolicy(temperature=1e-320).probabilities(logits), [1, 0, 0])
    np.testing.assert_allclose(SamplingPolicy(top_k=2).probabilities(logits), [.625, .375, 0], rtol=1e-15)
    np.testing.assert_allclose(SamplingPolicy(top_p=.7).probabilities(logits), [.625, .375, 0], rtol=1e-15)
    np.testing.assert_array_equal(SamplingPolicy(top_p=0).probabilities(logits), [1, 0, 0])
    np.testing.assert_array_equal(SamplingPolicy(min_p=.7).probabilities(logits), [1, 0, 0])
    filtered = SamplingPolicy(min_p=.5, top_p=.6).probabilities(logits)
    np.testing.assert_array_equal(filtered, [1, 0, 0])


def test_explicit_penalties_bias_and_mask_share_one_probability_definition():
    logits = np.array([2., -1., 3., 0.], np.float32)
    counts = np.array([2, 1, 0, 0], np.int32)
    policy = SamplingPolicy(repetition_penalty=2, presence_penalty=.25, frequency_penalty=.5)
    actual = policy.probabilities(logits, counts=counts, logit_bias={3: .2}, allowed=np.array([True, True, False, True]))
    adjusted = np.array([1 - .25 - 1, -2 - .25 - .5, -np.inf, .2])
    expected = np.exp(adjusted - adjusted.max())
    np.testing.assert_allclose(actual, expected / expected.sum(), rtol=1e-15)
    with pytest.raises(ValidationError, match="every"):
        policy.probabilities(logits, allowed=np.zeros(4, bool))
    with pytest.raises(ValidationError):
        SamplingPolicy(temperature=True)


def test_categorical_never_selects_a_zero_probability_at_cdf_boundaries():
    probabilities = np.array([0., .25, 0., .75, 0.], np.float64)
    assert categorical(probabilities, lambda: 0.) == 1
    assert categorical(probabilities, lambda: .25) == 3
    assert categorical(probabilities, lambda: np.nextafter(1., 0.).item()) == 3


def test_exact_discrete_rejection_law_recovers_target_marginal():
    target = np.array([.3, .7], np.float64)
    proposal = np.array([.8, .2], np.float64)
    np.testing.assert_array_equal(correction_distribution(target, proposal), [0, 1])
    output_mass = [Fraction(0), Fraction(0)]
    proposal_mass = (Fraction(4, 5), Fraction(1, 5))
    # The only nontrivial acceptance ratio is 3/8. Eight interval midpoints
    # integrate it exactly, independently of any pseudo-random test outcome.
    for token in (0, 1):
        for uniform in (np.arange(8) + .5) / 8:
            decision = verify_drafts((target, target), (proposal,), (token,),
                                     ControlledUniforms([float(uniform)]), greedy=False, output_limit=2)
            output_mass[decision.output_ids[0]] += proposal_mass[token] / 8
    assert output_mass == [Fraction(3, 10), Fraction(7, 10)]


@pytest.mark.parametrize("rejected", range(6))
def test_every_rejection_length_and_full_acceptance_preserves_processed_pending_split(rejected):
    proposal = np.array([.5, .5], np.float64)
    targets = [np.array([.75, .25], np.float64) for _ in range(6)]
    if rejected < 5:
        targets[rejected] = np.array([.25, .75], np.float64)
    decision = verify_drafts(targets, [proposal] * 5, [0] * 5, ControlledUniforms([.75] * 5),
                             greedy=False, output_limit=10)
    assert decision.accepted_drafts == rejected
    assert decision.accepted_target_rows == 1 + rejected
    assert len(decision.output_ids) == rejected + 1
    assert decision.pending_token == decision.output_ids[-1]
    assert decision.stop_reason is None


def test_eos_or_output_exhaustion_inside_accepted_drafts_never_draws_a_bonus():
    target = np.array([.9, .1], np.float64)
    proposal = np.array([.5, .5], np.float64)
    eos = verify_drafts([target] * 6, [proposal] * 5, [0] * 5, ControlledUniforms([.1]),
                        greedy=False, output_limit=10, eos_ids={0})
    assert (eos.output_ids, eos.pending_token, eos.accepted_target_rows, eos.stop_reason) == ((0,), None, 2, "eos")
    limit = verify_drafts([target] * 6, [proposal] * 5, [0] * 5, ControlledUniforms([.1, .1]),
                          greedy=False, output_limit=2)
    assert (limit.output_ids, limit.pending_token, limit.accepted_target_rows, limit.stop_reason) == ((0, 0), None, 3, "length")


def test_greedy_verification_consumes_no_randomness():
    streams = RandomStreams(1017)
    before = streams.snapshot()
    transaction = streams.begin()
    target = np.array([.2, .8], np.float64)
    proposal = np.array([.5, .5], np.float64)
    decision = verify_drafts([target] * 6, [proposal] * 5, [0] * 5, transaction, greedy=True, output_limit=5)
    assert decision.output_ids == (1,)
    transaction.commit()
    assert streams.snapshot() == before


def test_normal_rejection_retains_all_generated_proposal_draws_but_full_abort_restores():
    streams = RandomStreams(42)
    before = streams.snapshot()
    transaction = streams.begin()
    for _ in range(5):
        transaction.random("draft")
    proposal_state = transaction.prepared_state()["domains"]["draft"]
    target, proposal = np.array([0., 1.]), np.array([.5, .5])
    decision = verify_drafts([target] * 6, [proposal] * 5, [0] * 5, transaction, greedy=False, output_limit=10)
    assert decision.accepted_drafts == 0
    transaction.commit()
    assert streams.snapshot()["domains"]["draft"] == proposal_state
    assert streams.snapshot() != before
    committed = streams.snapshot()
    transaction = streams.begin()
    for domain in ("target", "draft", "accept"):
        transaction.random(domain)
    with pytest.raises(ValidationError, match="idle"):
        streams.snapshot()
    transaction.abort()
    assert streams.snapshot() == committed


def test_rng_snapshot_restore_is_atomic_and_resumes_the_same_domain_sequences():
    first, restored = RandomStreams(9), RandomStreams(0)
    saved = first.snapshot()
    invalid = deepcopy(saved)
    invalid["domains"]["accept"]["increment"] = "0" * 32
    before = restored.snapshot()
    with pytest.raises(ValidationError, match="odd"):
        restored.restore(invalid)
    assert restored.snapshot() == before
    restored.restore(saved)
    left, right = first.begin(), restored.begin()
    for domain in ("draft", "accept", "target"):
        assert [left.random(domain) for _ in range(5)] == [right.random(domain) for _ in range(5)]
    assert left.commit() == right.commit()


def test_terminal_seed_and_normal_decode_commit_only_the_matching_decision():
    from deepseekrpc.terminal_sampling import TerminalSampler
    sampler = TerminalSampler(1017, vocabulary=3)
    counts = np.array([1, 0, 0], np.int32)
    policy = SamplingPolicy(temperature=0)
    seed = sampler.begin_seed(np.array([[0., 4., 0.]], np.float32), policy=policy,
                              counts=counts, remaining=10)
    assert seed.decision.output_ids == (1,)
    assert seed.decision.accepted_target_rows == 0
    with pytest.raises(ValidationError, match="match"):
        seed.commit("0" * 64)
    seed.commit(seed.prepared()["digest"])
    counts[1] += 1  # Published seed belongs to counts before being forwarded.
    work = sampler.begin_verification(seed_token=1, policy=policy, counts=counts, remaining=1)
    work.use_normal_decode()
    decision = work.verify(np.array([[0., 0., 4.]], np.float32))
    assert (decision.accepted_target_rows, decision.output_ids, decision.pending_token, decision.stop_reason) == (
        1, (2,), 2, "length")
    work.commit(work.prepared()["digest"])
    assert sampler.active is None


def test_terminal_matching_prefix_controls_include_the_seed_and_conditional_markov_history():
    from deepseekrpc.terminal_sampling import TerminalSampler
    class ControlledLogits:
        """Sampling fixture only; this does not emulate a model graph."""
        def __init__(self):
            self.previous = []

        def draft_conditional_logits(self, result, row, previous):
            self.previous.append(previous)
            return np.array([[0., 0., 0.]], np.float32), None

    sampler = TerminalSampler(2, vocabulary=3)
    counts = np.array([5, 1, 0], np.int32)  # Pending seed 1 is already published.
    seen = []
    def grammar(prefix):
        seen.append(prefix)
        return np.array([False, len(prefix) % 2 == 1, len(prefix) % 2 == 0])
    work = sampler.begin_verification(seed_token=1, policy=SamplingPolicy(temperature=0),
                                      counts=counts, remaining=3, allowed=grammar, grammar_identity="e" * 64)
    fixture = ControlledLogits()
    assert work.propose(fixture, None) == (2, 1, 2)
    assert fixture.previous == [1, 2, 1]
    assert seen == [(), (2,), (2, 1)]
    target = np.zeros((4, 3), np.float32)
    decision = work.verify(target)
    assert seen[3:] == [(), (2,), (2, 1)]  # No distribution after the length stop.
    assert (decision.output_ids, decision.pending_token, decision.accepted_target_rows) == ((2, 1, 2), None, 4)
    work.commit(work.prepared()["digest"])
    assert counts.tolist() == [5, 1, 0]  # No tentative proposal mutated published counts.


def test_terminal_invalid_target_restores_proposal_rng_and_releases_the_transaction():
    from deepseekrpc.terminal_sampling import TerminalSampler
    class ControlledLogits:
        def draft_conditional_logits(self, result, row, previous):
            return np.array([[0., 0.]], np.float32), None
    sampler = TerminalSampler(7, vocabulary=2)
    before = sampler.random.snapshot()
    work = sampler.begin_verification(seed_token=0, policy=SamplingPolicy(),
                                      counts=np.array([1, 0], np.int32), remaining=10)
    work.propose(ControlledLogits(), None)
    with pytest.raises(ValidationError, match="vocabulary"):
        work.verify(np.zeros((5, 2), np.float32))  # Need seed + five rows.
    assert sampler.active is None
    assert sampler.random.snapshot() == before


def test_terminal_accepted_eos_never_queries_a_distribution_after_finished_grammar():
    from deepseekrpc.terminal_sampling import TerminalSampler
    class ControlledLogits:
        def draft_conditional_logits(self, result, row, previous):
            return np.array([[0., 1.]], np.float32), None
    def grammar(prefix):
        assert not prefix, "EOS has ended this grammar; no later distribution exists"
        return np.array([False, True])
    sampler = TerminalSampler(7, vocabulary=2)
    work = sampler.begin_verification(seed_token=0, policy=SamplingPolicy(),
                                      counts=np.array([1, 0], np.int32), remaining=10,
                                      eos_ids={1}, allowed=grammar, grammar_identity="f" * 64)
    assert work.propose(ControlledLogits(), None) == (1,)
    decision = work.verify(np.array([[0., 1.], [0., 1.]], np.float32))
    assert (decision.output_ids, decision.stop_reason, decision.pending_token) == ((1,), "eos", None)
    work.commit(work.prepared()["digest"])


def test_presence_and_frequency_penalties_exclude_prompt_counts_but_repetition_includes_them():
    counts = np.array([5, 1, 0], np.int32)
    generated = np.array([0, 1, 0], np.int32)
    values = np.array([2., 2., 0.], np.float32)
    policy = SamplingPolicy(repetition_penalty=2, presence_penalty=.5, frequency_penalty=.25)
    actual = policy.probabilities(values, counts=counts, generated_counts=generated)
    expected = np.exp(np.array([1., .25, 0.]) - 1)
    np.testing.assert_allclose(actual, expected / expected.sum(), rtol=1e-15, atol=0)
