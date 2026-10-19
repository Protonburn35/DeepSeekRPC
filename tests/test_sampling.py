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
