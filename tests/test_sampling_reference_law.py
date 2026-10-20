"""Independent finite probability laws prompted by the DeepInfer source audit.

These are tiny sampling fixtures, never a substitute for a model/GPU run.
See docs/DEEPINFER_REVIEW.md and the pinned source hashes in provenance/.
"""
from fractions import Fraction
from itertools import product

import numpy as np

from deepseekrpc.sampling import SamplingPolicy, categorical, correction_distribution, verify_drafts
from deepseekrpc.terminal_sampling import TerminalSampler


def test_conditional_two_token_verification_has_the_exact_target_joint_law():
    # Enumerate every proposal and exact equal-width RNG cell. No statistical
    # tolerance or fortunate pseudorandom outcome can hide an incorrect law.
    q0 = (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4))
    p0 = (Fraction(1, 4), Fraction(3, 8), Fraction(3, 8))
    q1 = [tuple(q0[(token - previous - 1) % 3] for token in range(3)) for previous in range(3)]
    p1 = [tuple(p0[(token - previous - 2) % 3] for token in range(3)) for previous in range(3)]
    observed = [[Fraction(0) for _ in range(3)] for _ in range(3)]

    class Draws:
        def __init__(self, accept, target):
            self.accept, self.target = iter(accept), target

        def random(self, domain):
            assert domain in {"accept", "target"}
            return next(self.accept) if domain == "accept" else self.target

    for first, second, a, b, c, d in product(range(3), range(3), range(4), range(4), range(8), range(8)):
        arrays = lambda rows: [np.array(row, np.float64) for row in rows]
        result = verify_drafts(arrays((p0, p1[first], p0)), arrays((q0, q1[first])), (first, second),
            Draws(((a + .5) / 4, (b + .5) / 4), (c + .5) / 8), greedy=False, output_limit=2)
        output = result.output_ids
        if len(output) == 1:
            # A first-position rejection leaves a new pending seed. Complete
            # the second token with its independently specified target law.
            output += (categorical(np.array(p1[output[0]], np.float64), lambda: (d + .5) / 8),)
        observed[output[0]][output[1]] += q0[first] * q1[first][second] / (16 * 64)
    assert observed == [[p0[first] * p1[first][second] for second in range(3)] for first in range(3)]


def test_equal_probability_coordinates_receive_no_rejection_correction_mass():
    # DeepInfer's 5e-5 draft penalty adds mass to coordinate 2. Exact rejection
    # must leave an equal target/proposal coordinate at precisely zero.
    target = np.array([.125, .625, .25], np.float64)
    proposal = np.array([.5, .25, .25], np.float64)
    np.testing.assert_array_equal(correction_distribution(target, proposal), [0., 1., 0.])


def test_future_confidence_values_cannot_change_actual_sampling_or_rng_commit():
    class LogitFixture:
        def __init__(self, confidence):
            self.confidence = confidence

        def draft_conditional_logits(self, result, row, previous):
            # Input logits depend on the previous sampled token, as Markov
            # conditioning does. Confidence must not select/prune a suffix.
            return np.array([[previous * .5, .25, -.5]], np.float32), self.confidence[row]

    receipts = []
    for confidence in ([0.] * 5, [1.] * 5, [0., 1., .5, .01, .99]):
        sampler = TerminalSampler(41017, vocabulary=3)
        work = sampler.begin_verification(seed_token=0, policy=SamplingPolicy(),
            counts=np.array([1, 0, 0], np.int32), remaining=8)
        proposals = work.propose(LogitFixture(confidence), None)
        assert len(proposals) == 5
        decision = work.verify(np.array([[.1, .5, -.5]] * 6, np.float32))
        work.commit(work.prepared()["digest"])
        receipts.append((proposals, decision, sampler.random.snapshot()))
    assert receipts[0] == receipts[1] == receipts[2]
