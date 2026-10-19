"""Stage C's bounded probability/Markov sampling around the real CUDA graph.

Full probability vectors stay local. The coordinator receives compact token
decisions plus a digest, and authorizes RNG commit after coherent GPU commits.
"""
from dataclasses import asdict, dataclass
import hashlib

import numpy as np

from .jsonutil import ValidationError, fingerprint, integer
from .sampling import RandomStreams, SamplingPolicy, categorical, verify_drafts
from .protocol import hex_id


def _logits_cpu(logits, rows, vocabulary):
    """A bounded logit transfer is sampling metadata, never CPU model math."""
    if isinstance(logits, np.ndarray):
        result = logits
    else:
        import torch
        if not isinstance(logits, torch.Tensor) or not logits.is_cuda or logits.dtype != torch.float32:
            raise ValidationError("terminal logits must come from the actual FP32 CUDA vocabulary head")
        result = logits.detach().cpu().numpy()
    if result.dtype != np.float32 or result.shape != (rows, vocabulary) or not np.isfinite(result).all():
        raise ValidationError("terminal logits do not match the complete bounded vocabulary window")
    return result


@dataclass(frozen=True)
class SeedDecision:
    token: int
    stop_reason: str | None

    @property
    def output_ids(self):
        return (self.token,)

    @property
    def pending_token(self):
        return self.token

    @property
    def accepted_target_rows(self):
        return 0


class TerminalSampler:
    def __init__(self, seed, *, vocabulary=129280):
        self.random = RandomStreams(seed)
        self.vocabulary = integer(vocabulary, "sampling vocabulary", 1, 129280)
        self.active = None

    def _begin(self, policy, counts, remaining, eos_ids, *, generated_counts, logit_bias, allowed, grammar_identity):
        integer(remaining, "remaining output capacity", 1, 1048576)
        if self.active is not None or not isinstance(policy, SamplingPolicy):
            raise ValidationError("terminal sampling is active or its policy is invalid")
        counts = np.asarray(counts)
        if (counts.shape != (self.vocabulary,) or counts.dtype not in (np.int32, np.int64) or
                np.any(counts < 0) or counts.sum(dtype=np.int64) > 1048576):
            raise ValidationError("terminal counts differ from the exact published token history")
        for token in eos_ids:
            integer(token, "terminal EOS token", 0, self.vocabulary - 1)
        if allowed is not None and not callable(allowed):
            raise ValidationError("grammar mask provider must evaluate a bounded tentative token prefix")
        if (allowed is None) != (grammar_identity is None):
            raise ValidationError("a grammar mask requires its frozen grammar/prefix identity")
        if grammar_identity is not None:
            hex_id(grammar_identity, 64, "frozen grammar and committed-prefix identity")
        generated_counts = counts if generated_counts is None else np.asarray(generated_counts)
        if (generated_counts.shape != counts.shape or generated_counts.dtype not in (np.int32, np.int64) or
                np.any(generated_counts < 0) or np.any(generated_counts > counts)):
            raise ValidationError("generated sampling counts must be a subset of the complete history")
        # Never let a caller mutate sampling controls during a transaction.
        bias = None if logit_bias is None else dict(logit_bias)
        work = SamplingWork(self, self.random.begin(), policy, counts.copy(), generated_counts.copy(), remaining,
                            frozenset(eos_ids), bias, allowed, grammar_identity)
        self.active = work
        return work

    def begin_seed(self, logits, *, policy, counts, remaining, eos_ids=frozenset(),
                   generated_counts=None, logit_bias=None, allowed=None, grammar_identity=None):
        work = self._begin(policy, counts, remaining, eos_ids, generated_counts=generated_counts,
                           logit_bias=logit_bias, allowed=allowed, grammar_identity=grammar_identity)
        try:
            values = _logits_cpu(logits, 1, self.vocabulary)[0]
            distribution = work.distribution(values, ())
            token = categorical(distribution, lambda: work.transaction.random("target"))
            work.decision = SeedDecision(token, "eos" if token in work.eos_ids else "length" if remaining == 1 else None)
            return work
        except BaseException:
            work.abort()
            raise

    def begin_verification(self, *, seed_token, policy, counts, remaining, eos_ids=frozenset(),
                           generated_counts=None, logit_bias=None, allowed=None, grammar_identity=None):
        integer(seed_token, "pending verification seed", 0, self.vocabulary - 1)
        work = self._begin(policy, counts, remaining, eos_ids, generated_counts=generated_counts,
                           logit_bias=logit_bias, allowed=allowed, grammar_identity=grammar_identity)
        if work.counts[seed_token] < 1 or work.generated_counts[seed_token] < 1:
            work.abort()
            raise ValidationError("pending seed must already belong to published sampling history")
        work.seed_token = seed_token
        return work


class SamplingWork:
    def __init__(self, owner, transaction, policy, counts, generated_counts, remaining, eos_ids, logit_bias, allowed, grammar_identity):
        self.owner, self.transaction, self.policy = owner, transaction, policy
        self.counts, self.remaining, self.eos_ids = counts, remaining, eos_ids
        self.generated_counts = generated_counts
        self.logit_bias, self.allowed = logit_bias, allowed
        self.grammar_identity = grammar_identity
        self.seed_token = None
        self.proposal_ids, self.proposal_rows = [], []
        self.decision, self.closed = None, False
        self._proposal_ready = False

    def _open(self):
        if self.closed or self.owner.active is not self:
            raise ValidationError("terminal sampling transaction is closed or stale")

    def distribution(self, logits, prefix):
        self._open()
        counts = self.counts.copy()
        generated = self.generated_counts.copy()
        for token in prefix:
            counts[token] += 1
            generated[token] += 1
        mask = None if self.allowed is None else self.allowed(tuple(prefix))
        return self.policy.probabilities(logits, counts=counts, generated_counts=generated,
                                         logit_bias=self.logit_bias, allowed=mask)

    def propose(self, graph, result):
        """Sequential original Markov conditioning over the five-row graph."""
        self._open()
        if self.seed_token is None or self._proposal_ready or self.decision is not None:
            raise ValidationError("proposal generation is duplicate or lacks a committed pending seed")
        try:
            previous = self.seed_token
            for row in range(min(5, self.remaining)):
                # Real graph method applies the learned conditional Markov
                # bias. Confidence is diagnostic and cannot prune a proposal.
                logits, _confidence = graph.draft_conditional_logits(result, row, previous)
                values = _logits_cpu(logits, 1, self.owner.vocabulary)[0]
                distribution = self.distribution(values, self.proposal_ids)
                token = categorical(distribution, lambda: self.transaction.random("draft"))
                self.proposal_rows.append(distribution)
                self.proposal_ids.append(token)
                previous = token
                if token in self.eos_ids:
                    break
            self._proposal_ready = True
            return tuple(self.proposal_ids)
        except BaseException:
            self.abort()
            raise

    def use_normal_decode(self):
        """Explicit no-proposal path, including the end of context."""
        self._open()
        if self.seed_token is None or self._proposal_ready:
            raise ValidationError("normal decode is duplicate or lacks its pending seed")
        self._proposal_ready = True

    def verify(self, target_logits):
        self._open()
        if self.seed_token is None or not self._proposal_ready or self.decision is not None:
            raise ValidationError("target verification is duplicate or proposals are not complete")
        try:
            values = _logits_cpu(target_logits, len(self.proposal_ids) + 1, self.owner.vocabulary)
            targets = _ConditionalTargets(self, values)
            self.decision = verify_drafts(targets, self.proposal_rows, self.proposal_ids, self.transaction,
                                          greedy=self.policy.temperature == 0, output_limit=self.remaining,
                                          eos_ids=self.eos_ids)
            return self.decision
        except BaseException:
            self.abort()
            raise

    def prepared(self):
        self._open()
        if self.decision is None:
            raise ValidationError("terminal decision has not been computed")
        data = {"decision": asdict(self.decision), "accepted_target_rows": self.decision.accepted_target_rows,
                "policy": self.policy.as_dict(), "RNG": self.transaction.prepared_state(),
                "seed": self.seed_token, "proposals": self.proposal_ids,
                "counts_sha256": hashlib.sha256(self.counts.astype("<i8").tobytes()).hexdigest(),
                "generated_counts_sha256": hashlib.sha256(self.generated_counts.astype("<i8").tobytes()).hexdigest(),
                "logit_bias": self.logit_bias, "grammar_identity": self.grammar_identity,
                "remaining": self.remaining, "eos_ids": sorted(self.eos_ids)}
        return {"decision": asdict(self.decision), "accepted_target_rows": self.decision.accepted_target_rows,
                "digest": fingerprint(data)}

    def commit(self, expected_digest):
        self._open()
        if self.prepared()["digest"] != expected_digest:
            raise ValidationError("all-stage commit does not match the prepared terminal decision/RNG")
        self.transaction.commit()
        self.closed, self.owner.active = True, None
        # The caller has acknowledged/recorded all corresponding GPU finishes.
        # Counts/history update from this compact decision after that barrier.
        return self.decision

    def abort(self):
        self._open()
        self.transaction.abort()
        self.closed, self.owner.active = True, None


class _ConditionalTargets:
    def __init__(self, work, values):
        self.work, self.values, self.cache = work, values, {}

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if index not in self.cache:
            self.cache[index] = self.work.distribution(self.values[index], self.work.proposal_ids[:index])
        return self.cache[index]
