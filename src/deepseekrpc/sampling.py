"""Terminal sampling and exact speculative rejection/correction arithmetic.

Only logits/probabilities and token metadata are processed here. Model execution
remains CUDA. Three independent PCG64 domains have explicit transactional state.
Ordinary rejection retains randomness consumed by the proposal/decision; a full
uncommitted abort restores every domain. Reusing rejection-conditioned proposal
draws would invalidate the proposal distribution assumed by the verifier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np

from .jsonutil import ValidationError, integer

DOMAINS = ("target", "draft", "accept")


def _real(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValidationError(f"invalid {name}")
    return float(value)


@dataclass(frozen=True)
class SamplingPolicy:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def __post_init__(self):
        for name, low, high in (("temperature", 0, 100), ("top_p", 0, 1), ("min_p", 0, 1),
                                ("repetition_penalty", 0.01, 100), ("presence_penalty", -2, 2),
                                ("frequency_penalty", -2, 2)):
            object.__setattr__(self, name, _real(getattr(self, name), name, low, high))
        integer(self.top_k, "top-k", 0, 129280)

    def as_dict(self):
        return asdict(self)

    def probabilities(self, logits, *, counts=None, logit_bias=None, allowed=None):
        values = np.asarray(logits)
        if (values.ndim != 1 or not 1 <= len(values) <= 129280 or values.dtype not in (np.float32, np.float64) or
                np.any(np.isnan(values)) or np.any(np.isposinf(values))):
            raise ValidationError("sampling needs one bounded finite-or-masked floating logit vector")
        values = values.astype(np.float64, copy=True)
        if counts is not None:
            counts = np.asarray(counts)
            if (counts.shape != values.shape or counts.dtype not in (np.int32, np.int64) or
                    np.any(counts < 0) or np.any(counts > 1048576) or counts.sum(dtype=np.int64) > 1048576):
                raise ValidationError("sampling frequency counts do not match the vocabulary")
            present = counts > 0
            values[present] = np.where(values[present] >= 0, values[present] / self.repetition_penalty,
                                       values[present] * self.repetition_penalty)
            values -= self.presence_penalty * present + self.frequency_penalty * counts
        if logit_bias is not None:
            if not isinstance(logit_bias, dict) or len(logit_bias) > len(values):
                raise ValidationError("invalid logit bias mapping")
            for token, bias in logit_bias.items():
                integer(token, "biased token", 0, len(values) - 1)
                values[token] += _real(bias, "logit bias", -100, 100)
        if allowed is not None:
            allowed = np.asarray(allowed)
            if allowed.shape != values.shape or allowed.dtype != np.bool_:
                raise ValidationError("sampling grammar/token mask does not match the vocabulary")
            values[~allowed] = -np.inf
        if not np.any(np.isfinite(values)):
            raise ValidationError("sampling controls masked every vocabulary token")
        if self.temperature == 0:
            result = np.zeros(len(values), dtype=np.float64)
            result[int(np.argmax(values))] = 1.
            return result
        values -= np.max(values)
        with np.errstate(over="ignore", under="ignore"):
            values /= self.temperature
        if self.top_k and self.top_k < len(values):
            order = np.argsort(-values, kind="stable")
            values[order[self.top_k:]] = -np.inf
        result = np.exp(values)
        result /= result.sum(dtype=np.float64)
        if self.min_p:
            result[result < self.min_p * np.max(result)] = 0.
            result /= result.sum(dtype=np.float64)
        if self.top_p < 1:
            order = np.argsort(-result, kind="stable")
            previous = np.cumsum(result[order], dtype=np.float64) - result[order]
            remove = previous >= self.top_p
            remove[0] = False
            result[order[remove]] = 0.
        total = result.sum(dtype=np.float64)
        if not np.isfinite(total) or total <= 0:
            raise ValidationError("sampling controls produced no positive probability")
        return result / total


def _distribution(values):
    values = np.asarray(values)
    if (values.ndim != 1 or not 1 <= len(values) <= 129280 or values.dtype != np.float64 or
            np.any(~np.isfinite(values)) or np.any(values < 0) or
            not math.isclose(float(values.sum(dtype=np.float64)), 1., rel_tol=0, abs_tol=1e-10)):
        raise ValidationError("verifier requires a normalized complete FP64 probability vector")
    return values


def categorical(probabilities, uniform):
    probabilities = _distribution(probabilities)
    positive = np.flatnonzero(probabilities)
    if len(positive) == 1:
        return int(positive[0])
    value = uniform()
    if not isinstance(value, float) or not 0 <= value < 1:
        raise ValidationError("sampling uniform must lie in [0,1)")
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative /= cumulative[-1]
    return int(np.searchsorted(cumulative, value, side="right"))


def correction_distribution(target, proposal):
    target, proposal = _distribution(target), _distribution(proposal)
    if target.shape != proposal.shape:
        raise ValidationError("target and proposal vocabulary sizes differ")
    residual = np.maximum(target - proposal, 0.)
    total = residual.sum(dtype=np.float64)
    if total <= 0 or not np.isfinite(total):
        raise ValidationError("a rejected proposal has no positive correction distribution")
    return residual / total


def _export(generator):
    state = generator.bit_generator.state
    return {"state": f"{state['state']['state']:032x}", "increment": f"{state['state']['inc']:032x}",
            "has_uint32": state["has_uint32"], "uinteger": state["uinteger"]}


def _import(state):
    if not isinstance(state, dict) or set(state) != {"state", "increment", "has_uint32", "uinteger"}:
        raise ValidationError("invalid PCG64 snapshot fields")
    from .protocol import hex_id
    for key in ("state", "increment"):
        hex_id(state[key], 32, f"PCG64 {key}")
    integer(state["has_uint32"], "PCG64 cached half-word flag", 0, 1)
    integer(state["uinteger"], "PCG64 cached word", 0, (1 << 32) - 1)
    increment = int(state["increment"], 16)
    if increment % 2 != 1:
        raise ValidationError("PCG64 increment must be odd")
    generator = np.random.Generator(np.random.PCG64(0))
    generator.bit_generator.state = {"bit_generator": "PCG64", "state": {
        "state": int(state["state"], 16), "inc": increment},
        "has_uint32": state["has_uint32"], "uinteger": state["uinteger"]}
    return generator


class RandomStreams:
    def __init__(self, seed: int):
        integer(seed, "sampling seed", 0, (1 << 64) - 1)
        self._streams = {}
        for domain in DOMAINS:
            digest = hashlib.sha256(b"DeepSeekRPC/PCG64/v1/" + seed.to_bytes(8, "little") + domain.encode()).digest()
            self._streams[domain] = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))
        self.active = False

    def snapshot(self):
        if self.active:
            raise ValidationError("terminal RNG snapshot requires a committed idle boundary")
        return self._snapshot()

    def _snapshot(self):
        return {"algorithm": "PCG64-domains-v1", "domains": {name: _export(value) for name, value in self._streams.items()}}

    def restore(self, state):
        if self.active:
            raise ValidationError("RNG restore requires an idle terminal sampler")
        self._restore(state)

    def _restore(self, state):
        if (not isinstance(state, dict) or set(state) != {"algorithm", "domains"} or
                state["algorithm"] != "PCG64-domains-v1" or not isinstance(state["domains"], dict) or
                set(state["domains"]) != set(DOMAINS)):
            raise ValidationError("incompatible terminal RNG snapshot")
        # Validate every domain before replacing any live state.
        streams = {name: _import(state["domains"][name]) for name in DOMAINS}
        self._streams = streams

    def begin(self):
        if self.active:
            raise ValidationError("terminal RNG transaction is already active")
        before = self.snapshot()
        self.active = True
        return RandomTransaction(self, before)


class RandomTransaction:
    def __init__(self, owner, before):
        self.owner, self.before, self.closed = owner, before, False
        self.draws = dict.fromkeys(DOMAINS, 0)

    def random(self, domain):
        if self.closed or domain not in DOMAINS or self.draws[domain] >= 64:
            raise ValidationError("terminal RNG domain is closed, invalid or exceeds its bounded window")
        self.draws[domain] += 1
        return float(self.owner._streams[domain].random())

    def commit(self):
        if self.closed:
            raise ValidationError("terminal RNG transaction is already closed")
        self.closed, self.owner.active = True, False
        return self.owner.snapshot()

    def prepared_state(self):
        if self.closed:
            raise ValidationError("terminal RNG transaction is already closed")
        return self.owner._snapshot()

    def abort(self):
        if self.closed:
            raise ValidationError("terminal RNG transaction is already closed")
        self.owner._restore(self.before)
        self.closed, self.owner.active = True, False


@dataclass(frozen=True)
class VerificationDecision:
    accepted_drafts: int
    output_ids: tuple[int, ...]
    pending_token: int | None
    stop_reason: str | None

    @property
    def accepted_target_rows(self):
        # The first verification input was the already sampled seed token.
        return 1 + self.accepted_drafts


def verify_drafts(target_rows, proposal_rows, proposal_ids, transaction: RandomTransaction, *,
                  greedy: bool, output_limit: int, eos_ids=frozenset()):
    """min(1,p/q), positive-part correction and one target bonus if all accept.

    Targets are computed through A/B/C, including Engram, before this operation.
    Full proposal distributions include the conditional Markov bias and the
    same requested sampling/grammar controls as their matching target prefix.
    """
    integer(len(proposal_ids), "proposal length", 0, 5)
    integer(output_limit, "remaining output limit", 1, 1048576)
    if type(greedy) is not bool or len(proposal_rows) != len(proposal_ids) or len(target_rows) != len(proposal_ids) + 1:
        raise ValidationError("target/proposal verification row counts differ")
    proposal_rows = tuple(_distribution(row) for row in proposal_rows)
    first_target = _distribution(target_rows[0])
    vocabulary = len(first_target)
    if any(len(row) != vocabulary for row in proposal_rows):
        raise ValidationError("verification probability vocabularies differ")
    def target_at(index):
        # Terminal grammar may end at an accepted EOS. A later distribution
        # is then undefined and never needed; construct only used prefixes.
        result = first_target if index == 0 else _distribution(target_rows[index])
        if len(result) != vocabulary:
            raise ValidationError("verification probability vocabularies differ")
        return result
    for token in (*proposal_ids, *eos_ids):
        integer(token, "verification token", 0, vocabulary - 1)
    output = []
    for index, token in enumerate(proposal_ids):
        target, proposal = target_at(index), proposal_rows[index]
        if proposal[token] <= 0:
            raise ValidationError("a proposed token has zero probability under its claimed conditional proposal")
        if greedy:
            accepted = token == int(np.argmax(target))
        else:
            accepted = transaction.random("accept") < min(1., float(target[token] / proposal[token]))
        if not accepted:
            replacement = (int(np.argmax(target)) if greedy else
                           categorical(correction_distribution(target, proposal), lambda: transaction.random("target")))
            return VerificationDecision(index, tuple(output) + (replacement,), replacement,
                                        "eos" if replacement in eos_ids else "length" if len(output) + 1 >= output_limit else None)
        output.append(token)
        if token in eos_ids or len(output) >= output_limit:
            return VerificationDecision(index + 1, tuple(output), None, "eos" if token in eos_ids else "length")
    bonus_target = target_at(len(proposal_ids))
    bonus = (int(np.argmax(bonus_target)) if greedy else
             categorical(bonus_target, lambda: transaction.random("target")))
    return VerificationDecision(len(proposal_ids), tuple(output) + (bonus,), bonus,
                                "eos" if bonus in eos_ids else "length" if len(output) + 1 >= output_limit else None)
