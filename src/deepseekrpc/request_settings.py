"""Immutable bounded per-request sampling settings shared by all three stages."""
from dataclasses import asdict, dataclass

from .jsonutil import ValidationError, canonical, fingerprint, integer
from .runtime_messages import fields
from .sampling import SamplingPolicy


@dataclass(frozen=True)
class RequestSettings:
    seed: int
    max_tokens: int
    policy: SamplingPolicy = SamplingPolicy()
    speculative: bool = False
    eos_ids: tuple[int, ...] = (1,)
    logit_bias: tuple[tuple[int, float], ...] = ()
    stop: tuple[str, ...] = ()

    def __post_init__(self):
        integer(self.seed, "request sampling seed", 0, (1 << 64) - 1)
        integer(self.max_tokens, "maximum generated token count", 1, 1048576)
        if not isinstance(self.policy, SamplingPolicy) or type(self.speculative) is not bool:
            raise ValidationError("invalid immutable request sampling policy")
        if (not isinstance(self.eos_ids, tuple) or len(self.eos_ids) > 64 or
                not isinstance(self.logit_bias, tuple) or len(self.logit_bias) > 256):
            raise ValidationError("EOS or logit-bias inventory exceeds bounded request metadata")
        for token in self.eos_ids:
            integer(token, "request EOS token", 0, 129279)
        previous = -1
        for item in self.logit_bias:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValidationError("logit bias requires immutable sorted token/value pairs")
            token, bias = item
            integer(token, "biased request token", previous + 1, 129279)
            if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not -100 <= bias <= 100:
                raise ValidationError("logit bias must be finite and in [-100, 100]")
            previous = token
        if len(set(self.eos_ids)) != len(self.eos_ids):
            raise ValidationError("request EOS token IDs must be unique")
        if (not isinstance(self.stop, tuple) or len(self.stop) > 16 or
                any(not isinstance(s, str) or not 1 <= len(s.encode("utf-8")) <= 4096 for s in self.stop) or
                len(set(self.stop)) != len(self.stop) or sum(len(s.encode("utf-8")) for s in self.stop) > 8192):
            raise ValidationError("request stop strings exceed their bounded immutable metadata inventory")
        if len(canonical(asdict(self))) > 12 * 1024:
            raise ValidationError("combined sampling settings exceed the bounded control metadata budget")

    @property
    def digest(self):
        return fingerprint(asdict(self))

    @classmethod
    def from_wire(cls, value):
        fields(value, cls.__dataclass_fields__)
        fields(value["policy"], SamplingPolicy.__dataclass_fields__)
        if any(not isinstance(value[key], list) for key in ("eos_ids", "logit_bias", "stop")):
            raise ValidationError("request EOS and bias metadata must be JSON arrays")
        if any(not isinstance(item, list) for item in value["logit_bias"]):
            raise ValidationError("request logit bias entries must be JSON pairs")
        return cls(**{**value, "policy": SamplingPolicy(**value["policy"]),
                      "eos_ids": tuple(value["eos_ids"]), "logit_bias": tuple(tuple(item) for item in value["logit_bias"]),
                      "stop": tuple(value["stop"])})

    def as_dict(self):
        result = asdict(self)
        result["eos_ids"], result["logit_bias"] = list(self.eos_ids), [list(item) for item in self.logit_bias]
        result["stop"] = list(self.stop)
        return result
