"""Strict typed metadata codecs shared by real stage services/coordinators."""
from dataclasses import asdict

from .attention_state import AttentionWindow
from .execution import StepInput
from .jsonutil import ValidationError, integer
from .phases import PhaseReceipt
from .planner import Role
from .publication import OutputPlan, OutputReceipt
from .state import Operation


def fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValidationError("unknown or missing runtime command fields")
    return value


def window_from(value):
    fields(value, AttentionWindow.__dataclass_fields__)
    operation = fields(value["operation"], Operation.__dataclass_fields__)
    return AttentionWindow(Operation(**operation), value["history_digest"], value["phase"], value["encoder_tokens"])


def input_from(value):
    fields(value, StepInput.__dataclass_fields__)
    if not isinstance(value["tokens"], list) or not isinstance(value["text_mask"], list):
        raise ValidationError("step input requires JSON token and image-mask arrays")
    return StepInput(tuple(value["tokens"]), tuple(value["text_mask"]))


def output_from(value):
    fields(value, OutputPlan.__dataclass_fields__)
    if not isinstance(value["output_ids"], list):
        raise ValidationError("published output IDs must be a JSON array")
    return OutputPlan(**{**value, "output_ids": tuple(value["output_ids"])})


def receipt_from(value, *, output=False):
    cls = OutputReceipt if output else PhaseReceipt
    fields(value, cls.__dataclass_fields__)
    return cls(**{**value, "role": Role(integer(value["role"], "receipt stage", 0, 2))})


def receipts_from(value, *, output=False):
    if not isinstance(value, list) or len(value) != 3:
        raise ValidationError("distributed commit requires three actual stage receipts")
    return tuple(receipt_from(r, output=output) for r in value)


def wire(value):
    """Return JSON-compatible dataclass metadata, preserving exact enum integers."""
    result = asdict(value)
    if "role" in result:
        result["role"] = int(result["role"])
    # canonical JSON emits tuples as arrays; returning lists also supports an
    # in-process coordinator without a different codec path.
    if isinstance(value, OutputPlan):
        result["output_ids"] = list(value.output_ids)
    elif isinstance(value, StepInput):
        result["tokens"], result["text_mask"] = list(value.tokens), list(value.text_mask)
    return result
