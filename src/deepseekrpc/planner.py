"""Configuration-derived contiguous ownership and shared-state dependencies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum

from .jsonutil import ValidationError, integer


class Role(IntEnum):
    HOST = 0
    MIDDLE = 1
    TERMINAL = 2

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def dns_name(self) -> str:
        return f"{self.label}.deepseekrpc"


@dataclass(frozen=True)
class Geometry:
    layers: int
    hidden: int
    streams: int
    context: int
    kv_width: int
    index_width: int
    window: int
    kv_sources: tuple[int, ...]
    index_sources: tuple[int, ...]
    compress_ratios: tuple[int, ...]
    engram_layers: tuple[int, ...]
    candidate_source: int

    @classmethod
    def from_config(cls, config: dict) -> Geometry:
        c = config.get("text_config", config)
        layers = integer(c["num_hidden_layers"], "layers", 1, 1000)
        def source_list(key):
            values = tuple(c[key])
            if values != tuple(sorted(set(values))):
                raise ValidationError(f"{key} must be sorted and unique")
            for value in values:
                integer(value, key, 0, layers - 1)
            return values
        kv = source_list("kv_source_layer_ids")
        indices = source_list("index_source_layer_ids")
        ratios = tuple(c["compress_ratios"][:layers])
        if len(ratios) != layers:
            raise ValidationError("one compression ratio per backbone layer required")
        for ratio in ratios:
            integer(ratio, "compression ratio", 0, 2)
        if not set(kv) <= set(indices):
            raise ValidationError("KV sources must also produce index K")
        for layer, ratio in enumerate(ratios):
            if ratio:
                if not any(s <= layer for s in kv) or not any(s <= layer for s in indices):
                    raise ValidationError("compressed consumer lacks a preceding source")
                source = max(s for s in kv if s <= layer)
                if ratios[source] != ratio:
                    raise ValidationError("consumer and KV source compression ratios disagree")
        if any(ratios[s] == 0 for s in kv):
            raise ValidationError("KV source must have positive compression ratio")
        return cls(
            layers, integer(c["hidden_size"], "hidden", 1, 65536),
            integer(c["hc_mult"], "streams", 1, 16),
            integer(c["max_position_embeddings"], "context", 1, 1 << 31),
            integer(c["head_dim"], "KV width", 1, 65536),
            integer(c["index_head_dim"], "index width", 1, 65536),
            integer(c["sliding_window"], "window", 1, 65536), kv, indices,
            ratios, source_list("engram_layer_ids"),
            integer(c["candidate_source_layer_id"], "candidate source", 0, layers - 1),
        )


@dataclass(frozen=True)
class Dependency:
    field: str
    source_layer: int
    owner: Role
    recipient: Role


@dataclass(frozen=True)
class StagePlan:
    geometry: Geometry
    split: tuple[int, int, int]
    cache_dtype: str = "bfloat16"

    def __post_init__(self):
        if self.cache_dtype != "bfloat16":
            raise ValidationError("this deployment requires unquantized BF16 KV/index caches")
        if len(self.split) != 3:
            raise ValidationError("exactly three nonempty stages required")
        for count in self.split:
            integer(count, "stage length", 1, self.geometry.layers)
        if sum(self.split) != self.geometry.layers:
            raise ValidationError("stage split must cover every layer exactly once")
        if any(self.owner(layer) != Role.HOST for layer in self.geometry.engram_layers):
            raise ValidationError("both Engram layers must remain on host")

    def bounds(self, role: Role) -> tuple[int, int]:
        begin = sum(self.split[:int(role)])
        return begin, begin + self.split[int(role)]

    def owner(self, layer: int) -> Role:
        integer(layer, "layer", 0, self.geometry.layers - 1)
        for role in Role:
            begin, end = self.bounds(role)
            if begin <= layer < end:
                return role
        raise AssertionError("validated split failed to cover layer")

    def dependencies(self) -> tuple[Dependency, ...]:
        result = set()
        g = self.geometry
        for layer, ratio in enumerate(g.compress_ratios):
            if not ratio:
                continue
            kv = max(s for s in g.kv_sources if s <= layer)
            index = max(s for s in g.index_sources if s <= layer)
            for field, source in (("kv", kv), ("index_k", kv), ("topk", index)):
                owner, recipient = self.owner(source), self.owner(layer)
                if owner != recipient:
                    result.add(Dependency(field, source, owner, recipient))
            if layer > g.candidate_source:
                owner, recipient = self.owner(g.candidate_source), self.owner(layer)
                if owner != recipient:
                    result.add(Dependency("candidates", g.candidate_source, owner, recipient))
        return tuple(sorted(result, key=lambda d: (d.recipient, d.source_layer, d.field)))

    def persistent_source_bytes(self, role: Role, *, element_bytes: int = 2) -> int:
        integer(element_bytes, "unquantized BF16 element bytes", 2, 2)
        sources = {s for s in self.geometry.kv_sources if self.owner(s) == role}
        sources.update(d.source_layer for d in self.dependencies()
                       if d.field == "kv" and d.recipient == role)
        return sum(
            ((self.geometry.context + self.geometry.compress_ratios[s] - 1)
             // self.geometry.compress_ratios[s])
            * (self.geometry.kv_width + self.geometry.index_width) * element_bytes
            for s in sources
        )

    def as_dict(self) -> dict:
        return {
            "split": list(self.split),
            "bounds_half_open": {r.label: list(self.bounds(r)) for r in Role},
            "dependencies": [dict(asdict(d), owner=d.owner.label, recipient=d.recipient.label)
                             for d in self.dependencies()],
            "cache_dtype": self.cache_dtype,
            "source_and_shadow_bytes": {r.label: self.persistent_source_bytes(r) for r in Role},
            "allocation_status": "prospective arithmetic; not allocated or hardware measured",
        }
