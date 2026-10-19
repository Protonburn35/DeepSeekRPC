"""Bounded BF16 index scoring and two-level logical-position selection."""
from __future__ import annotations

from dataclasses import dataclass

from .jsonutil import ValidationError, integer


def merge_ranked(scores, ids, new_scores, new_ids, *, keep: int):
    """Streaming exact score ranking, with lower logical IDs breaking ties.

    This small Torch operation is also exercised on CPU as a ranking diagnostic;
    production scoring below requires real CUDA cache pools and native kernels.
    """
    import torch
    scores, ids = torch.cat((scores, new_scores)), torch.cat((ids, new_ids))
    order = torch.argsort(ids, stable=True)
    order = order[torch.argsort(scores[order], descending=True, stable=True)[:keep]]
    return scores[order], ids[order]


@dataclass(frozen=True)
class IndexSelection:
    topk: object
    candidate_blocks: object | None


class TiledIndexer:
    """Score logical key tiles without a full-context logits/gather allocation.

    Queries/weights already contain the graph's RoPE and 128^-0.5*32^-0.5
    head-weight scale. They are FP32; stored keys are unquantized BF16. Dense
    index sources read bounded logical tiles. Later indexers read only the
    candidate source's blocks. Slot offsets are resolved before the tile gather,
    so native visibility never confuses logical position with a shadow's slot.
    """
    def __init__(self, *, tile_rows: int = 8192):
        self.tile_rows = integer(tile_rows, "index key tile rows", 8, 8192)
        if tile_rows % 8:
            raise ValidationError("index key tiles must align with candidate blocks of eight")

    def select(self, pool, operation, query, weights, visible_counts, *,
               produce_candidates: bool = False, candidate_blocks=None, cancelled=None):
        with pool.read_window(operation):
            return self._select(pool, operation, query, weights, visible_counts,
                                produce_candidates=produce_candidates, candidate_blocks=candidate_blocks,
                                cancelled=cancelled)

    def _select(self, pool, operation, query, weights, visible_counts, *,
                produce_candidates, candidate_blocks, cancelled):
        import torch
        from .cuda_ops import index_scores
        if (pool.closed or pool.width != 128 or pool.map.active != operation or
                not query.is_cuda or query.device != pool.device or query.dtype != torch.float32 or
                query.shape != (operation.rows, 32, 128) or not query.is_contiguous() or
                weights.device != query.device or weights.dtype != torch.float32 or
                weights.shape != (len(query), 32) or not weights.is_contiguous() or
                len(visible_counts) != len(query) or produce_candidates and candidate_blocks is not None):
            raise ValidationError("index selection state, precision, query geometry or candidate mode mismatch")
        for visible in visible_counts:
            integer(visible, "visible logical index keys", 0, pool.map.slots)
        candidates = None
        if candidate_blocks is not None:
            if (candidate_blocks.dtype != torch.int32 or candidate_blocks.device != query.device or
                    candidate_blocks.shape != (len(query), 2048)):
                raise ValidationError("candidate block tensor does not match the pinned topology")
            # At most 512 KiB of encoded integer metadata plus its bounded
            # Python decode overhead, never cached key payload.
            candidates = candidate_blocks.tolist()
            for row, blocks in enumerate(candidates):
                last, padded = -1, False
                for block in blocks:
                    if block == -1:
                        padded = True
                    elif padded or block <= last or not 0 <= block < (visible_counts[row] + 7) // 8:
                        raise ValidationError("candidate blocks are stale, duplicated or out of order")
                    else:
                        last = block
        topk = torch.full((len(query), 512), -1, dtype=torch.int32, device=query.device)
        block_output = torch.full((len(query), 2048), -1, dtype=torch.int32, device=query.device) if produce_candidates else None
        for row, visible in enumerate(visible_counts):
            if not visible:
                continue
            if candidates is None:
                positions = range(visible)
            else:
                positions = [position for block in candidates[row] if block >= 0
                             for position in range(block * 8, min((block + 1) * 8, visible))]
                if not positions:
                    raise ValidationError("a nonempty index query has no published candidate blocks")
            best_scores = torch.empty(0, dtype=torch.float32, device=query.device)
            best_ids = torch.empty(0, dtype=torch.int64, device=query.device)
            block_scores, block_ids = best_scores, best_ids
            for start in range(0, len(positions), self.tile_rows):
                if cancelled is not None and cancelled():
                    raise InterruptedError("index selection cancelled at a key tile boundary")
                logical = positions[start:start + self.tile_rows]
                physical = pool.physical(logical)
                slots = torch.tensor(physical, dtype=torch.int64, device=query.device)
                # A bounded BF16 gather is the sole copy. There is no cache-wide
                # dequantization, concatenation, quantization or CPU key mirror.
                keys = pool.data.index_select(0, slots)
                local = torch.arange(len(logical), dtype=torch.int32, device=query.device).unsqueeze(0)
                count = torch.tensor([len(logical)], dtype=torch.int32, device=query.device)
                scores = index_scores(query[row:row + 1], keys, weights[row:row + 1], local, count)[0]
                logical_ids = torch.tensor(logical, dtype=torch.int64, device=query.device)
                best_scores, best_ids = merge_ranked(best_scores, best_ids, scores, logical_ids, keep=512)
                if produce_candidates:
                    # Dense source tiles start at multiples of eight. Only the
                    # last tile can end in a partial block.
                    if len(scores) % 8:
                        padded_scores = torch.nn.functional.pad(scores, (0, 8 - len(scores) % 8), value=-float("inf"))
                    else:
                        padded_scores = scores
                    reduced = padded_scores.reshape(-1, 8).amax(-1)
                    new_ids = torch.arange(start // 8, start // 8 + len(reduced), dtype=torch.int64, device=query.device)
                    reduced = reduced.masked_fill(new_ids == (visible - 1) // 8, float("inf"))
                    block_scores, block_ids = merge_ranked(block_scores, block_ids, reduced, new_ids, keep=2048)
            chosen = best_ids.sort().values.to(torch.int32)
            topk[row, :len(chosen)] = chosen
            if produce_candidates:
                chosen_blocks = block_ids.sort().values.to(torch.int32)
                block_output[row, :len(chosen_blocks)] = chosen_blocks
        return IndexSelection(topk, block_output)
