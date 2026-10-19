"""Disk lookup replacement retaining vLLM's real learned Engram projection/gate."""
from __future__ import annotations

import hashlib
import threading

import numpy as np
import torch
from torch import nn

from .cuda_ops import decode_engram
from .engram_storage import ENCODED_BYTES, EngramRows, MAX_BATCH
from .jsonutil import ValidationError, integer
from .planner import Role

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.models.deepseek_v41.common.engram import Engram as BaseEngram


class DiskEngramEmbedding(nn.Module):
    """At most 1024 encoded rows per read/upload, not a whole host/UVA table."""
    def __init__(self, store: EngramRows, *, num_embeddings: int, head_sizes, device):
        super().__init__()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("disk Engram execution requires the actual CUDA runtime")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if store.closed or not store.verified or store.rows != num_embeddings or get_tensor_model_parallel_world_size() != 1:
            raise ValidationError("disk Engram requires the verified original table and TP=1")
        if len(head_sizes) != 24 or any(type(size) is not int or size <= 0 for size in head_sizes) or sum(head_sizes) > num_embeddings:
            raise ValidationError("disk Engram hash-head geometry differs from October 1017")
        self.store, self.num_embeddings, self.dim = store, num_embeddings, 256
        self.tp_size = 1
        self.n_hash_cols = self.part_n_hash_cols = 24
        self._head_starts = np.cumsum([0, *head_sizes[:-1]], dtype=np.int64)
        self._head_ends = self._head_starts + np.asarray(head_sizes, dtype=np.int64)
        self._mutex = threading.Lock()
        self._pinned = torch.empty((MAX_BATCH, ENCODED_BYTES), dtype=torch.uint8, device="cpu", pin_memory=True)
        self._encoded = torch.empty((MAX_BATCH, ENCODED_BYTES), dtype=torch.uint8, device=self.device)
        # Includes worst-case simultaneous encoded NumPy gather/construction,
        # pinned and CUDA staging, decoded output, and integer hash metadata.
        self.staging_reservation_bytes = 2 * 1024 * 1024

    def lookup(self, indices, out, background=False):
        if (indices.device != self.device or indices.dtype != torch.int64 or indices.ndim != 2 or
                indices.shape[1] != 24 or not 1 <= len(indices) <= 64 or out.device != self.device or
                out.dtype != torch.bfloat16 or out.shape != (len(indices), 24, 256) or not out.is_contiguous()):
            raise ValidationError("disk Engram lookup has incompatible bounded hash/output metadata")
        ids = indices.detach().cpu().numpy()
        valid = ids != -1
        if np.any(valid & ((ids < self._head_starts) | (ids >= self._head_ends))):
            raise ValidationError("disk Engram hash points outside its original head's bucket range")
        ids = ids.reshape(-1)
        with self._mutex, torch.cuda.device(self.device):
            flat_output = out.view(-1, 256)
            for start in range(0, len(ids), MAX_BATCH):
                selected = ids[start:start + MAX_BATCH]
                encoded = np.zeros((len(selected), ENCODED_BYTES), dtype=np.uint8)
                valid = selected >= 0
                # -1 is vLLM's explicit dead/padding hash. Valid missing rows
                # never become zero embeddings or alternative model payloads.
                if np.any(valid):
                    encoded[valid] = self.store.gather_encoded(selected[valid].astype(np.int64))
                self._pinned[:len(selected)].copy_(torch.from_numpy(encoded))
                self._encoded[:len(selected)].copy_(self._pinned[:len(selected)], non_blocking=False)
                decoded = decode_engram(self._encoded[:len(selected)])
                flat_output[start:start + len(selected)].copy_(decoded)
                del decoded, encoded


class DiskEngram(BaseEngram):
    """Original vLLM wkv/q/k weights and learned CUDA gate, with disk row lookup."""
    def __init__(self, config, quant_config, layout, layer_hash_index, use_sequence_parallel,
                 prefix, *, role: Role, row_store: EngramRows, device):
        if (role != Role.HOST or use_sequence_parallel or row_store.layer != config.engram_layer_ids[layer_hash_index]):
            raise ValidationError("both disk Engram layers belong to stage A with TP=1")
        if not torch.cuda.is_available():
            raise RuntimeError("learned disk Engram needs the actual CUDA model runtime")
        self._disk_store, self._disk_device = row_store, torch.device(device)
        self._lookup_stamp = None
        self._forward_mutex = threading.RLock()
        with torch.device(self._disk_device):
            super().__init__(config, quant_config, layout, layer_hash_index, False, prefix)

    def _create_embedding(self, layout, layer_hash_index):
        heads = tuple(int(size) for order in layout.primes[layer_hash_index] for size in order)
        return DiskEngramEmbedding(self._disk_store, num_embeddings=int(layout.num_embeddings[layer_hash_index]),
                                   head_sizes=heads, device=self._disk_device)

    def _init_staging(self, max_tokens, head_dim):
        integer(max_tokens, "Engram staged token rows", 1, 64)
        if head_dim != 256:
            raise ValidationError("Engram staging width differs from the selected table")
        super()._init_staging(max_tokens, head_dim)

    @staticmethod
    def _stamp(hash_ids):
        if (not hash_ids.is_cuda or hash_ids.dtype != torch.int64 or hash_ids.ndim != 2 or
                hash_ids.shape[1] != 24 or not 1 <= len(hash_ids) <= 64):
            raise ValidationError("Engram hash stamp requires bounded CUDA int64 rows")
        return hashlib.sha256(hash_ids.detach().cpu().numpy().astype("<i8", copy=False).tobytes()).digest()

    def prepare_embeddings(self, hash_ids):
        with self._forward_mutex:
            self._lookup_stamp = None
            super().prepare_embeddings(hash_ids)
            self._lookup_stamp = self._stamp(hash_ids)

    def forward(self, hidden_states, hash_ids, token_mask=None):
        with self._forward_mutex:
            if (hidden_states.device != self.embed_tokens.device or hidden_states.dtype != torch.bfloat16 or
                    hidden_states.shape != (len(hash_ids), self.hc_mult, self.dim) or
                    token_mask is not None and (token_mask.device != hidden_states.device or
                                               token_mask.dtype != torch.bool or token_mask.shape != (len(hash_ids),))):
                raise ValidationError("learned Engram state/mask differs from the admitted CUDA graph")
            if self._lookup_stamp is None or self._lookup_stamp != self._stamp(hash_ids):
                raise ValidationError("Engram rows were not prepared for the current hashes")
            try:
                result = super().forward(hidden_states, hash_ids, token_mask)
                torch.cuda.current_stream(hidden_states.device).synchronize()
                return result
            finally:
                self._lookup_stamp = None
