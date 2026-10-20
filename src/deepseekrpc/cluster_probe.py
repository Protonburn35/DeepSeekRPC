"""Real installed-cluster diagnostics and an incarnation-bound DSpark gate.

Split-versus-itself comparisons are metamorphic evidence, not independent
full-model correctness. Operation goldens remain the small local CPU equations.
This module never substitutes a model or emulates another stage.
"""
from contextlib import closing
from dataclasses import replace
import base64
import hashlib
import io
import threading
from types import SimpleNamespace
import uuid

import numpy as np

from .jsonutil import ValidationError, fingerprint
from .planner import Role


def sampler_checks():
    """Deterministic real verifier branches plus independent correction law."""
    from .sampling import RandomStreams, correction_distribution, verify_drafts
    point = np.eye(3, dtype=np.float64)
    count = 0
    for rejected in range(6):
        streams = RandomStreams(41017)
        transaction = streams.begin()
        targets = [point[0] if row < rejected else point[1] for row in range(5)] + [point[2]]
        result = verify_drafts(targets, [point[0]] * 5, [0] * 5, transaction,
            greedy=True, output_limit=64)
        expected = (0,) * rejected + ((1,) if rejected < 5 else (2,))
        if result.accepted_drafts != rejected or result.output_ids != expected:
            raise AssertionError("real verifier differs from independent greedy rejection-prefix equation")
        transaction.abort()
        if streams.snapshot() != RandomStreams(41017).snapshot():
            raise AssertionError("aborted verifier changed a transactional RNG domain")
        count += 1
    for limit in range(1, 6):
        transaction = RandomStreams(41017).begin()
        result = verify_drafts([point[0]] * 6, [point[0]] * 5, [0] * 5, transaction,
            greedy=False, output_limit=limit)
        if result.output_ids != (0,) * limit or result.stop_reason != "length" or result.pending_token is not None:
            raise AssertionError("real verifier crossed the independently fixed output boundary")
        transaction.abort()
        count += 1
    transaction = RandomStreams(41017).begin()
    result = verify_drafts([point[0]] * 6, [point[0]] * 5, [0] * 5, transaction,
        greedy=False, output_limit=64, eos_ids=frozenset({0}))
    if result.output_ids != (0,) or result.accepted_drafts != 1 or result.stop_reason != "eos" or result.pending_token is not None:
        raise AssertionError("real verifier failed the independent EOS-within-drafts boundary")
    transaction.abort()
    p, q = np.array([.1, .6, .3]), np.array([.5, .2, .3])
    accepted_mass = np.minimum(p, q)
    corrected = accepted_mass + (1 - accepted_mass.sum()) * correction_distribution(p, q)
    np.testing.assert_allclose(corrected, p, rtol=0, atol=2e-16)
    return {"checks": count + 2, "status": "passed", "scope": "independent tiny probability/branch equations; no model execution"}


def probe_stage(service):
    """Called only by the actual service actor under its CUDA compute lock."""
    import torch
    from .resident_weights import LockedOwnedWeights
    stage, ledger = service.execution.stage, service.execution.ledger
    if ledger.phase != "decode" or not ledger.quiescent or service.execution._window is not None:
        raise ValidationError("cluster probe requires an idle committed target boundary")
    startup = stage.report["small_CUDA_checks"]
    if any(item.get("status") != "passed" for item in startup["checks"]):
        raise ValidationError("required actual local CUDA diagnostics have not passed")
    stage.stream.synchronize()
    pools, eager_bytes = {}, 0
    for kind in ("sources", "index_keys"):
        for layer, pool in getattr(stage.attention, kind).items():
            if pool.closed or pool.data.dtype != torch.bfloat16 or pool.map.active is not None:
                raise ValidationError("probe found a closed, non-BF16 or tentative source allocation")
            count = ledger.processed // stage.plan.geometry.compress_ratios[layer]
            positions = sorted(set(range(min(count, 8))) | set(range(max(0, count - 8), count)))
            digest = hashlib.sha256()
            if positions:
                physical = torch.tensor(pool.physical(positions), dtype=torch.int64, device=pool.device)
                raw = pool.data.index_select(0, physical).contiguous().view(torch.uint8).cpu().numpy().tobytes()
                digest.update(raw)
            pools[f"{kind}.{layer}"] = {"logical_rows": count, "sampled_positions": positions,
                "width": pool.width, "dtype": "bf16", "sha256": digest.hexdigest()}
    for pool in stage.attention._pools:
        if pool.closed or pool.data.shape != (pool.map.slots, pool.width):
            raise ValidationError("eager cache allocation does not match its admitted capacity")
        eager_bytes += pool.allocation_bytes
    residency = stage.weights.revalidate_residency() if isinstance(stage.weights, LockedOwnedWeights) else None
    return {"stage": int(stage.role), "identity": stage.identity, "mesh": service.mesh_nonce,
        "history": service.execution.history.digest, "processed": ledger.processed,
        "pending_token": ledger.pending_token, "pools": pools, "eager_pool_bytes": eager_bytes,
        "budgets": stage.guard.check(), "locked_weight_pages": residency,
        "local_GPU_checks": len(startup["checks"]), "architecture": startup["architecture"],
        "rng_sha256": fingerprint(service._sampler.random.snapshot()) if stage.role == Role.TERMINAL else None,
        "stops_sha256": fingerprint(service._stops.snapshot()) if stage.role == Role.TERMINAL else None}


def validate_replicas(peers, plan):
    if set(peers) != set(Role) or any(peers[role]["stage"] != int(role) for role in Role):
        raise ValidationError("probe needs one real authenticated receipt from every stage")
    first = peers[Role.HOST]
    for peer in peers.values():
        if any(peer[key] != first[key] for key in ("identity", "mesh", "history", "processed", "pending_token")):
            raise ValidationError("probe stages disagree on their committed identity/timeline")
    for dependency in plan.dependencies():
        if dependency.field not in {"kv", "index_k"}:
            continue
        key = f"{'sources' if dependency.field == 'kv' else 'index_keys'}.{dependency.source_layer}"
        if peers[dependency.owner]["pools"][key] != peers[dependency.recipient]["pools"][key]:
            raise ValidationError("actual source and independent shadow sampled BF16 bytes differ")
    if peers[Role.MIDDLE]["locked_weight_pages"] is None:
        raise ValidationError("B did not revalidate its actual physically locked owned weight pages")
    return fingerprint({int(role): peer for role, peer in peers.items()})


class ClusterProbe:
    def __init__(self, coordinator, assets, tokenizer, *, model, snapshots):
        from .coordinator import ClusterCoordinator
        from .image_assets import ImageAssets
        from .snapshot_archive import SnapshotStore
        if (not isinstance(coordinator, ClusterCoordinator) or not isinstance(assets, ImageAssets) or
                not isinstance(snapshots, SnapshotStore) or assets.identity != coordinator.host.execution.stage.identity):
            raise ValidationError("cluster diagnostics require the actual loaded coordinator, image and snapshot stores")
        self.coordinator, self.assets, self.tokenizer = coordinator, assets, tokenizer
        self.model, self.snapshots = model, snapshots
        self.cancelled = threading.Event()
        self._observer = self.cancelled.is_set
        self._administration = None

    def cancel(self):
        self.cancelled.set()
        if self._administration is not None:
            self._administration.cancel()
        coordinator = self.coordinator
        with coordinator._cancel_lock:
            if coordinator._external_cancelled is not self._observer or not coordinator._request_lock.locked():
                return False
            coordinator._cancelled.set()
            coordinator.host.actor.invalidate()
            coordinator.host.data.close()
            return True

    def _peers(self):
        coordinator, ledger = self.coordinator, self.coordinator.host.execution.ledger
        with coordinator._cancel_lock:
            if not coordinator._request_lock.acquire(blocking=False):
                raise ValidationError("another request owns the cluster diagnostic boundary")
            coordinator._cancelled.clear()
            coordinator._external_cancelled = self._observer
            coordinator._spec = SimpleNamespace(session=ledger.session, request=ledger.request,
                epoch=ledger.epoch, transaction=uuid.uuid4().hex)
        try:
            return coordinator._all({"action": "diagnostic_probe"})
        finally:
            with coordinator._cancel_lock:
                coordinator._spec = coordinator._window = coordinator._external_cancelled = None
                coordinator._request_lock.release()

    def _collect(self, generator, *, output):
        stats, final = [], None
        with closing(generator):
            for batch in generator:
                if self.cancelled.is_set():
                    raise InterruptedError("cluster diagnostics cancelled")
                output.extend(batch.output_ids)
                if batch.mode != "seed":
                    stats.append(dict(self.coordinator.last_decode))
                final = batch
        if final is None or final.stop_reason is None and not final.paused:
            raise ValidationError("diagnostic generator did not finish at a committed boundary")
        peers = self._peers()
        validate_replicas(peers, self.coordinator.host.execution.stage.plan)
        return final, stats, peers

    def _case(self, request, *, speculative, temperature):
        from .sampling import SamplingPolicy
        output = []
        settings = replace(request.settings, seed=41017, max_tokens=32, eos_ids=(),
            speculative=speculative, policy=SamplingPolicy(temperature=temperature, top_p=.95 if temperature else 1.))
        final, stats, peers = self._collect(self.coordinator.generate(request.history, settings, cancelled=self._observer), output=output)
        if len(output) != 32 or final.stop_reason != "length":
            raise ValidationError("fixed-length diagnostic did not commit exactly its requested output capacity")
        return output, stats, peers

    def run(self):
        from .chat_input import prepare_chat
        from .snapshot_coordinator import SnapshotCoordinator
        coordinator = self.coordinator
        ledger = coordinator.host.execution.ledger
        if ledger.phase == "decode" and ledger.stop_reason is None:
            raise ValidationError("finish or save the paused user session before running destructive session diagnostics")
        if coordinator._qualification_thread is not None:
            raise ValidationError("cluster qualification is already active")
        coordinator._speculation_proof = None
        coordinator._qualification_thread = threading.get_ident()
        report = {"format": 1, "identity": coordinator.host.execution.stage.identity,
            "mesh": coordinator.host.mesh_nonce, "scope": "actual loaded three-stage metamorphic diagnostics; no independent full-model golden",
            "checks": [],
            "EOS_policy": "ignored only in fixed-length internal probes; independent verifier boundary checks run separately"}
        coordinator.last_qualification = report
        try:
            report["sampling_equations"] = sampler_checks()
            from PIL import Image
            pixels = np.indices((32, 32)).sum(axis=0) % 2 * 255
            image = Image.fromarray(np.stack([pixels, 255 - pixels, pixels], axis=-1).astype(np.uint8))
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            uri = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
            text = "Return a numbered list of short words. Reference numbers: " + ", ".join(str(n) for n in range(64))
            for label, content in (("text", text), ("image", [{"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": uri}}])):
                body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                        "thinking": {"type": "disabled"}, "max_tokens": 64, "speculative": False}
                with prepare_chat(body, model=self.model, tokenizer=self.tokenizer, assets=self.assets) as request:
                    baseline, _, _ = self._case(request, speculative=False, temperature=0)
                    repeated, _, _ = self._case(request, speculative=False, temperature=0)
                    if (repeated != baseline or coordinator.last_prefill["reused_encoder_tokens"] <= 0 or
                            coordinator.last_prefill["target_replay_tokens"] != request.history.length):
                        raise ValidationError("exact repeated/prefix-reused target decoding changed deterministic tokens")
                    speculative, stats, _ = self._case(request, speculative=True, temperature=0)
                    if speculative != baseline or not any(item["proposed_tokens"] for item in stats):
                        raise ValidationError("actual greedy DSpark output differs from normal target decoding or did not exercise proposals")
                    sampled, sampled_stats, first = self._case(request, speculative=True, temperature=.7)
                    repeated, _, second = self._case(request, speculative=True, temperature=.7)
                    if (sampled != repeated or first[Role.TERMINAL]["rng_sha256"] != second[Role.TERMINAL]["rng_sha256"] or
                            any(first[role]["pools"] != second[role]["pools"] for role in Role)):
                        raise ValidationError("actual sampled DSpark replay changed tokens, RNG or committed source bytes")
                    report["checks"].append({"name": f"{label}: target repeat/prefix reuse, greedy DSpark parity, sampled repeat/RNG/source replicas",
                        "status": "passed", "prompt_tokens": request.history.length,
                        "output_sha256": hashlib.sha256(np.array(sampled, dtype="<i4").tobytes()).hexdigest(),
                        "greedy_windows": stats, "sampled_windows": sampled_stats})
                    if label == "text":
                        # Test the actual all-stage snapshot/restore and pending
                        # input continuation with speculative state and RNG.
                        settings = replace(request.settings, seed=41017, max_tokens=32, eos_ids=(), speculative=True)
                        prefix = []
                        paused, _, _ = self._collect(coordinator.generate(request.history, settings,
                            cancelled=self._observer, paused=lambda: len(prefix) >= 7), output=prefix)
                        if not paused.paused:
                            raise ValidationError("actual diagnostic pause did not retain a pending input")
                        snapshots = SnapshotCoordinator(coordinator, self.snapshots)
                        self._administration = snapshots
                        if self.cancelled.is_set():
                            snapshots.cancel()
                        saved = snapshots.save()
                        try:
                            expected, actual = [], []
                            _, _, before = self._collect(coordinator.resume(coordinator.host.execution.ledger.session,
                                cancelled=self._observer), output=expected)
                            restored = snapshots.restore(saved["id"])
                            _, _, after = self._collect(coordinator.resume(restored["session"], cancelled=self._observer), output=actual)
                            if (actual != expected or before[Role.TERMINAL]["rng_sha256"] != after[Role.TERMINAL]["rng_sha256"] or
                                    any(before[role]["pools"] != after[role]["pools"] for role in Role)):
                                raise ValidationError("actual restore/continuation changed pending-input tokens, RNG or source bytes")
                        finally:
                            self.snapshots.remove(saved["id"])
                            self._administration = None
                        report["checks"].append({"name": "actual all-stage pause/save/restore/pending-input continuation with DSpark RNG",
                                                 "status": "passed"})
            if coordinator.host.mesh_nonce != report["mesh"] or self.cancelled.is_set():
                raise InterruptedError("cluster mesh changed during qualification")
            coordinator._speculation_proof = ((report["identity"], report["mesh"]), fingerprint(report))
            report["speculation"] = coordinator.speculation_status()
            report["status"] = "passed"
            return report
        except BaseException as error:
            coordinator._speculation_proof = None
            report["status"], report["failure"] = "failed", {"type": type(error).__name__}
            raise
        finally:
            coordinator._qualification_thread = None
