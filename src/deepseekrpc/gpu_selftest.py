"""Small independent CUDA operation checks; no model or network is loaded."""
from __future__ import annotations

import json
import platform

import numpy as np

from .jsonutil import ValidationError
from .runtime_policy import configure_exact_cuda_math


def run_cuda_checks():
    configure_exact_cuda_math()
    import torch
    from . import cuda_ops
    from .engram_storage import decode_fp8_rows
    from .reference_attention import index_scores_reference, sparse_attention_reference

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA operation checks require a real GPU and the installed native module")
    device = torch.device("cuda", torch.cuda.current_device())
    architecture = torch.cuda.get_device_capability(device)
    if architecture not in ((8, 9), (12, 0)):
        raise RuntimeError("this candidate targets SM89 and SM120")
    report = {"scope": "small CUDA operation checks against independent CPU equations",
              "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "architecture": f"sm_{architecture[0]}{architecture[1]}", "checks": [],
              "full_model_validation": "not run", "physical_residency_validation": "not run"}
    from .runtime_identity import require_windows_runtime
    from .host_selftest import run_host_checks
    report["runtime_identity"] = require_windows_runtime(device.index)
    report["host_checks"] = run_host_checks()
    properties = torch.cuda.get_device_properties(device)
    report["device"] = {"name": properties.name, "total_bytes": properties.total_memory,
                        "multiprocessors": properties.multi_processor_count}
    rng = np.random.default_rng(41017)

    def tensor(array, dtype):
        return torch.as_tensor(array, dtype=dtype, device=device).contiguous()

    def bf16_array(array):
        # Use the exact stored BF16 values as oracle input; the oracle computes
        # its equations independently of the tested CUDA implementation.
        value = tensor(array, torch.bfloat16)
        return value, value.float().cpu().numpy()

    def record(name, function):
        function()
        torch.cuda.synchronize(device)
        report["checks"].append({"name": name, "status": "passed"})

    def attention():
        query = rng.normal(0, .2, (3, 64, 512)).astype(np.float32)
        window, window_ref = bf16_array(rng.normal(0, .3, (7, 512)))
        source, source_ref = bf16_array(rng.normal(0, .3, (11, 512)))
        indices = np.array([[0, 6, 7, 17, -1, -1], [5, 5, 9, 10, -1, 11], [-1] * 6], np.int32)
        sink = rng.normal(0, .3, 64).astype(np.float32)
        expected = sparse_attention_reference(query, window_ref, source_ref, indices, sink, scale=512 ** -.5)
        actual = cuda_ops.sparse_attention(tensor(query, torch.float32), window, source,
                                           tensor(indices, torch.int32), tensor(sink, torch.float32), scale=512 ** -.5)
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=5e-5, atol=3e-6)

    def index_scores():
        query = rng.normal(0, .3, (5, 32, 128)).astype(np.float32)
        keys, keys_ref = bf16_array(rng.normal(0, .3, (19, 128)))
        weights = rng.normal(0, .2, (5, 32)).astype(np.float32)
        indices = np.tile(np.array([0, 1, 18, 8, 8, -1, 3, 9], np.int32), (5, 1))
        visible = np.array([0, 1, 9, 18, 19], np.int32)
        expected = index_scores_reference(query, keys_ref, weights, indices, visible)
        actual = cuda_ops.index_scores(tensor(query, torch.float32), keys, tensor(weights, torch.float32),
                                       tensor(indices, torch.int32), tensor(visible, torch.int32))
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=8e-5, atol=5e-6)

    def engram_decode():
        encoded = np.empty((5, 264), np.uint8)
        values = np.arange(256, dtype=np.uint8)
        values[[127, 255]] = 0
        encoded[:, :256] = values
        encoded[:, 256:] = np.array([0, 64, 126, 127, 130], np.uint8)[:, None]
        expected = torch.from_numpy(decode_fp8_rows(encoded)).to(torch.bfloat16)
        actual = cuda_ops.decode_engram(tensor(encoded, torch.uint8)).cpu()
        torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)

    def scatter():
        pool = torch.zeros((13, 512), dtype=torch.bfloat16, device=device)
        generations = torch.arange(13, dtype=torch.int64, device=device)
        slots = tensor([9, 2, 12], torch.int32)
        expected_generations = tensor([9, 2, 12], torch.int64)
        values, _ = bf16_array(rng.normal(0, .2, (3, 512)))
        cuda_ops.scatter_slots(pool, generations, slots, expected_generations, values)
        torch.testing.assert_close(pool[slots.long()].view(torch.int16), values.view(torch.int16), rtol=0, atol=0)
        stale = expected_generations + 1
        try:
            cuda_ops.scatter_slots(pool, generations, slots, stale, values)
        except ValidationError:
            pass
        else:
            raise AssertionError("stale generation was accepted by native scatter")
        # This disposable pool is never published. Failed writes in an actual
        # session require its transaction journal to restore tentative state.

    def stream_events():
        producer, consumer = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
        ready, complete = torch.cuda.Event(), torch.cuda.Event()
        host = torch.zeros((3, 264), dtype=torch.uint8, pin_memory=True)
        host[:, :256], host[:, 256:] = 0x38, 127
        with torch.cuda.stream(producer):
            encoded = host.to(device, non_blocking=True)
            ready.record(producer)
        with torch.cuda.stream(consumer):
            consumer.wait_event(ready)
            result = cuda_ops.decode_engram(encoded)
            encoded.record_stream(consumer)
            complete.record(consumer)
        complete.synchronize()
        # Both producer storage and the pinned source remain alive until the
        # consuming event. Exercise the real custom kernel's current stream.
        torch.testing.assert_close(result.cpu(), torch.ones((3, 256), dtype=torch.bfloat16), rtol=0, atol=0)

    def row_pool_rollback():
        from .cache_pool import DeviceRowPool
        from .state import Operation
        pool = DeviceRowPool(2, 128, context=16, offset=1, admitted_bytes=1024 * 1024, device=device)
        try:
            pool.reset("a" * 32, 0)
            first = Operation("a" * 32, "b" * 32, 0, 1, "c" * 32, "d" * 64, 0, 2)
            pool.begin(first)
            original, _ = bf16_array(rng.normal(0, .2, (2, 128)))
            pool.write([0, 1], [0, 1], original)
            pool.finish(first, 2)
            second = Operation("a" * 32, "b" * 32, 0, 2, "e" * 32, "f" * 64, 2, 4)
            pool.begin(second)
            replacement, _ = bf16_array(rng.normal(0, .2, (4, 128)))
            pool.write([2, 3], [2, 3], replacement[:2].contiguous())
            pool.write([4, 5], [4, 5], replacement[2:].contiguous())
            pool.finish(second, 1)
            actual = pool.data[pool.physical([1, 2])].view(torch.int16)
            expected = torch.stack([original[1], replacement[0]]).view(torch.int16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            generations = pool.generations.clone()
            third = Operation("a" * 32, "b" * 32, 0, 3, "1" * 32, "2" * 64, 3, 2)
            pool.begin(third)
            pool.write([3, 4], [3, 4], replacement[2:].contiguous())
            pool.finish(third, 0)
            torch.testing.assert_close(pool.generations, generations, rtol=0, atol=0)
            torch.testing.assert_close(pool.data[pool.physical([1, 2])].view(torch.int16), expected, rtol=0, atol=0)
        finally:
            pool.close()

    def prefix_pool_retention():
        from .cache_pool import DeviceRowPool
        from .state import Operation
        pool = DeviceRowPool(8, 128, context=16, offset=3, admitted_bytes=1024 * 1024, device=device)
        try:
            pool.reset("a" * 32, 0)
            op = Operation("a" * 32, "b" * 32, 0, 1, "c" * 32, "d" * 64, 0, 8)
            pool.begin(op)
            values, _ = bf16_array(rng.normal(0, .2, (8, 128)))
            pool.write(list(range(8)), list(range(8)), values)
            pool.finish(op, 8)
            pool.retain_prefix("e" * 32, 1, 4)
            torch.testing.assert_close(pool.data[pool.physical(range(4))].view(torch.int16),
                                       values[:4].view(torch.int16), rtol=0, atol=0)
            np.testing.assert_array_equal(pool.generations.cpu().numpy(), pool.map.generations)
            try:
                pool.physical([4])
            except ValidationError:
                pass
            else:
                raise AssertionError("prefix retention left a tail row valid")
        finally:
            pool.close()

    def verification_rollback_lengths():
        from .cache_pool import DeviceRowPool
        from .state import Operation
        for accepted in range(7):
            pool = DeviceRowPool(8, 128, context=32, offset=3, admitted_bytes=1024 * 1024, device=device)
            try:
                pool.reset("a" * 32, 0)
                initial = Operation("a" * 32, "b" * 32, 0, 1, "c" * 32, "d" * 64, 0, 8)
                original, _ = bf16_array(rng.normal(0, .2, (8, 128)))
                pool.begin(initial)
                pool.write(list(range(8)), list(range(8)), original)
                pool.finish(initial, 8)
                tentative = Operation("a" * 32, "b" * 32, 0, 2, "e" * 32, "f" * 64, 8, 6)
                replacement, _ = bf16_array(rng.normal(0, .2, (6, 128)))
                pool.begin(tentative)
                pool.write(list(range(8, 14)), list(range(8, 14)), replacement)
                pool.finish(tentative, accepted)
                logical = list(range(accepted, 8 + accepted))
                expected = torch.cat((original[accepted:], replacement[:accepted]))
                torch.testing.assert_close(pool.data[pool.physical(logical)].view(torch.int16),
                    expected.view(torch.int16), rtol=0, atol=0)
                np.testing.assert_array_equal(pool.generations.cpu().numpy(), pool.map.generations)
                for rejected in range(8 + accepted, 14):
                    try:
                        pool.physical([rejected])
                    except ValidationError:
                        pass
                    else:
                        raise AssertionError("verification rollback retained a rejected logical input")
            finally:
                pool.close()

    def tiled_index():
        from .cache_pool import DeviceRowPool
        from .index_selection import TiledIndexer
        from .state import Operation
        pool = DeviceRowPool(2048, 128, context=4096, offset=997, admitted_bytes=4 * 1024 * 1024, device=device)
        try:
            pool.reset("a" * 32, 0)
            keys = np.zeros((1025, 128), np.float32)
            positions = np.arange(1025)
            keys[:, 0], keys[:, 1] = positions // 256, (positions % 256) / 256
            stored, _ = bf16_array(keys)
            for number, start in enumerate(range(0, 1025, 64), 1):
                count = min(64, 1025 - start)
                op = Operation("a" * 32, "b" * 32, 0, number, "c" * 32, "d" * 64, start, count)
                pool.begin(op)
                pool.write(list(range(start, start + count)), list(range(start, start + count)), stored[start:start + count])
                pool.finish(op, count)
            op = Operation("a" * 32, "b" * 32, 0, 18, "c" * 32, "d" * 64, 1025, 3)
            pool.begin(op)
            query = torch.zeros((3, 32, 128), dtype=torch.float32, device=device)
            query[:, 0, 0], query[:, 0, 1] = 256., 1.
            weights = torch.zeros((3, 32), dtype=torch.float32, device=device)
            weights[:, 0] = tensor([1., -1., 1.], torch.float32)
            selection = TiledIndexer(tile_rows=128).select(pool, op, query, weights, [513, 1025, 1025], produce_candidates=True)
            expected = np.stack([np.arange(1, 513), np.arange(512), np.arange(513, 1025)]).astype(np.int32)
            np.testing.assert_array_equal(selection.topk.cpu().numpy(), expected)
            candidates = torch.full((3, 2048), -1, dtype=torch.int32, device=device)
            blocks = [[2, 30, 64], [0, 127, 128], [3, 14, 80]]
            for row in range(3):
                candidates[row, :3] = tensor(blocks[row], torch.int32)
            selected = TiledIndexer(tile_rows=128).select(pool, op, query, weights, [513, 1025, 1025], candidate_blocks=candidates)
            for row, visible in enumerate([513, 1025, 1025]):
                expected_row = [p for block in blocks[row] for p in range(block * 8, min(block * 8 + 8, visible))]
                actual = selected.topk[row].cpu().tolist()
                assert actual == expected_row + [-1] * (512 - len(expected_row))
            pool.finish(op, 0)
        finally:
            pool.close()

    def vision_attention_shape():
        from torch.nn.attention import SDPBackend, sdpa_kernel
        # The actual tower's maximum admitted patch shape. Only four query
        # rows need independent CPU equations; the GPU computes every row.
        count, heads, width = 9216, 16, 64
        query, q = bf16_array(rng.normal(0, .2, (1, heads, count, width)))
        key, k = bf16_array(rng.normal(0, .2, (1, heads, count, width)))
        value, v = bf16_array(rng.normal(0, .2, (1, heads, count, width)))
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0., is_causal=False)
        if not bool(torch.isfinite(actual).all()):
            raise AssertionError("full-size fused vision attention produced nonfinite output")
        selected = [0, 305, 3001, count - 1]
        scores = np.einsum("hrd,hnd->hrn", q[0][:, selected, :].astype(np.float64),
                           k[0].astype(np.float64)) / np.sqrt(width)
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        expected = np.einsum("hrn,hnd->hrd", weights, v[0].astype(np.float64))
        observed = actual[0][:, selected, :].float().cpu().numpy()
        # BF16 output has a 1/128 maximum relative ULP. The small absolute
        # allowance handles values around zero; this is not a model tolerance.
        np.testing.assert_allclose(observed, expected, rtol=1 / 128, atol=4e-6)

    def compression_pair_rollback():
        from .compression_state import DeviceCompressionState
        from .state import Operation
        norm, norm_ref = bf16_array(rng.uniform(.8, 1.2, 512))
        scores = rng.normal(0, .3, (5, 1024)).astype(np.float32)
        state = DeviceCompressionState(2, norm, eps=1e-6, context=32, admitted_bytes=1024 * 1024)
        state.reset("a" * 32, 0)
        def operation(number, start, rows):
            return Operation("a" * 32, "b" * 32, 0, number, f"{number:032x}", "c" * 64, start, rows)
        def expected(pair):
            values, gates = pair[:, :512].astype(np.float64), pair[:, 512:].astype(np.float64)
            probability = np.exp(gates - gates.max(axis=0))
            probability /= probability.sum(axis=0)
            latent = (values * probability).sum(axis=0)
            return latent / np.sqrt(np.mean(latent ** 2) + 1e-6) * norm_ref.astype(np.float64)
        try:
            first = operation(1, 0, 1)
            state.begin(first)
            assert state.forward(first, tensor(scores[:1], torch.float32)).values.shape == (0, 512)
            state.finish(first, 1)
            tentative = operation(2, 1, 3)
            state.begin(tentative)
            closed = state.forward(tentative, tensor(scores[1:4], torch.float32))
            assert closed.logical_positions == (0, 1) and closed.token_positions == (1, 3)
            reference = np.stack([expected(scores[:2]), expected(scores[2:4])])
            np.testing.assert_allclose(closed.values.float().cpu().numpy(), reference, rtol=1 / 128, atol=2e-5)
            state.finish(tentative, 2)  # Accept positions 1,2; reject position 3.
            torch.testing.assert_close(state._ring[0, 0], tensor(scores[2], torch.float32), rtol=0, atol=0)
            torch.testing.assert_close(state._ring[0, 1], tensor(scores[1], torch.float32), rtol=0, atol=0)
            retry = operation(3, 3, 1)
            state.begin(retry)
            closed = state.forward(retry, tensor(scores[4:5], torch.float32))
            np.testing.assert_allclose(closed.values[0].float().cpu().numpy(), expected(scores[[2, 4]]),
                                       rtol=1 / 128, atol=2e-5)
            state.finish(retry, 0)
            assert state.position == 3
            torch.testing.assert_close(state._ring[0, 0], tensor(scores[2], torch.float32), rtol=0, atol=0)
            torch.testing.assert_close(state._ring[0, 1], tensor(scores[1], torch.float32), rtol=0, atol=0)
        finally:
            state.close()

    record("BF16 sparse attention: sink, duplicates and empty rows", attention)
    record("BF16 index scores: signed weights and causal visibility", index_scores)
    record("Engram: finite E4M3 codes and UE8M0 byte zero", engram_decode)
    record("BF16 scatter: bit preservation and stale generation refusal", scatter)
    record("CUDA streams: pinned producer, dependency event, actual Engram consumer and retained storage", stream_events)
    record("BF16 row pool: partial acceptance, ring overwrite and abort", row_pool_rollback)
    record("BF16 encoder prefix: preserved bytes, new generations and invalid tail", prefix_pool_retention)
    record("BF16 verification rollback: every accepted length 0..6, overwritten ring bytes and generations", verification_rollback_lengths)
    record("BF16 tiled index: independent slots, global ranking and candidate restriction", tiled_index)
    record("BF16 vision attention: fused 9216 patches, 16 heads, head width 64", vision_attention_shape)
    record("FP32 compressor: odd chunks, exact pair closures, partial acceptance and retry", compression_pair_rollback)
    from .packed_selftest import run_packed_checks
    report["packed_checks"] = run_packed_checks(device)
    report["checks"].append({"name": "actual EXL3 K2/K3/K4/K5/K6/K8 layouts, 5120/2304 projections and independent rotations", "status": "passed"})
    from .graph_selftest import run_graph_checks
    report["checks"].extend(run_graph_checks(device))
    return report


def main():
    print(json.dumps(run_cuda_checks(), indent=2))


if __name__ == "__main__":
    main()
