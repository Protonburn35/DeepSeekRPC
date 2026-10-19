from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.storage import ExpertKey, RAMCache, ResidentStore


def key(expert=0):
    return ExpertKey("a" * 64, 18, expert, "bundle", "b" * 64)


def test_resident_store_has_no_inference_load_fallback():
    source = ResidentStore(16)
    source.preload(key(), [b"abc", b"def"], expected_bytes=6)
    source.seal({key()})
    assert bytes(source.get(key())) == b"abcdef"
    assert source.get(key()).readonly
    with pytest.raises(ValidationError, match="forbidden"):
        source.get(key(1))
    with pytest.raises(ValidationError):
        source.preload(key(1), [b"x"], expected_bytes=1)


@pytest.mark.parametrize("chunks,size", [([b"short"], 6), ([b"too long"], 2), ([b""], 1)])
def test_failed_resident_load_never_publishes(chunks, size):
    source = ResidentStore(32)
    with pytest.raises(ValidationError):
        source.preload(key(), chunks, expected_bytes=size)
    assert source.used == 0
    with pytest.raises(ValidationError):
        source.seal({key()})


def test_resident_boot_peak_and_hash_admission():
    with pytest.raises(ValidationError, match="peak"):
        ResidentStore(8).preload(key(), [b"12345"], expected_bytes=5)
    with pytest.raises(ValidationError, match="checksum"):
        ResidentStore(16).preload(key(), [b"abcdef"], expected_bytes=6, sha256="0" * 64)


def test_leases_prevent_eviction_and_generations_change():
    cache = RAMCache(4, 4)
    with cache.lease(key(), nbytes=4, loader=lambda: b"data") as (first, generation):
        with pytest.raises(ValidationError, match="leased"):
            with cache.lease(key(1), nbytes=4, loader=lambda: b"xxxx"):
                pass
        assert bytes(first) == b"data"
    with cache.lease(key(1), nbytes=4, loader=lambda: b"next") as (_, second_generation):
        assert second_generation > generation
    assert cache.used == 4 and cache.evictions == 1


def test_inflight_load_is_deduplicated():
    cache = RAMCache(8, 8)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def loader():
        calls.append(1)
        entered.set()
        assert release.wait(timeout=3)
        return b"data"
    def consumer():
        with cache.lease(key(), nbytes=4, loader=loader) as (data, generation):
            return bytes(data), generation
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(consumer)
        assert entered.wait(timeout=3)
        second = pool.submit(consumer)
        release.set()
        assert first.result(timeout=3) == second.result(timeout=3)
    assert calls == [1] and cache.reads == 1 and cache.hits == 1


def test_short_cache_read_clears_reservations_and_retry_is_exact():
    cache = RAMCache(8, 8)
    with pytest.raises(ValidationError, match="short"):
        with cache.lease(key(), nbytes=4, loader=lambda: b"no"):
            pass
    assert cache.used == cache.staging_used == 0
    with cache.lease(key(), nbytes=4, loader=lambda: b"good") as (data, _):
        assert bytes(data) == b"good"
    assert cache.reads == 1


def test_io_staging_exhaustion_fails_before_new_loader():
    cache = RAMCache(8, 4)
    entered, release = threading.Event(), threading.Event()
    def first():
        def loader():
            entered.set()
            assert release.wait(timeout=3)
            return b"1111"
        with cache.lease(key(), nbytes=4, loader=loader):
            pass
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(first)
        try:
            assert entered.wait(timeout=3)
            with pytest.raises(ValidationError, match="staging"):
                with cache.lease(key(1), nbytes=4, loader=lambda: pytest.fail("loader must not run")):
                    pass
        finally:
            release.set()
        job.result(timeout=3)
