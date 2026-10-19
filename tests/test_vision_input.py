"""Original processor goldens and real bounded image storage; no model fixture."""
from dataclasses import replace
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from deepseekrpc.image_assets import ImageAssets
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.vision_input import PROCESSOR_SHA256, expand_images, inspect_image, process_image


def original_image(width, height, *, offset=0):
    y, x = np.indices((height, width), dtype=np.uint32)
    pixels = np.stack(((x * 13 + y + offset) % 256, (x + y * 7) % 256,
                       (x * 3 + y * 5) % 256), axis=-1).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize("width,height,grid,digest", [
    (13, 17, (45, 34, 15, 12), "4a5755216a6ff843e7011e7f3312abfb1ad426303d0bafea7230e95d7999deb4"),
    (641, 479, (35, 46, 12, 16), "4519494239db1239240a4ea084f3be1aa9b5c997898930a2f52e2fbc535773ca"),
    (3000, 2000, (75, 112, 25, 38), "ea70fd5fd5c5f3c2fe1d9c79732a5f4b306a8c5d17bf76356c279ff2520ac3eb"),
    (19, 3000, (489, 4, 163, 2), "adaa026dd17fe424f11bb47ab69b6d504927a7daf9d1cd40d450c5b64184854a"),
])
def test_unchanged_original_resize_pad_patch_bf16_goldens(width, height, grid, digest):
    # Frozen using the pinned original processor and Pillow 13, before adding
    # the runtime/cache integration. This checks image preprocessing only.
    from deepseekrpc._vendor.deepseek_1017 import image_processor
    assert hashlib.sha256(Path(image_processor.__file__).read_bytes()).hexdigest() == PROCESSOR_SHA256
    data = original_image(width, height)
    info = inspect_image(data, identity="a" * 64)
    assert (info.vit_h, info.vit_w, info.llm_h, info.llm_w) == grid
    patches, types = process_image(data, info)
    assert hashlib.sha256(patches.view(torch.uint16).numpy().tobytes()).hexdigest() == digest
    assert patches.dtype == torch.bfloat16 and patches.shape == (grid[0] * grid[1], 3, 14, 14)
    assert types.tolist() == [0] + ([1] * grid[3] + [2]) * grid[2] + [3]
    assert info.span_tokens == len(types) <= 1024


def test_images_with_equal_grids_but_changed_content_never_share_history_identity():
    data = original_image(13, 17)
    image = inspect_image(data, identity="a" * 64)
    changed = inspect_image(original_image(13, 17, offset=1), identity="a" * 64)
    assert image.span_tokens == changed.span_tokens and image.key != changed.key
    tokens = [0, 4, 129264, 7, 129264, 8]
    history = expand_images(tokens, [image, changed], identity="a" * 64)
    expected = [0, 4] + [129264] * image.span_tokens + [7] + [129264] * changed.span_tokens + [8]
    actual, mask, spans = history.export_prompt()
    assert actual.tolist() == expected and len(spans) == 2
    assert mask.sum() == 4
    other = expand_images(tokens, [changed, image], identity="a" * 64)
    np.testing.assert_array_equal(actual, other.export_prompt()[0])
    assert history.digest != other.digest
    with pytest.raises(ValidationError, match="placeholders"):
        expand_images(tokens, [image], identity="a" * 64)
    with pytest.raises(ValidationError):
        expand_images(tokens, [image, changed], identity="a" * 64, capacity=32)
    with pytest.raises(ValidationError):
        replace(image, width=14)


def test_bounded_image_feature_tiles_pins_eviction_and_persistent_checksum(tmp_path):
    first, second = original_image(13, 17), original_image(13, 17, offset=1)
    info = inspect_image(first, identity="a" * 64)
    feature_bytes = bytes(info.span_tokens * 5120 * 2)  # A storage fixture, no vision/model result claim.
    budget = len(first) + len(feature_bytes) + 16
    assets = ImageAssets(tmp_path / "images", identity="a" * 64, disk_bytes=budget, ram_bytes=0)
    try:
        assert assets.ingest(first) == info
        with assets.pin([info.key]):
            assets.store_features(info.key, feature_bytes)
            assert assets.read_rows(info.key, 63, 64) == bytes(64 * 5120 * 2)
            assert assets.used <= assets.disk_bytes
            with pytest.raises(ValidationError, match="pinned"):
                assets.ingest(second)
        assets.close()
        assets = ImageAssets(tmp_path / "images", identity="a" * 64, disk_bytes=budget, ram_bytes=0)
        with assets.pin([info.key]):
            assert assets.original(info.key) == first
            assert assets.read_rows(info.key, info.span_tokens - 1, 1) == bytes(10240)
            # The disk read must not depend on the peer choosing a shape/path.
            path = tmp_path / "images" / f"{info.key}.bf16"
            data = bytearray(path.read_bytes())
            data[0] = 1
            path.write_bytes(data)
            with pytest.raises(ValidationError, match="checksum"):
                assets.read_rows(info.key, 0, 1)
        new = assets.ingest(second)
        assert new.key != info.key and assets.used == len(second)
        with pytest.raises(ValidationError, match="no longer available"):
            assets.info(info.key)
        with pytest.raises(ValidationError):
            assets.info("../private")
    finally:
        assets.close()


def test_image_feature_ram_is_bounded_and_rejects_nonfinite_cache_bytes(tmp_path):
    assets = ImageAssets(tmp_path / "images", identity="a" * 64, ram_bytes=3 * 1024 ** 2)
    try:
        info = assets.ingest(original_image(13, 17))
        size = info.span_tokens * 10240
        with pytest.raises(ValidationError, match="nonfinite"):
            assets.store_features(info.key, b"\x80\x7f" * (size // 2))
        assets.store_features(info.key, bytes(size))
        assert assets.read_rows(info.key, 0, 2) == bytes(20480)
        assert assets._ram_used == size <= assets.ram_bytes
    finally:
        assets.close()
