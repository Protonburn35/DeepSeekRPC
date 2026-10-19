"""Bounded original October image processing and exact placeholder expansion."""
from dataclasses import asdict, dataclass
import hashlib
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image, __version__ as pillow_version

from .history import ImageSpan, TokenHistory
from .jsonutil import ValidationError, fingerprint, integer
from .protocol import hex_id
from .runtime_messages import fields

PROCESSOR_SHA256 = "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272"
MAX_IMAGE_BYTES = 32 * 1024 ** 2
MAX_IMAGE_PIXELS = 32 * 1024 ** 2
IMAGE_TOKEN = 129264


def _arguments():
    return SimpleNamespace(vision_patch_size=14, vision_downsample_ratio=3,
                           vision_max_n_token=1024, vision_min_pixels=295936,
                           vision_max_wh_ratio=None)


@dataclass(frozen=True)
class ImageInfo:
    key: str
    identity: str
    original_sha256: str
    original_bytes: int
    width: int
    height: int
    vit_h: int
    vit_w: int
    llm_h: int
    llm_w: int
    processor: str = PROCESSOR_SHA256
    pillow: str = "13.0.0"

    def __post_init__(self):
        for name in ("key", "identity", "original_sha256", "processor"):
            hex_id(getattr(self, name), 64, name)
        integer(self.original_bytes, "original image byte count", 1, MAX_IMAGE_BYTES)
        for name in ("width", "height"):
            integer(getattr(self, name), name, 1, 32768)
        for name in ("vit_h", "vit_w", "llm_h", "llm_w"):
            integer(getattr(self, name), name, 1, 4096)
        if (self.width * self.height > MAX_IMAGE_PIXELS or self.span_tokens > 1024 or
                self.vit_h * self.vit_w > 9 * 1024 or self.processor != PROCESSOR_SHA256 or self.pillow != "13.0.0"):
            raise ValidationError("image geometry/processor differs from the admitted October image pipeline")
        body = asdict(self)
        del body["key"]
        if fingerprint(body) != self.key:
            raise ValidationError("image key does not bind original bytes, grid, processor and model/runtime identity")

    @property
    def span_tokens(self):
        return self.llm_h * (self.llm_w + 1) + 2

    @classmethod
    def from_wire(cls, value):
        return cls(**fields(value, cls.__dataclass_fields__))


def inspect_image(data: bytes, *, identity):
    hex_id(identity, 64, "image model/runtime identity")
    if not isinstance(data, bytes) or not 1 <= len(data) <= MAX_IMAGE_BYTES or pillow_version != "13.0.0":
        raise ValidationError("image needs bounded original bytes and the admitted Pillow version")
    from ._vendor.deepseek_1017.image_processor import plan_image_grid
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        if not (1 <= width <= 32768 and 1 <= height <= 32768 and width * height <= MAX_IMAGE_PIXELS):
            raise ValidationError("decoded original image dimensions exceed the admitted input budget")
        lh, lw, pixel_h, pixel_w = plan_image_grid(width, height, _arguments())
        # Force decoding while the bounded source is open; malformed/truncated
        # data cannot enter the request's exact image inventory.
        image.load()
    body = {"identity": identity, "original_sha256": hashlib.sha256(data).hexdigest(), "original_bytes": len(data),
            "width": width, "height": height, "vit_h": pixel_h // 14, "vit_w": pixel_w // 14,
            "llm_h": lh, "llm_w": lw, "processor": PROCESSOR_SHA256, "pillow": pillow_version}
    return ImageInfo(fingerprint(body), **body)


def process_image(data: bytes, info: ImageInfo):
    """Call the unchanged official resize/pad/normalize/BF16 patch transform."""
    from ._vendor.deepseek_1017.image_processor import image_token_types, load_image
    if inspect_image(data, identity=info.identity) != info:
        raise ValidationError("original image differs from its exact prepared grid/identity")
    patches, vh, vw, lh, lw = load_image({"data": data}, _arguments())
    if (vh, vw, lh, lw) != (info.vit_h, info.vit_w, info.llm_h, info.llm_w):
        raise ValidationError("original image processor changed its admitted patch/token grid")
    return patches, image_token_types(lh, lw)


def expand_images(prompt_tokens, images, *, identity, capacity=1048576):
    """Exactly replace each training image ID with its pad-free original span."""
    if (not isinstance(prompt_tokens, (tuple, list)) or not 1 <= len(prompt_tokens) <= capacity or
            not isinstance(images, (tuple, list)) or len(images) > 4096 or
            any(not isinstance(image, ImageInfo) or image.identity != identity for image in images)):
        raise ValidationError("image expansion requires bounded original token and image inventories")
    for token in prompt_tokens:
        integer(token, "rendered prompt token", 0, 129279)
    if prompt_tokens.count(IMAGE_TOKEN) != len(images):
        raise ValidationError("original image placeholders and supplied image count differ")
    length = len(prompt_tokens) + sum(image.span_tokens - 1 for image in images)
    integer(length, "expanded image/text context", 1, capacity)
    tokens, mask, spans = np.empty(length, np.int32), np.ones(length, np.bool_), []
    image_iter, cursor = iter(images), 0
    for token in prompt_tokens:
        if token != IMAGE_TOKEN:
            tokens[cursor] = token
            cursor += 1
        else:
            image = next(image_iter)
            end = cursor + image.span_tokens
            tokens[cursor:end], mask[cursor:end] = IMAGE_TOKEN, False
            spans.append(ImageSpan(cursor, image.span_tokens, image.key))
            cursor = end
    history = TokenHistory(identity, capacity=capacity)
    history.set_prompt(tokens, mask, spans)
    return history
