"""Resolve exact original image rows for each bounded stage A graph window."""
from .bootstrap import LoadedStage
from .image_assets import ImageAssets
from .jsonutil import ValidationError
from .planner import Role
from .vision_input import process_image


class ImageWindowResolver:
    def __init__(self, stage: LoadedStage, assets: ImageAssets):
        if (not isinstance(stage, LoadedStage) or stage.role != Role.HOST or stage.closed or
                not isinstance(assets, ImageAssets) or assets.identity != stage.identity):
            raise ValidationError("image row resolver requires the actual owned A graph and matching original assets")
        self.stage, self.assets = stage, assets

    def __call__(self, history, window):
        import torch
        if history.identity != self.stage.identity or window.history_digest != history.digest:
            raise ValidationError("image window differs from its immutable original request history")
        start, rows = window.operation.start, window.operation.rows
        end = start + rows
        spans = [span for span in history.images if span.start < end and span.start + span.length > start]
        if not spans:
            raise ValidationError("current image mask has no original image identity/span")
        with self.assets.pin([span.key for span in spans]), torch.cuda.device(self.stage.attention.device), \
                torch.cuda.stream(self.stage.stream), torch.inference_mode():
            result = torch.zeros((rows, 5120), dtype=torch.bfloat16, device=self.stage.attention.device)
            for span in spans:
                if self.stage.cancellation.is_set():
                    raise InterruptedError("image window cancelled")
                info = self.assets.info(span.key)
                if span.length != info.span_tokens:
                    raise ValidationError("original image span changed its exact processor/grid length")
                if not self.assets.has_features(span.key):
                    original = self.assets.original(span.key)
                    patches, types = process_image(original, info)
                    features = self.stage.graph.run_image(patches.to(self.stage.attention.device), types.to(self.stage.attention.device),
                        vit_h=info.vit_h, vit_w=info.vit_w, cancelled=self.stage.cancellation.is_set)
                    # Only one <=1024-row image is materialized at a time. The
                    # whole prompt's image features remain on bounded storage.
                    data = features.view(torch.uint16).cpu().numpy().astype("<u2", copy=False).tobytes()
                    self.assets.store_features(span.key, data)
                    del features, data, patches, types, original
                left, right = max(start, span.start), min(end, span.start + span.length)
                data = bytearray(self.assets.read_rows(span.key, left - span.start, right - left))
                tile = torch.frombuffer(data, dtype=torch.bfloat16).reshape(right - left, 5120)
                result[left - start:right - start].copy_(tile, non_blocking=False)
            self.stage.guard.check()
            return result
