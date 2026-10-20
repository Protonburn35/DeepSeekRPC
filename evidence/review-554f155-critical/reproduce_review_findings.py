"""CPU reproductions of three review findings; never loads weights or runs CUDA.

Usage: python reproduce_review_findings.py /path/to/DeepSeekRPC/source
The platform override in check_cache only reaches the real cache guard.
The vision check substitutes tensor/asset boundaries to reach the real Python
request-control code; it does not validate vision tensor math.
"""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))
from deepseekrpc.deployment_config import DeploymentConfig
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.protocol import hex_id


def check_run_id(temporary):
    shared = json.loads((ROOT / "config/cluster.example.json").read_text())
    shared["cluster_id"] = "a" * 32
    shared["workers"]["B"]["address"] = "192.168.50.144"
    local = json.loads((ROOT / "config/B.example.json").read_text())
    local["model_root"] = str(temporary / "model")
    local["state_root"] = str(temporary / "state-B")
    (temporary / "cluster.json").write_text(json.dumps(shared))
    path = temporary / "B.json"
    path.write_text(json.dumps(local))
    config = DeploymentConfig.read(path)
    # Exactly the validator used at bootstrap.py:136.
    hex_id(config.run, 64, "protocol run")
    try:
        hex_id(config.run, 32, "worker run")
    except ValidationError as error:
        return {"finding": "R1", "reproduced": True,
                "configuration_run_characters": len(config.run), "error": str(error)}
    raise AssertionError("expected the original loader's 32-character gate to fail")


def check_cache(temporary):
    from deepseekrpc import runtime_identity, runtime_policy
    cache_root = temporary / "worker-state/kernels/triton"
    # This is the exact directory expression used by the first runtime check.
    metadata_root = cache_root.absolute().parent / "vllm"
    runtime_policy.configure_owned_vllm_cache(metadata_root)
    actual_sys = runtime_identity.sys
    # Change only this module's platform observer, not Python's platform or
    # the global sys module. The real guard fails before importing Torch.
    runtime_identity.sys = SimpleNamespace(platform="win32", prefix=sys.prefix)
    try:
        try:
            runtime_identity.require_windows_runtime(0)
        except ValidationError as error:
            assert "cache ownership cannot change" in str(error)
            return {"finding": "R2", "reproduced": True,
                    "explicit_root": str(metadata_root),
                    "implicit_root": str(Path(sys.prefix) / ".deepseekrpc-vllm"),
                    "error": str(error)}
        raise AssertionError("expected the nested no-root identity check to fail")
    finally:
        runtime_identity.sys = actual_sys


def check_first_image():
    path = ROOT / "src/deepseekrpc/vision_runtime.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    definition = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "ImageWindowResolver")
    class TensorBoundary:
        def to(self, *args, **kwargs): return self
        def view(self, *args, **kwargs): return self
        def cpu(self): return self
        def numpy(self): return np.zeros((1, 5120), dtype=np.uint16)
    class AssetsBoundary:
        stored = False
        def pin(self, keys): return nullcontext()
        def info(self, key): return SimpleNamespace(span_tokens=1, vit_h=1, vit_w=1)
        def has_features(self, key): return False
        def original(self, key): return b"fixture"
        def store_features(self, key, data): self.stored = True
    torch_boundary = ModuleType("torch")
    torch_boundary.cuda = SimpleNamespace(device=lambda _: nullcontext(), stream=lambda _: nullcontext())
    torch_boundary.inference_mode = nullcontext
    torch_boundary.zeros = lambda *args, **kwargs: TensorBoundary()
    torch_boundary.bfloat16 = object()
    torch_boundary.uint16 = object()
    from deepseekrpc import metrics
    namespace = {"LoadedStage": object, "ImageAssets": object, "Role": object,
                 "ValidationError": ValidationError, "metrics": metrics, "time": time,
                 "process_image": lambda *args: (TensorBoundary(), TensorBoundary())}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    resolver = object.__new__(namespace["ImageWindowResolver"])
    resolver.assets = AssetsBoundary()
    resolver.stage = SimpleNamespace(identity="fixture", attention=SimpleNamespace(device="cuda:0"),
        stream=object(), cancellation=SimpleNamespace(is_set=lambda: False),
        graph=SimpleNamespace(run_image=lambda *args, **kwargs: TensorBoundary()),
        guard=SimpleNamespace(check=lambda: None))
    history = SimpleNamespace(identity="fixture", digest="fixture-history",
        images=[SimpleNamespace(start=0, length=1, key="fixture-image")])
    window = SimpleNamespace(history_digest="fixture-history", operation=SimpleNamespace(start=0, rows=1))
    missing = object()
    original_torch = sys.modules.get("torch", missing)
    sys.modules["torch"] = torch_boundary
    try:
        try:
            resolver(history, window)
        except NameError as error:
            assert "first" in str(error) and resolver.assets.stored
            return {"finding": "R3", "reproduced": True, "error": str(error),
                    "image_features_stored_before_failure": resolver.assets.stored}
        raise AssertionError("expected first-image timing to raise NameError")
    finally:
        if original_torch is missing: del sys.modules["torch"]
        else: sys.modules["torch"] = original_torch


with tempfile.TemporaryDirectory(prefix="deepseek-review-repro-") as directory:
    temporary = Path(directory)
    results = [check_run_id(temporary), check_cache(temporary), check_first_image()]
print(json.dumps({"source_commit": "554f155dc88376c962074167149a77c8486fcc66",
                  "scope": "CPU control-flow reproductions; no GPU or model execution",
                  "results": results}, indent=2))
