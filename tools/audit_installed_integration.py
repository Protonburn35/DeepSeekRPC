#!/usr/bin/env python3
"""Audit installed upstream Python packages without CUDA or model downloads.

This deliberately exercises real distributions, entry points and registration.
It does not inject namespace stubs, build native extensions or qualify a GPU.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    report = {"scope": "installed CPU-only empty vLLM Python integration; no CUDA extensions",
              "CUDA_execution": "not run", "native_Windows_build": "not run", "checks": []}

    def record(name, function):
        try:
            value = function()
        except Exception as exc:
            report["checks"].append({"name": name, "status": "failed", "error": repr(exc)})
            raise
        report["checks"].append({"name": name, "status": "passed", "result": value})

    def import_vllm():
        import torch
        import vllm
        if torch.version.cuda is not None or vllm.__version__ != "0.31.0+empty":
            raise RuntimeError("this audit requires the explicitly CPU-only source-audit build")
        return {"version": vllm.__version__, "torch": torch.__version__}

    def plugin_entrypoint():
        entries = [e for e in metadata.entry_points(group="vllm.general_plugins") if e.name == "vllm_exl3"]
        if len(entries) != 1 or entries[0].value != "vllm_exl3.bf16_madv_compat:register":
            raise RuntimeError("installed plugin entry point differs from the pinned source")
        return [f"{e.name}:{e.value}" for e in entries]

    def import_plugin():
        import vllm_exl3.exl3 as exl3
        if not exl3._VLLM_AVAILABLE or not exl3._TORCH_AVAILABLE:
            raise RuntimeError("plugin imported its missing-dependency fallback classes")
        return metadata.version("vllm-exl3")

    def register_plugin():
        entry, = [e for e in metadata.entry_points(group="vllm.general_plugins") if e.name == "vllm_exl3"]
        entry.load()()
        return "registered"

    def owned_adapters():
        from deepseekrpc.runtime_policy import configure_exact_exl3
        configure_exact_exl3()
        from deepseekrpc.vllm_experts import BoundedMoEMethod
        from deepseekrpc.vllm_engram import DiskEngram
        from deepseekrpc.vllm_attention import OwnedBF16Attention
        from vllm.models.deepseek_v41.attention import DeepseekV4Attention
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
        from vllm.models.deepseek_v41.common.engram import Engram
        if not issubclass(BoundedMoEMethod, FusedMoEMethodBase) or DiskEngram.__bases__ != (Engram,):
            raise RuntimeError("owned adapters do not bind the real upstream model classes")
        if not issubclass(OwnedBF16Attention, DeepseekV4Attention) or OwnedBF16Attention.__abstractmethods__:
            raise RuntimeError("owned BF16 attention is not a concrete real upstream subclass")
        return "actual vLLM routed-method, learned Engram and concrete BF16 attention classes; CUDA constructors/execution not run"

    try:
        record("import_vllm", import_vllm)
        record("plugin_entrypoint", plugin_entrypoint)
        record("import_plugin", import_plugin)
        record("register_plugin", register_plugin)
        record("owned_adapter_imports", owned_adapters)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
