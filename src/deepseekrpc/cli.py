"""Inspect engineering capabilities and model ownership without loading weights."""
import argparse
import json
from pathlib import Path

from . import __version__
from .jsonutil import ValidationError, loads
from .planner import Geometry, StagePlan


def capabilities():
    return {
        "version": __version__, "status": "incomplete engineering checkpoint",
        "release_candidate": False,
        "implemented": ["bounded mutual TLS control/data framing and RPC", "streamed CUDA tensor transfer source",
                        "complete physical pack inventory and publisher-verified selective loader",
                        "Win32 locked resident preload and shared bounded pinned transfers",
                        "actual owned vLLM graph/bootstrap and whole-stage admission source",
                        "permanent packed and bounded routed EXL3 device tiers",
                        "BF16 attention/index/source/shadow/SWA and exact Engram adapter source",
                        "original tokenizer/encoding/hash history and exact token/image identities",
                        "phase/commit ledger and actual stage execution boundary",
                        "DSpark graph, Markov sampling and exact rejection/RNG arithmetic"],
        "native_build_evidence": {"project_ABI_2": "Windows compile/link/import and both SMs passed",
                                  "full_EXLlama": "Windows compile/link/import and both SMs passed",
                                  "full_vLLM": "in progress; full serving dependency bundle pending"},
        "unqualified_or_missing": ["production service/coordinator integration", "actual EXL3/GPU numerical execution",
                                    "full-model exact CED and every-phase cancellation", "full-size laptop residency",
                                    "original image preprocessing/cache and streaming API/parser", "distributed DSpark publication",
                                    "prefix checkpoint reuse", "three-stage snapshots", "Windows deployment bundle/launchers"],
        "hardware_validation": "not run", "independent_astra_review": "pending",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="DeepSeekRPC development inspection tools; inference is not yet implemented")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities", help="report implemented and missing capabilities truthfully")
    commands.add_parser("selftest-cuda", help="run small native CUDA operation checks on a real GPU")
    plan = commands.add_parser("plan", help="derive ownership and prospective source/shadow bytes from config")
    plan.add_argument("--model-config", type=Path, required=True)
    plan.add_argument("--split", default="18,5,17")
    plan.add_argument("--layer-stage-wire", choices=["fp32"], default="fp32")
    args = parser.parse_args(argv)
    try:
        if args.command == "capabilities":
            result = capabilities()
        elif args.command == "selftest-cuda":
            from .gpu_selftest import run_cuda_checks
            result = run_cuda_checks()
        else:
            config = loads(args.model_config.read_bytes(), limit=128 * 1024)
            split = tuple(int(value) for value in args.split.split(","))
            result = StagePlan(Geometry.from_config(config), split, layer_stage_wire=args.layer_stage_wire).as_dict()
        print(json.dumps(result, indent=2))
        return 0
    except (ValidationError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"DeepSeekRPC: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
