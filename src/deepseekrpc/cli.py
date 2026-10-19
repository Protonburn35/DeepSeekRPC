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
        "implemented": ["bounded framing", "mutual TLS transport", "fragmented tensor transfer",
                        "source-derived ownership", "header-only inventory", "transaction journal",
                        "resident-only byte storage", "leased bounded RAM tier",
                        "pinned tokenizer", "transactional official Engram hashes"],
        "unqualified_or_missing": ["Windows/CUDA build", "vLLM stage adapter", "EXL3 CUDA execution",
                                    "BF16 sparse attention/index integration", "exact CED scheduler", "disk Engram execution",
                                    "physical laptop residency", "GPU cache tier", "vision/API integration",
                                    "DSpark verification", "prefix reuse", "three-stage snapshots"],
        "hardware_validation": "not run", "independent_astra_review": "pending",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="DeepSeekRPC development inspection tools; inference is not yet implemented")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities", help="report implemented and missing capabilities truthfully")
    plan = commands.add_parser("plan", help="derive ownership and prospective source/shadow bytes from config")
    plan.add_argument("--model-config", type=Path, required=True)
    plan.add_argument("--split", default="18,5,17")
    args = parser.parse_args(argv)
    try:
        if args.command == "capabilities":
            result = capabilities()
        else:
            config = loads(args.model_config.read_bytes(), limit=128 * 1024)
            split = tuple(int(value) for value in args.split.split(","))
            result = StagePlan(Geometry.from_config(config), split).as_dict()
        print(json.dumps(result, indent=2))
        return 0
    except (ValidationError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Invalid plan input: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
