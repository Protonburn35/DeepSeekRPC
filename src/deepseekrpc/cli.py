"""Deployment, persistent native workers and truthful engineering diagnostics."""
import argparse
import json
from pathlib import Path

from . import __version__
from .jsonutil import ValidationError, loads
from .planner import Geometry, StagePlan


def capabilities():
    return {
        "version": __version__, "status": "runtime implementation under qualification",
        "release_candidate": False,
        "implemented": ["bounded mutual TLS control/data framing and RPC", "streamed CUDA tensor transfer source",
                        "complete physical pack inventory and publisher-verified selective loader",
                        "Win32 locked resident preload and shared bounded pinned transfers",
                        "actual owned vLLM graph/bootstrap and whole-stage admission source",
                        "permanent packed and bounded routed EXL3 device tiers",
                        "BF16 attention/index/source/shadow/SWA and exact Engram adapter source",
                        "original tokenizer/encoding/hash history and exact token/image identities",
                        "phase/commit ledger and actual stage execution boundary",
                        "persistent stage actor/service and full CED/speculative coordinator source",
                        "OpenAI streaming API, persistent Windows launch path and authenticated mesh recovery source",
                        "CED encoder prefix checkpoints with full exact target replay",
                        "bounded all-stage snapshots, prevalidated restore and paused pending-input continuation source",
                        "all-stage history publication and Stage C RNG finalization",
                        "original image processing and bounded exact BF16 feature storage source",
                        "exact ByteLevel UTF-8 decoding and transactional token-aligned stops",
                        "DSpark graph, Markov sampling and exact rejection/RNG arithmetic",
                        "loaded-cluster qualification and incarnation-bound speculative admission gate",
                        "bounded numeric logs, diagnostics archives and matched actual API benchmark mode",
                        "private Windows installer, credential generation and launchers",
                        "pinned private Triton compiler policy and offline compiler qualification source",
                        "pinned OpenCode SDK text/image/tool/reasoning/cancellation protocol audits"],
        "native_build_evidence": {"project_ABI_2": "Windows compile/link/import and both SMs passed",
                                  "full_EXLlama": "Windows compile/link/import and both SMs passed",
                                  "private_Triton_compilers": "Windows standalone Python/TinyCC extension build/load and offline SM89/SM120 cubins passed; no GPU execution",
                                  "full_vLLM": "in progress; full serving dependency bundle pending"},
        "unqualified_or_missing": ["actual EXL3/GPU numerical execution",
                                    "full-model exact CED and every-phase cancellation", "full-size laptop residency",
                                    "actual full-model streaming API and image/GPU execution", "actual distributed DSpark execution",
                                    "actual GPU prefix/snapshot/continuation qualification", "complete Windows runtime installation qualification"],
        "hardware_validation": "not run", "independent_astra_review": "pending",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="DeepSeekRPC native Windows three-stage runtime and diagnostics")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities", help="report implemented and missing capabilities truthfully")
    commands.add_parser("selftest-cuda", help="run small native CUDA operation checks on a real GPU")
    cluster = commands.add_parser("selftest-cluster", help="probe the already loaded cluster and qualify DSpark on its current mesh")
    cluster.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    cluster.add_argument("--api-key-file", type=Path)
    diagnostics = commands.add_parser("diagnostics", help="write a bounded content-free diagnostic archive from live API or saved local metrics")
    source = diagnostics.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="collect saved local A/B/C metrics without loading a model")
    source.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    diagnostics.add_argument("--api-key-file", type=Path)
    diagnostics.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser("benchmark", help="measure matched requests on the already loaded actual private API")
    benchmark.add_argument("--request-file", type=Path, required=True)
    benchmark.add_argument("--repeat", type=int, default=3)
    benchmark.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    benchmark.add_argument("--api-key-file", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    credentials = commands.add_parser("init-credentials", help="generate a new private deployment CA and role/API keys locally")
    credentials.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("serve", help="load the real owned Windows CUDA stage once and keep it resident")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--stage", choices=["A", "B", "C"], help="refuse a role configuration passed to the wrong launcher")
    worker.add_argument("--log-level", choices=["error", "warn", "info", "debug", "trace"], default="info")
    worker.add_argument("--trace", choices=["rpc", "experts", "cache", "io", "kv", "cuda", "vision", "spec"], action="append", default=[])
    mode = worker.add_mutually_exclusive_group()
    mode.add_argument("--print-effective-config", action="store_true")
    mode.add_argument("--preflight", action="store_true", help="check local metadata/TLS without opening listeners or constructing a model")
    mode.add_argument("--self-test", action="store_true", help="run local metadata/runtime/native GPU diagnostics without loading weights")
    snapshots = commands.add_parser("snapshots", help="manual bounded host-owned state administration; no model reload")
    snapshots.add_argument("action", choices=["list", "save", "restore", "delete"])
    snapshots.add_argument("--id", dest="identifier")
    snapshots.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    snapshots.add_argument("--api-key-file", type=Path)
    session = commands.add_parser("session", help="inspect, pause or continue the current exact token/RNG state")
    session.add_argument("action", choices=["status", "pause", "resume"])
    session.add_argument("--id", dest="identifier")
    session.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    session.add_argument("--api-key-file", type=Path)
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
        elif args.command == "selftest-cluster":
            from .admin_client import AdminClient
            result = AdminClient(args.base_url, api_key_file=args.api_key_file).request("POST", "/diagnostics/cluster")
        elif args.command == "diagnostics":
            from .diagnostics import offline_report, write_bundle
            if args.config is not None:
                from .deployment_config import DeploymentConfig
                config = DeploymentConfig.read(args.config)
                report, logs = offline_report(config), config.state_root / "diagnostics"
            else:
                from .admin_client import AdminClient
                report = AdminClient(args.base_url, api_key_file=args.api_key_file).request("GET", "/diagnostics")
                logs = None
            result = write_bundle(report, args.output, logs=logs)
        elif args.command == "benchmark":
            from .admin_client import AdminClient
            from .benchmark import run_benchmark
            from .jsonutil import canonical
            if args.output.exists() or args.output.is_symlink():
                raise ValidationError("benchmark output must be a new file")
            with args.request_file.open("rb") as stream:
                body = loads(stream.read(64 * 1024 ** 2 + 1), limit=64 * 1024 ** 2)
            result = run_benchmark(AdminClient(args.base_url, api_key_file=args.api_key_file), body, repeat=args.repeat)
            with args.output.open("xb") as stream:
                stream.write(canonical(result) + b"\n")
        elif args.command == "init-credentials":
            from .pki import initialize_credentials
            result = initialize_credentials(args.output)
        elif args.command in {"snapshots", "session"}:
            from .admin_client import AdminClient
            from .protocol import hex_id
            client = AdminClient(args.base_url, api_key_file=args.api_key_file)
            if args.command == "snapshots":
                if args.action in {"restore", "delete"}:
                    hex_id(args.identifier, 32, "opaque snapshot")
                elif args.identifier is not None:
                    raise ValidationError("snapshot list/save do not take an existing snapshot ID")
                suffix = "/snapshots" if args.action in {"list", "save"} else f"/snapshots/{args.identifier}" + ("/restore" if args.action == "restore" else "")
                result = client.request({"list": "GET", "save": "POST", "restore": "POST", "delete": "DELETE"}[args.action], suffix)
            elif args.action == "resume":
                for event in client.resume(args.identifier):
                    print(json.dumps(event, ensure_ascii=True), flush=True)
                return 0
            else:
                if args.identifier is not None:
                    raise ValidationError("session status/pause select the currently active session")
                result = client.request("GET" if args.action == "status" else "POST", "/session" if args.action == "status" else "/sessions/pause")
        elif args.command == "serve":
            from .deployment_config import DeploymentConfig
            from .worker import PersistentWorker, preflight
            config = DeploymentConfig.read(args.config)
            if args.stage is not None and "ABC"[int(config.role)] != args.stage:
                raise ValidationError("this launcher requires the matching A/B/C role configuration")
            if args.print_effective_config:
                result = config.effective()
            elif args.preflight or args.self_test:
                result = preflight(config, device_checks=args.self_test)
            else:
                return PersistentWorker(config, log_level=args.log_level, traces=args.trace).run()
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
