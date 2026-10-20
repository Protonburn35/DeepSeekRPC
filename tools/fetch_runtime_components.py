#!/usr/bin/env python3
"""Fetch only successful native component evidence from this repository."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--vllm-run", required=True)
    options.add_argument("--exllama-run", required=True)
    options.add_argument("--native-run", required=True)
    options.add_argument("--output", type=Path, required=True)
    args = options.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    observed = {}
    for component, prefix, expected_name in [
        ("vllm", "windows-vllm", "Pinned Windows vLLM native compilation"),
        ("exllama", "windows-exllama", "Pinned Windows EXL3 runtime compilation"),
        ("native", "windows-cuda", "Native Windows CUDA compilation")]:
        identifier = getattr(args, f"{component}_run")
        if not identifier.isdigit():
            raise RuntimeError("component run IDs must be numeric GitHub Actions runs")
        info = json.loads(subprocess.check_output(["gh", "run", "view", identifier, "--repo", "Protonburn35/DeepSeekRPC",
            "--json", "headSha,headBranch,name,status,conclusion"], text=True))
        if (info["status"], info["conclusion"], info["name"], info["headBranch"]) != ("completed", "success", expected_name, "main"):
            raise RuntimeError("runtime assembly requires the expected successful native build on main")
        destination = root / component
        subprocess.run(["gh", "run", "download", identifier, "--repo", "Protonburn35/DeepSeekRPC",
            "--name", f"{prefix}-{info['headSha']}", "--dir", str(destination)], check=True)
        reports = list(destination.rglob("BUILD.json"))
        if len(reports) != 1 or json.loads(reports[0].read_text())["source_commit"] != info["headSha"]:
            raise RuntimeError("component build report does not match its actual workflow source")
        observed[component] = {"run": identifier, **info}
    (root / "RUNS.json").write_text(json.dumps(observed, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(observed, indent=2))


if __name__ == "__main__":
    main()
