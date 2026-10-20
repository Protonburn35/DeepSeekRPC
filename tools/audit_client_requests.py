#!/usr/bin/env python3
"""Prepare actual pinned SDK-produced wire requests with the original renderer.

This CPU source/adapter audit does not launch OpenCode or a GPU model.
"""
import argparse
from contextlib import closing
import gzip
import hashlib
import json
from pathlib import Path

from deepseekrpc.chat_input import prepare_chat
from deepseekrpc.image_assets import ImageAssets
from deepseekrpc.jsonutil import canonical, loads
from deepseekrpc.tokenizer import TokenizerAdapter


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--input", type=Path, required=True)
    args = options.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = args.input.resolve(strict=True)
    with (work / "requests.json").open("rb") as stream:
        requests = loads(stream.read(1024 * 1024 + 1), limit=1024 * 1024)
    if not isinstance(requests, list) or len(requests) != 2:
        raise RuntimeError("the pinned SDK audit must supply exactly its two actual captured requests")
    destination = work / "request-preparation"
    destination.mkdir(exist_ok=False)
    path = destination / "tokenizer.json"
    path.write_bytes(gzip.decompress((root / "tests/fixtures/tokenizer.json.gz").read_bytes()))
    token_map = json.loads((root / "provenance/token-map.lock.json").read_bytes())
    tokenizer = TokenizerAdapter(path, expected_sha256=token_map["tokenizer_sha256"])
    report = {"scope": "actual pinned SDK wire requests through original template/tokenizer/image preparation; no model or OpenCode process",
        "checks": [], "GPU_execution": "not run"}
    with closing(ImageAssets(destination / "images", identity="a" * 64)) as assets:
        for index, body in enumerate(requests):
            before = canonical(body)
            with prepare_chat(body, model="deepseek-v4.1-flash-1017-exl3", tokenizer=tokenizer, assets=assets) as request:
                assert request.stream and request.include_usage and request.settings.max_tokens == 8192
                assert request.settings.seed == 41017 and request.settings.policy.temperature == .7
                assert not request.settings.speculative and request.thinking
                assert request.history.capacity == 1048576 and request.history.length + 8192 <= request.history.capacity
                assert bool(request.history.images) == bool(index)
                if index:
                    assert request.tools and request.tool_choice == "auto"
                    assert any(message.get("reasoning_content") == "retained original reasoning" for message in request.messages)
                assert canonical(body) == before
                report["checks"].append({"name": "actual SDK " + ("image/tool-result/reasoning" if index else "text") + " request",
                    "status": "passed", "prompt_tokens": request.history.length,
                    "context_capacity": request.history.capacity, "requested_output": request.settings.max_tokens,
                    "wire_sha256": hashlib.sha256(before).hexdigest()})
    (work / "preparation.json").write_bytes(canonical(report) + b"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
