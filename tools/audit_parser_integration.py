#!/usr/bin/env python3
"""Exercise actual installed 1017 streaming parser against original encoding.

No model, native CUDA call, checkpoint payload or dependency stub is used.
"""
import argparse
import gzip
from importlib import metadata
import json
from pathlib import Path
import tempfile

from deepseekrpc._vendor.deepseek_1017 import encoding
from deepseekrpc.completion_parser import CompletionParser
from deepseekrpc.detokenize import StopTracker, TokenBytes
from deepseekrpc.jsonutil import canonical
from deepseekrpc.tokenizer import TokenizerAdapter


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--output", type=Path, required=True)
    args = options.parse_args()
    root = Path(__file__).resolve().parents[1]
    pin = json.loads((root / "provenance/token-map.lock.json").read_bytes())
    report = {"scope": "actual installed Python parser/tokenizer; no model or GPU execution",
              "vllm": metadata.version("vllm"), "checks": [], "GPU_execution": "not run"}
    samples = [
        (False, {"content": "Hello 世界 🌍\n\n"}, None),
        (True, {"content": " Answer.  \n", "reasoning_content": "  reason 中文 🌍  \n"}, None),
        (False, {"content": "\n  ", "tool_calls": [{"type": "function", "function": {"name": "echo", "arguments": {
            "s": "5", "n": 5, "flag": True, "items": [1, "two", None], "object": {"k": "v"}}}}]},
         [{"type": "function", "function": {"name": "echo", "parameters": {"type": "object", "properties": {
             "s": {"type": "integer"}, "n": {"type": "string"}}}}}]),
        (True, {"reasoning_content": "think\n ", "content": "summary\n", "tool_calls": [
            {"type": "function", "function": {"name": "first", "arguments": {"arguments": {"k": 3}}}},
            {"type": "function", "function": {"name": "second", "arguments": {"input": "a\nb \"c\""}}}]},
         [{"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {"k": {"type": "integer"}}}}}
          for name in ("first", "second")]),
        (False, {"content": "", "tool_calls": [{"type": "function", "function": {"name": "echo", "arguments": {
            "text": ("a > b < c 世界 🌍\n" * 100), "empty": "", "number": -1.25e20}}}]}, None),
    ]
    try:
        with tempfile.TemporaryDirectory(prefix="deepseekrpc-parser-") as temporary:
            path = Path(temporary) / "tokenizer.json"
            path.write_bytes(gzip.decompress((root / "tests/fixtures/tokenizer.json.gz").read_bytes()))
            tokenizer = TokenizerAdapter(path, expected_sha256=pin["tokenizer_sha256"])
            for index, (thinking, assistant, tools) in enumerate(samples):
                messages = [{"role": "user", "content": "check parser"}, {"role": "assistant", **assistant}]
                text = encoding.render_message(1, messages, "thinking" if thinking else "chat")
                expected = encoding.parse_message_from_completion_text(text, "thinking" if thinking else "chat")
                tokens = tokenizer.encode(text)
                for chunk in (1, 2, 5, 6):
                    parser = CompletionParser(tokenizer, model="deepseekrpc", messages=messages[:1], tools=tools, thinking=thinking)
                    decoder = StopTracker(TokenBytes(tokenizer))
                    for start in range(0, len(tokens), chunk):
                        ids = tokens[start:start + chunk]
                        delta = decoder.commit(decoder.preview(ids))
                        parser.feed(delta, ids)
                    parser.feed(decoder.finish(), [], finished=True)
                    actual = parser.message(require_eos=True)
                    if (actual.get("reasoning_content", ""), actual.get("content") or "") != (expected["reasoning_content"], expected["content"]):
                        raise AssertionError("actual parser changed original reasoning/content bytes")
                    actual_calls, expected_calls = actual.get("tool_calls", []), expected["tool_calls"]
                    if len(actual_calls) != len(expected_calls):
                        raise AssertionError("actual parser changed original tool count")
                    for got, want in zip(actual_calls, expected_calls):
                        if (got["function"]["name"] != want["function"]["name"] or
                            canonical(json.loads(got["function"]["arguments"])) != canonical(json.loads(want["function"]["arguments"]))):
                            raise AssertionError("actual parser changed original tool argument values/types")
                    report["checks"].append({"fixture": index, "token_chunk": chunk, "status": "passed"})
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
