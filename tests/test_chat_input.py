import base64
import copy
import gzip
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from deepseekrpc._vendor.deepseek_1017 import encoding
from deepseekrpc.chat_input import prepare_chat
from deepseekrpc.image_assets import ImageAssets
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.tokenizer import TokenizerAdapter


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    root = Path(__file__).parents[1]
    path = tmp_path_factory.mktemp("chat-tokenizer") / "tokenizer.json"
    path.write_bytes(gzip.decompress((root / "tests/fixtures/tokenizer.json.gz").read_bytes()))
    pin = json.loads((root / "provenance/token-map.lock.json").read_bytes())
    return TokenizerAdapter(path, expected_sha256=pin["tokenizer_sha256"])


@pytest.fixture
def assets(tmp_path):
    store = ImageAssets(tmp_path / "images", identity="a" * 64)
    yield store
    store.close()


def test_chat_uses_original_full_template_and_keeps_requested_controls(tokenizer, assets):
    body = {"model": "deepseekrpc", "messages": [{"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Hello 中文 🌍"}], "seed": 1017, "temperature": .7, "top_p": .9,
            "max_tokens": 123, "stop": ["STOP"], "logit_bias": {"1": -.5}, "stream": True,
            "stream_options": {"include_usage": True}, "reasoning_effort": 77, "speculative": False}
    original = copy.deepcopy(body)
    expected = tokenizer.encode(encoding.encode_messages(body["messages"], "thinking", reasoning_effort=77))
    with prepare_chat(body, model="deepseekrpc", tokenizer=tokenizer, assets=assets) as request:
        assert request.history.length == len(expected)
        actual, mask, images = request.history.export_prompt()
        np.testing.assert_array_equal(actual, expected)
        assert mask.all() and not images
        assert (request.settings.seed, request.settings.max_tokens, request.settings.policy.temperature) == (1017, 123, .7)
        assert request.settings.logit_bias == ((1, -.5),) and request.settings.stop == ("STOP",)
        assert request.stream and request.include_usage and not request.settings.speculative
    assert body == original


def test_tool_result_only_turn_retains_reasoning_ids_and_original_result_order(tokenizer, assets):
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
    messages = [{"role": "user", "content": "Look up two values."},
                {"role": "assistant", "content": "", "reasoning_content": "Keep this reasoning.", "tool_calls": [
                    {"id": "first", "type": "function", "function": {"name": "lookup", "arguments": '{"key":"a"}'}},
                    {"id": "second", "type": "function", "function": {"name": "lookup", "arguments": '{"key":"b"}'}}]},
                {"role": "tool", "tool_call_id": "second", "content": "second result"},
                {"role": "tool", "tool_call_id": "first", "content": "first result"}]
    rendered = copy.deepcopy(messages)
    rendered[0]["tools"] = tools
    expected = encoding.encode_messages(rendered, "thinking")
    with prepare_chat({"model": "deepseekrpc", "messages": messages, "tools": tools},
                      model="deepseekrpc", tokenizer=tokenizer, assets=assets) as request:
        assert request.messages == messages and "tools" not in request.messages[0]
        actual = tokenizer.decode(request.history.export_prompt()[0].tolist())
        assert actual == expected and "Keep this reasoning." in actual
        assert actual.index("first result") < actual.index("second result")


def test_inline_images_expand_exact_spans_and_request_pins_release_on_cancel(tokenizer, assets):
    def block(color):
        output = io.BytesIO()
        Image.new("RGB", (64, 64), color).save(output, format="PNG")
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()}}
    body = {"model": "deepseekrpc", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Before"}, block("red"), {"type": "text", "text": "Between"}, block("blue"),
        {"type": "text", "text": "After"}]}], "thinking": {"type": "disabled"}, "max_completion_tokens": 8}
    with prepare_chat(body, model="deepseekrpc", tokenizer=tokenizer, assets=assets) as request:
        _, mask, spans = request.history.export_prompt()
        assert len(spans) == 2 and spans[0].length == spans[1].length and spans[0].key != spans[1].key
        assert int((~mask).sum()) == sum(span.length for span in spans)
        assert set(assets._pins) == {span.key for span in spans}
        assert not request.thinking
    assert not assets._pins


@pytest.mark.parametrize("change", [{"response_format": {"type": "json_object"}}, {"logprobs": True}, {"n": 2},
                                    {"parallel_tool_calls": False}, {"temperature": -1}, {"stream_options": []},
                                    {"logit_bias": []}, {"tool_choice": "required"}, {"seed": True},
                                    {"max_tokens": 1048576}, {"max_tokens": 2, "max_completion_tokens": 3}])
def test_unsupported_or_invalid_requested_controls_fail_before_model_work(tokenizer, assets, change):
    body = {"model": "deepseekrpc", "messages": [{"role": "user", "content": "Hi"}], **change}
    with pytest.raises(ValidationError):
        with prepare_chat(body, model="deepseekrpc", tokenizer=tokenizer, assets=assets):
            pytest.fail("invalid generation controls were admitted")


@pytest.mark.parametrize("block", [{"type": "input_audio", "data": "ignored"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "low"}},
    {"type": "image_url", "image_url": {"url": "file:///private/image.png"}}])
def test_unsupported_content_is_rejected_without_lossy_placeholder_text(tokenizer, assets, block):
    body = {"model": "deepseekrpc", "messages": [{"role": "user", "content": [block]}]}
    with pytest.raises(ValidationError):
        with prepare_chat(body, model="deepseekrpc", tokenizer=tokenizer, assets=assets):
            pytest.fail("unsupported content was admitted")
