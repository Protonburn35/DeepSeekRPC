"""Pinned vLLM streaming parser with original 1017 semantic validation."""
from dataclasses import replace

from ._vendor.deepseek_1017 import encoding
from .jsonutil import ValidationError, canonical, integer, loads
from .runtime_policy import configure_exact_exl3

MAX_COMPLETION_BYTES = 64 * 1024 ** 2


def _stream_parser(tokenizer, tools, thinking):
    configure_exact_exl3()
    from vllm.parser.deepseek_v41 import DeepSeekV41Parser, deepseek_v41_config

    class OriginalV41Parser(DeepSeekV41Parser):
        """Retain original strings, types, wrappers and content whitespace."""
        @staticmethod
        def parser_config(thinking=False):
            return replace(deepseek_v41_config(thinking=thinking),
                           strip_trailing_reasoning_whitespace=False,
                           drop_whitespace_only_content_before_tools=False,
                           strip_content_whitespace_with_tools=False)

        def _convert_args(self, raw_args, partial):
            # The stock V4 adapter can unwrap a literal arguments/input field.
            # It is meaningful original tool data and must retain its shape.
            return self.parser_engine_config.arg_converter(raw_args, partial)

        def _fix_arg_types(self, args_json, func_name):
            # The original DSML string attribute determines the value type.
            # A schema must not silently rewrite an emitted string/number.
            return args_json

    return OriginalV41Parser(tokenizer, tools, chat_template_kwargs={"thinking": thinking})


class CompletionParser:
    """One actual parser per completion, with bounded raw/output accumulation.

    Streamed JSON argument deltas come from the pinned parser engine. A complete
    EOS response must agree with the unchanged original parser, including tool
    values/ordering and exact reasoning/content whitespace. Truncated responses
    remain explicit partial streams; no EOS or repaired JSON is fabricated.
    """
    def __init__(self, tokenizer, *, model, messages, tools=None, thinking=True):
        if type(thinking) is not bool:
            raise ValidationError("completion parsing requires an explicit original thinking mode")
        configure_exact_exl3()
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        self.request = ChatCompletionRequest(model=model, messages=messages, tools=tools,
                                             tool_choice="auto" if tools else None, include_reasoning=True)
        self.parser = _stream_parser(tokenizer, self.request.tools, thinking)
        self.thinking = thinking
        self._raw, self._reasoning, self._content = [], [], []
        self._tools, self._bytes, self._closed = {}, 0, False
        self._delimiter_pending, self._delimiter_done = "", False

    def _without_block_newlines(self, text, *, finished):
        if self._delimiter_done:
            return text
        marker = "\n\n<｜DSML｜ calls>"
        text = self._delimiter_pending + text
        position = text.find(marker)
        if position >= 0:
            self._delimiter_done, self._delimiter_pending = True, ""
            # Exactly the original block delimiter's two newlines. Nothing
            # inside a tool parameter is rewritten after this first boundary.
            return text[:position] + text[position + 2:]
        if finished:
            self._delimiter_pending = ""
            return text
        keep = max((size for size in range(1, min(len(marker), len(text) + 1))
                    if text.endswith(marker[:size])), default=0)
        self._delimiter_pending = text[-keep:] if keep else ""
        return text[:-keep] if keep else text

    def feed(self, text, token_ids, *, finished=False):
        if (self._closed or not isinstance(text, str) or type(finished) is not bool or
                not isinstance(token_ids, (tuple, list)) or len(token_ids) > 6):
            raise ValidationError("completion parser requires bounded open output deltas")
        for token in token_ids:
            integer(token, "parsed generated token", 0, 129279)
        self._bytes += len(text.encode("utf-8"))
        if self._bytes > MAX_COMPLETION_BYTES:
            raise ValidationError("completion exceeds the admitted parser byte budget")
        self._raw.append(text)
        filtered = self._without_block_newlines(text, finished=finished)
        delta = self.parser.parse_delta(filtered, list(token_ids), self.request, finished=finished)
        if finished:
            self._closed = True
        result = {}
        if delta is not None and delta.reasoning is not None:
            self._reasoning.append(delta.reasoning)
            result["reasoning_content"] = delta.reasoning
        content = delta.content if delta is not None else None
        if content:
            self._content.append(content)
            result["content"] = content
        if delta is not None and delta.tool_calls:
            calls = []
            for call in delta.tool_calls:
                if not 0 <= call.index < 1024:
                    raise ValidationError("completion exceeds its bounded tool-call count")
                item = call.model_dump(exclude_none=True)
                target = self._tools.setdefault(call.index, {"id": None, "name": "", "argument_chunks": []})
                if call.id is not None:
                    if target["id"] not in {None, call.id}:
                        raise ValidationError("streamed tool-call identity changed")
                    target["id"] = call.id
                if call.function is not None:
                    if call.function.name is not None:
                        if target["name"] and target["name"] != call.function.name:
                            raise ValidationError("streamed tool-call function name changed")
                        target["name"] = call.function.name
                    if call.function.arguments:
                        target["argument_chunks"].append(call.function.arguments)
                calls.append(item)
            result["tool_calls"] = calls
        return result or None

    def message(self, *, require_eos):
        if not self._closed:
            raise ValidationError("completion assembly requires a finished actual parser stream")
        reasoning, content = "".join(self._reasoning), "".join(self._content)
        tools = [{"id": value["id"], "type": "function", "function": {"name": value["name"], "arguments": "".join(value["argument_chunks"])}}
                 for _, value in sorted(self._tools.items())]
        if require_eos:
            try:
                original = encoding.parse_message_from_completion_text("".join(self._raw), "thinking" if self.thinking else "chat")
            except (AssertionError, ValueError, TypeError) as exc:
                raise ValidationError("generated response does not follow the original October completion format") from exc
            if (reasoning, content) != (original["reasoning_content"], original["content"]):
                raise ValidationError("streamed reasoning/content differ from the unchanged original parser")
            if len(tools) != len(original["tool_calls"]):
                raise ValidationError("streamed tool-call count differs from the unchanged original parser")
            for actual, expected in zip(tools, original["tool_calls"]):
                namespace = expected.get("namespace")
                expected_name = (namespace + "::" if namespace else "") + expected["function"]["name"]
                if (actual["function"]["name"] != expected_name or
                        canonical(loads(actual["function"]["arguments"], limit=MAX_COMPLETION_BYTES)) !=
                        canonical(loads(expected["function"]["arguments"], limit=MAX_COMPLETION_BYTES))):
                    raise ValidationError("streamed tool-call values differ from the unchanged original parser")
        result = {"role": "assistant", "content": content or None}
        if reasoning:
            result["reasoning_content"] = reasoning
        if tools:
            result["tool_calls"] = tools
        return result
