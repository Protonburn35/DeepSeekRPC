"""Bounded OpenAI chat validation and unchanged October prompt preparation."""
import base64
from contextlib import contextmanager, ExitStack
import copy
from dataclasses import dataclass
import secrets

from ._vendor.deepseek_1017 import encoding
from .image_assets import ImageAssets
from .jsonutil import ValidationError, canonical, integer
from .request_settings import RequestSettings
from .sampling import SamplingPolicy
from .vision_input import MAX_IMAGE_BYTES, expand_images

_FIELDS = {"model", "messages", "tools", "tool_choice", "temperature", "top_p", "top_k", "min_p",
           "repetition_penalty", "presence_penalty", "frequency_penalty", "seed", "max_tokens",
           "max_completion_tokens", "stop", "stream", "stream_options", "thinking", "reasoning_effort",
           "logit_bias", "n", "parallel_tool_calls", "logprobs", "store"}
_FIELDS.add("speculative")


@dataclass(frozen=True)
class PreparedChat:
    model: str
    history: object
    settings: RequestSettings
    messages: list
    tools: list | None
    thinking: bool
    stream: bool
    include_usage: bool
    tool_choice: str


def _inline_image(record):
    """Accept bounded inline OpenAI/Anthropic images, never a server file path."""
    data = record.get("data")
    source = record.get("source")
    if data is None and isinstance(source, dict):
        data = source.get("data")
    if data is None:
        url = record.get("url")
        if not isinstance(url, str) or not url.startswith("data:image/"):
            raise ValidationError("image input requires an inline image data URL or base64 image data")
        header, separator, data = url.partition(",")
        if not separator or not header.endswith(";base64"):
            raise ValidationError("image data URL requires base64 encoding")
    if not isinstance(data, str) or not 1 <= len(data) <= 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValidationError("encoded image exceeds the admitted original image byte limit")
    try:
        result = base64.b64decode(data, validate=True)
    except (ValueError, UnicodeError):
        raise ValidationError("image input is not valid bounded base64") from None
    if not 1 <= len(result) <= MAX_IMAGE_BYTES:
        raise ValidationError("decoded image exceeds the admitted original image byte limit")
    return result


def _messages(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 65536:
        raise ValidationError("messages must be a bounded nonempty conversation")
    allowed = {"role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name", "content_blocks"}
    for message in value:
        if (not isinstance(message, dict) or set(message) - allowed or
                message.get("role") not in {"system", "user", "assistant", "tool", "latest_reminder"}):
            raise ValidationError("conversation contains unsupported message fields or roles")
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise ValidationError("message content must be original text or supported content blocks")
        blocks = message.get("content_blocks")
        if blocks is not None and content not in (None, ""):
            raise ValidationError("message content and content_blocks cannot both provide content")
        if blocks is not None:
            _blocks(blocks)
        if isinstance(content, list):
            _blocks(content)
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValidationError("reasoning_content must retain its original text")
        calls = message.get("tool_calls")
        if calls is not None and (not isinstance(calls, list) or len(calls) > 1024):
            raise ValidationError("message tool calls exceed the bounded call inventory")
        if calls and message["role"] != "assistant":
            raise ValidationError("only assistant messages may contain tool calls")
        seen = set()
        for call in calls or []:
            if (not isinstance(call, dict) or set(call) - {"id", "type", "function", "namespace"} or
                    call.get("type") != "function" or not isinstance(call.get("id"), str) or not 1 <= len(call["id"]) <= 256 or
                    call["id"] in seen or not isinstance(call.get("function"), dict)):
                raise ValidationError("assistant tool calls require unique bounded IDs and function metadata")
            function = call["function"]
            if (set(function) - {"name", "arguments", "namespace"} or not isinstance(function.get("name"), str) or
                    not 1 <= len(function["name"]) <= 256 or not isinstance(function.get("arguments"), str)):
                raise ValidationError("tool calls require their original function name and JSON argument string")
            seen.add(call["id"])
        if message["role"] == "tool" and (not isinstance(message.get("tool_call_id"), str) or
                                           not 1 <= len(message["tool_call_id"]) <= 256):
            raise ValidationError("tool results require a bounded original tool_call_id")
    return copy.deepcopy(value)


def _blocks(value, *, nested=False):
    if not isinstance(value, list) or len(value) > 16384:
        raise ValidationError("content blocks exceed the bounded message inventory")
    for block in value:
        if not isinstance(block, dict):
            raise ValidationError("content blocks must be structured text, image or tool-result objects")
        kind = block.get("type")
        if kind == "text":
            if set(block) != {"type", "text"} or not isinstance(block["text"], str):
                raise ValidationError("text blocks require their exact original text")
        elif kind == "image_url":
            if set(block) != {"type", "image_url"}:
                raise ValidationError("unsupported image_url content fields")
            image = block["image_url"]
            if isinstance(image, dict):
                if set(image) - {"url", "detail"} or not isinstance(image.get("url"), str) or image.get("detail", "auto") != "auto":
                    raise ValidationError("image_url supports an inline URL with the original automatic grid")
            elif not isinstance(image, str):
                raise ValidationError("image_url must contain an original inline image URL")
        elif kind == "image":
            if set(block) - {"type", "source", "url", "data"} or len(set(block) - {"type"}) != 1:
                raise ValidationError("image blocks require one unambiguous original source")
        elif kind == "tool_result" and not nested:
            if (set(block) != {"type", "tool_use_id", "content"} or not isinstance(block["tool_use_id"], str) or
                    not 1 <= len(block["tool_use_id"]) <= 256):
                raise ValidationError("tool-result blocks require original IDs and content")
            if isinstance(block["content"], list):
                _blocks(block["content"], nested=True)
            elif not isinstance(block["content"], str):
                raise ValidationError("tool-result content must be text or supported text/image blocks")
        else:
            raise ValidationError("unsupported message content block type")


def _booleans(body):
    for name, default in (("stream", False), ("parallel_tool_calls", True), ("logprobs", False), ("store", False),
                          ("speculative", True)):
        if type(body.get(name, default)) is not bool:
            raise ValidationError(f"{name} must be a boolean")
    if body.get("parallel_tool_calls") is False or body.get("logprobs") or body.get("store"):
        raise ValidationError("single-tool constraints, logprobs and stored chat responses are not supported by this request path")
    if body.get("n", 1) != 1 or type(body.get("n", 1)) is not int:
        raise ValidationError("this runtime supports exactly one completion per request")
    options = {} if body.get("stream_options") is None else body["stream_options"]
    if (not isinstance(options, dict) or set(options) - {"include_usage"} or
            type(options.get("include_usage", False)) is not bool):
        raise ValidationError("stream_options supports only boolean include_usage")
    if options and not body.get("stream", False):
        raise ValidationError("stream_options requires streaming")
    return body.get("stream", False), options.get("include_usage", False)


@contextmanager
def prepare_chat(body, *, model, tokenizer, assets: ImageAssets, default_output=8192):
    integer(default_output, "default output limit", 1, 1048576)
    if not isinstance(body, dict) or set(body) - _FIELDS:
        raise ValidationError("chat request contains unsupported generation fields")
    if body.get("model") != model:
        raise ValidationError("requested model is not served by this runtime")
    stream, usage = _booleans(body)
    messages = _messages(body.get("messages"))
    tools = body.get("tools")
    if tools is not None and (not isinstance(tools, list) or len(tools) > 256 or len(canonical(tools)) > 16 * 1024 ** 2):
        raise ValidationError("tool definitions exceed the bounded schema inventory")
    tools = copy.deepcopy(tools) or None
    choice = ("auto" if tools else "none") if body.get("tool_choice") is None else body["tool_choice"]
    if choice not in ("auto", "none"):
        raise ValidationError("this request path currently supports tool_choice auto or none")
    if choice == "auto" and not tools:
        raise ValidationError("automatic tool choice requires tool definitions")
    thinking = body.get("thinking", {"type": "enabled"})
    if not isinstance(thinking, dict) or set(thinking) != {"type"} or thinking["type"] not in ("enabled", "disabled"):
        raise ValidationError("thinking must specify type enabled or disabled")
    thinking = thinking["type"] == "enabled"
    effort = body.get("reasoning_effort")
    if effort is not None and not thinking:
        raise ValidationError("reasoning_effort requires the original thinking mode")
    rendered = copy.deepcopy(messages)
    if choice != "none":
        rendered[0]["tools"] = tools
    try:
        prompt, media = encoding.encode_messages(rendered, "thinking" if thinking else "chat", reasoning_effort=effort,
                                                  return_multi_modal_data=True)
    except (AssertionError, ValueError, TypeError, KeyError, NotImplementedError):
        raise ValidationError("messages/tools do not follow the original October conversation format") from None
    tokens = tokenizer.encode(prompt)
    records = media["images"]
    if len(records) > 4096:
        raise ValidationError("prompt image count exceeds the exact image-span inventory")
    with ExitStack() as leases:
        images = []
        for record in records:
            info = assets.ingest(_inline_image(record))
            leases.enter_context(assets.pin([info.key]))
            images.append(info)
        history = expand_images(tokens, images, identity=assets.identity)
        available = history.capacity - history.length
        if not available:
            raise ValidationError("full context is occupied by input; generation requires space for an output token")
        if "max_tokens" in body and "max_completion_tokens" in body and body["max_tokens"] != body["max_completion_tokens"]:
            raise ValidationError("max_tokens and max_completion_tokens disagree")
        maximum = body.get("max_completion_tokens", body.get("max_tokens", min(default_output, available)))
        integer(maximum, "requested maximum output tokens", 1, available)
        seed = body.get("seed", secrets.randbits(64))
        integer(seed, "request seed", 0, (1 << 64) - 1)
        bias = {} if body.get("logit_bias") is None else body["logit_bias"]
        if not isinstance(bias, dict) or len(bias) > 256:
            raise ValidationError("logit_bias exceeds the bounded token bias inventory")
        pairs = []
        for token, value in bias.items():
            if not isinstance(token, str) or not token.isascii() or not token.isdecimal() or str(int(token)) != token:
                raise ValidationError("logit_bias keys must be canonical vocabulary token ID strings")
            pairs.append((int(token), value))
        stops = body.get("stop", ())
        if stops is None:
            stops = ()
        elif isinstance(stops, str):
            stops = (stops,)
        elif isinstance(stops, list):
            stops = tuple(stops)
        policy = SamplingPolicy(**{name: body[name] for name in SamplingPolicy.__dataclass_fields__ if name in body})
        settings = RequestSettings(seed, maximum, policy, speculative=body.get("speculative", True),
                                   logit_bias=tuple(sorted(pairs)), stop=stops)
        yield PreparedChat(model, history, settings, messages, tools, thinking, stream, usage, choice)
