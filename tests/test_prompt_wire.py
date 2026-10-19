from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import pytest

from deepseekrpc.history import ImageSpan, TokenHistory
from deepseekrpc.jsonutil import ValidationError, canonical, loads
from deepseekrpc.planner import Role
from deepseekrpc.prompt_wire import PromptSpec, receive_prompt, send_prompt
from deepseekrpc.request_settings import RequestSettings
from deepseekrpc.runtime_messages import input_from, output_from, receipt_from, window_from, wire
from deepseekrpc.execution import StepInput
from deepseekrpc.publication import OutputPlan, OutputReceipt
from deepseekrpc.sampling import SamplingPolicy
from deepseekrpc.transport import accept, connect, listen


def test_real_tls_transfers_full_million_token_prompt_in_bounded_fragments(certificates):
    history = TokenHistory("a" * 64)
    tokens = np.arange(1048576, dtype=np.int32) % 129264
    tokens[64:72] = [129264, 129265, 129266, 129267, 129268, 129265, 129266, 129268]
    mask = np.ones(len(tokens), dtype=np.bool_)
    mask[64:72] = False
    history.set_prompt(tokens, mask, (ImageSpan(64, 8, "b" * 64),))
    spec = PromptSpec.from_history(history, session="c" * 32, request="d" * 32, epoch=3, transaction="e" * 32)
    spec = PromptSpec.from_wire(loads(canonical(asdict(spec)), limit=4096))
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        def server():
            with accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                        run="f" * 64, plane="data", timeout=5) as connection:
                received = receive_prompt(connection, spec, identity=history.identity)
                assert received.digest == history.digest
                actual, actual_mask, actual_images = received.export_prompt()
                np.testing.assert_array_equal(actual, tokens)
                np.testing.assert_array_equal(actual_mask, mask)
                assert actual_images == history.images
                return connection.bytes_received
        result = pool.submit(server)
        with connect(listener.getsockname(), certificates[Role.HOST], local=Role.HOST, peer=Role.MIDDLE,
                     run="f" * 64, plane="data", timeout=5) as connection:
            send_prompt(connection, spec, history)
            assert result.result(timeout=10) > 5 * 1048576


def test_strict_runtime_metadata_roundtrips_without_losing_identity_or_adding_fields():
    def json_value(value):
        return loads(canonical(value), limit=16384)
    inputs = StepInput((2, 3), (True, True))
    window = inputs.make_window(session="a" * 32, request="b" * 32, epoch=2, number=3,
                                transaction="c" * 32, start=4, phase="decode", history_digest="d" * 64, encoder_tokens=6)
    assert input_from(json_value(wire(inputs))) == inputs
    assert window_from(json_value(wire(window))) == window
    plan = OutputPlan("e" * 32, "a" * 32, "b" * 32, 2, 5, "d" * 64, "f" * 64,
                      "1" * 64, (7, 8), 8, False, None)
    assert output_from(json_value(wire(plan))) == plan
    receipt = OutputReceipt(Role.TERMINAL, plan.digest, "committed", plan.history_digest, 5, 8)
    assert receipt_from(json_value(wire(receipt)), output=True) == receipt
    with pytest.raises(ValidationError):
        receipt_from({**wire(receipt), "role": True}, output=True)
    with pytest.raises(ValidationError):
        window_from({**wire(window), "unknown": 1})
    settings = RequestSettings(15, 100, SamplingPolicy(top_p=.9), logit_bias=((2, 5.), (100, -3.)))
    assert RequestSettings.from_wire(json_value(settings.as_dict())) == settings
    assert settings.digest == RequestSettings.from_wire(json_value(settings.as_dict())).digest
    with pytest.raises(ValidationError):
        replace(settings, logit_bias=((100, 1.), (2, 1.)))
    with pytest.raises(ValidationError):
        replace(settings, logit_bias=((2, float("nan")),))


def test_prompt_spec_rejects_peer_sized_unbounded_allocations():
    history = TokenHistory("a" * 64)
    history.set_prompt([1, 2, 3])
    spec = PromptSpec.from_history(history, session="b" * 32, request="c" * 32, epoch=1, transaction="d" * 32)
    for change in ({"tokens": 1048577}, {"image_count": 4097}, {"image_bytes": 524289}, {"epoch": True}):
        with pytest.raises(ValidationError):
            replace(spec, **change)
