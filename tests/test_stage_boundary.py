import pytest

torch = pytest.importorskip("torch")

from deepseekrpc.attention_state import AttentionWindow
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.stage_graph import StageBoundary
from deepseekrpc.state import Operation


def test_fp32_stage_transfer_preserves_internal_bf16_bits_and_fp32_mix():
    operation = Operation("a" * 32, "b" * 32, 0, 1, "c" * 32, "d" * 64, 0, 2)
    window = AttentionWindow(operation, "e" * 64, "target", 2)
    hidden = torch.linspace(-3.5, 4.25, 2 * 4 * 5120).reshape(2, 4, 5120).to(torch.bfloat16)
    mix = torch.tensor([[.1234567, .8765433, .17, .83], [.11, .29, .31, .69]], dtype=torch.float32)
    wire = StageBoundary(window, hidden.float(), mix)
    restored, carried = wire.internal(window, torch.device("cpu"))
    torch.testing.assert_close(restored.view(torch.int16), hidden.view(torch.int16), rtol=0, atol=0)
    torch.testing.assert_close(carried.view(torch.int32), mix.view(torch.int32), rtol=0, atol=0)
    with pytest.raises(ValidationError, match="FP32"):
        StageBoundary(window, hidden, mix).internal(window, torch.device("cpu"))
    changed = hidden.float()
    changed[0, 0, 0] += .0001
    with pytest.raises(ValidationError, match="preserve"):
        StageBoundary(window, changed, mix).internal(window, torch.device("cpu"))
