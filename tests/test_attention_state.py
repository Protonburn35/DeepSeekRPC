import pytest

from deepseekrpc.attention_state import AttentionWindow
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.state import Operation


def window(phase, extent=64):
    operation = Operation("a" * 32, "b" * 32, 0, 1, "c" * 32, "d" * 64, 7, 3)
    return AttentionWindow(operation, "e" * 64, phase, extent)


def test_full_target_replay_visibility_keeps_encoder_sources_causal():
    target = window("target")
    assert target.visible(20, 1) == (64, 64, 64)
    assert target.visible(14, 2) == (4, 4, 5)
    assert window("encoder").visible(14, 2) == (4, 4, 5)
    assert window("decode", 10).visible(20, 1) == (8, 9, 10)


def test_attention_never_silently_truncates_a_window_or_selects_an_unknown_phase():
    with pytest.raises(ValidationError, match="extent"):
        window("target", 9)
    with pytest.raises(ValidationError, match="phase"):
        window("approximate_tail_replay")


def test_target_prefill_projects_only_the_last_prompt_logit_but_verification_projects_every_row():
    assert window("target", 64).target_head_rows() is None
    assert window("target", 10).target_head_rows() == (2, 3)
    assert window("decode", 10).target_head_rows() == (0, 3)
    assert window("encoder", 10).target_head_rows() is None
