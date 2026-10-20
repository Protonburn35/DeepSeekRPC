from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def test_undefined_name_gate_catches_missing_and_wrong_scope_timers(tmp_path):
    spec = spec_from_file_location("undefined_name_gate", Path(__file__).parents[1] / "tools/check_undefined_names.py")
    gate = module_from_spec(spec)
    spec.loader.exec_module(gate)
    good = tmp_path / "good.py"
    good.write_text("import time\ndef encode():\n    first = time.perf_counter_ns()\n    return time.perf_counter_ns() - first\n")
    assert gate.check([good]) == []
    bad = tmp_path / "wrong_scope.py"
    bad.write_text("def reuse():\n    first = 1\ndef encode():\n    return first\n")
    assert len(gate.check([bad])) == 1 and "first" in gate.check([bad])[0]
    bad.write_text("def encode():\n    elapsed = first\n    first = 1\n    return elapsed\n")
    assert len(gate.check([bad])) == 1 and "first" in gate.check([bad])[0]
