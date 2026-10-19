import json
import os
from pathlib import Path
import subprocess
import sys


def test_exact_policy_overrides_int8_defaults_before_import_and_detects_changes():
    code = '''
import json, os
from deepseekrpc.runtime_policy import configure_exact_exl3, assert_exact_exl3
from deepseekrpc.jsonutil import ValidationError
assert configure_exact_exl3() == configure_exact_exl3()
assert_exact_exl3()
print(json.dumps({k:os.environ[k] for k in ['EXL3_INT8_GEMV','EXL3_GR_INT8']}))
os.environ['EXL3_INT8_GEMV']='2'
try:
    assert_exact_exl3()
except ValidationError:
    pass
else:
    raise AssertionError('changed fidelity policy was accepted')
'''
    env = dict(os.environ, EXL3_INT8_GEMV="2", EXL3_GR_INT8="1")
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"EXL3_INT8_GEMV": "0", "EXL3_GR_INT8": "0"}
