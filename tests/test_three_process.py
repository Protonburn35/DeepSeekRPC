import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from deepseekrpc.planner import Role


@pytest.mark.integration
def test_three_real_processes_fragmented_tcp_commit_cancel_resume(certificates, tmp_path):
    root = Path(__file__).parents[1]
    script = Path(__file__).with_name("tcp_fixture.py")
    processes = []
    def start(role, downstream=None):
        ready, result = tmp_path / f"{role}.ready", tmp_path / f"{role}.result"
        args = [sys.executable, "-u", str(script), "--role", role,
                "--cert-dir", str(certificates[Role.HOST].ca.parent),
                "--ready", str(ready), "--result", str(result)]
        if downstream is not None:
            args.extend(("--downstream-port", str(downstream)))
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = str(root / "src")
        process = subprocess.Popen(args, cwd=root, env=child_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        processes.append(process)
        if role != "host":
            deadline = time.monotonic() + 10
            while not ready.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise AssertionError(f"{role} failed to become ready: {process.communicate(timeout=1)}")
                time.sleep(0.01)
            return json.loads(ready.read_text())["port"]
    try:
        c_port = start("terminal")
        b_port = start("middle", c_port)
        start("host", b_port)
        for process in reversed(processes):
            stdout, stderr = process.communicate(timeout=25)
            assert process.returncode == 0, (stdout, stderr)
        results = {role.label: json.loads((tmp_path / f"{role.label}.result").read_text()) for role in Role}
        for result in results.values():
            assert result["positions"] == [17, 17, 19]
        for identities in results["host"]["identities"]:
            assert {v["role"] for v in identities} == {"host", "middle", "terminal"}
            assert all(v["loaded_id"] == results[v["role"]]["loaded_id"] for v in identities)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=5)
