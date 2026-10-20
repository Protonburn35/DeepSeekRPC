#!/usr/bin/env python3
"""Exercise the real isolated PowerShell/Python bootstrap on a Windows runner."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    if os.name != "nt":
        raise RuntimeError("this check requires actual native Windows PowerShell and registry semantics")
    root = Path(__file__).resolve().parents[1]
    work = Path(os.environ["RUNNER_TEMP"]) / "deepseekrpc-bootstrap-check"
    if work.exists():
        raise RuntimeError("bootstrap check requires a fresh dedicated runner directory")
    work.mkdir()
    bundle = work / "bundle"
    bundle.mkdir()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    files = {}
    for target, origin in [("Install.ps1", "packaging/Install.ps1"),
            ("provenance/windows-bootstrap.lock.json", "provenance/windows-bootstrap.lock.json"),
            ("provenance/python-windows-download.json", "provenance/python-windows-download.json")]:
        destination = bundle / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / origin, destination)
        with destination.open("rb") as stream:
            files[target] = {"bytes": destination.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    (bundle / "BUNDLE.json").write_text(json.dumps({"format": 1, "kind": "bootstrap-test", "source_commit": source,
                                                "files": files}, indent=2) + "\n", encoding="utf-8")
    import winreg

    def path_values():
        values = []
        for authority, key in ((winreg.HKEY_CURRENT_USER, r"Environment"),
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
            with winreg.OpenKey(authority, key) as handle:
                try:
                    values.append(winreg.QueryValueEx(handle, "Path")[0])
                except FileNotFoundError:
                    values.append(None)
        return values

    before = path_values()
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    command = [str(powershell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
               "-File", str(bundle / "Install.ps1"), "-BundleRoot", str(bundle), "-BootstrapOnly"]
    report = {"scope": "actual native Windows isolated Python bootstrap only; application/native/GPU dependencies not installed",
              "source_commit": source, "checks": [], "GPU_execution": "not run"}
    try:
        for attempt in (1, 2):
            with (work / f"bootstrap-{attempt}.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=work, stdout=stream, stderr=subprocess.STDOUT)
            report["checks"].append({"name": f"bootstrap-{attempt}", "exit_code": result.returncode})
            if result.returncode:
                raise RuntimeError(f"actual Windows bootstrap failed; see bootstrap-{attempt}.log")
            if path_values() != before:
                raise RuntimeError("isolated bootstrap modified a user/machine PATH registry value")
            if (bundle / ".install-complete.json").exists():
                raise RuntimeError("bootstrap-only check falsely marked an application runtime installed")
        python = bundle / "runtime/Scripts/python.exe"
        subprocess.run([str(python), "-c", "import sys; assert sys.version_info[:3] == (3,12,14); print(sys.version)"], check=True)
        report["checks"].append({"name": "repeat-install-and-registry-path-preservation", "status": "passed"})
    finally:
        (work / "BUILD.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
