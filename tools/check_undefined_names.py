#!/usr/bin/env python3
"""Reject undefined Python names without importing runtime or model code."""
import ast
from pathlib import Path
import sys

from pyflakes.checker import Checker
from pyflakes.messages import UndefinedExport, UndefinedLocal, UndefinedName


def check(paths):
    failures = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            failures.append(f"{path}:{error.lineno}: {error.msg}")
            continue
        checker = Checker(tree, filename=str(path))
        failures.extend(str(message) for message in checker.messages
                        if isinstance(message, (UndefinedName, UndefinedLocal, UndefinedExport)))
    return failures


def main():
    root = Path(__file__).resolve().parents[1]
    paths = sorted(path for directory in ("src", "tools", "tests")
                   for path in (root / directory).rglob("*.py"))
    failures = check(paths)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Checked {len(paths)} Python files for undefined names, locals and exports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
