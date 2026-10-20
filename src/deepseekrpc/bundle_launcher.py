"""Refuse incomplete/mixed bundles before entering the actual CLI worker."""
import argparse
from pathlib import Path
import sys

from .jsonutil import ValidationError, loads


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage", choices=["A", "B", "C"], required=True)
    options, arguments = parser.parse_known_args()
    try:
        root = options.bundle.resolve(strict=True)
        def read(name):
            with (root / name).open("rb") as stream:
                return loads(stream.read(1024 * 1024 + 1), limit=1024 * 1024)
        installed, manifest = read(".install-complete.json"), read("BUNDLE.json")
        if (installed.get("format") != 1 or manifest.get("kind") != "runtime" or
                installed.get("source_commit") != manifest.get("source_commit") or
                installed.get("python") != "3.12.14" or sys.version_info[:3] != (3, 12, 14) or
                not Path(sys.prefix).resolve().samefile(root / "runtime")):
            raise ValidationError("completed installation does not match this bundle and interpreter")
        from .runtime_identity import application_source_identity
        if application_source_identity() != manifest.get("application_source_identity"):
            raise ValidationError("installed application source differs from the completed bundle")
    except (OSError, ValidationError, ValueError, TypeError, KeyError):
        parser.exit(2, "DeepSeekRPC: run Install.ps1 successfully in this dedicated bundle directory first.\n")
    from .cli import main as cli_main
    return cli_main(["serve", "--stage", options.stage, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
