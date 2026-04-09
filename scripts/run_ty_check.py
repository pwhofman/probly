#!/usr/bin/env python3
"""Run ty after ensuring the generated stubs directory exists."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

DEFAULT_ARGS = [
    "check",
    "--verbose",
    "--output-format=full",
]


def main() -> int:
    """Create the stubs directory and delegate to ty."""
    repo_root = Path(__file__).resolve().parents[1]
    (repo_root / "stubs").mkdir(parents=True, exist_ok=True)

    ty_executable = shutil.which("ty")
    if ty_executable is None:
        ty_executable = str(Path(sys.executable).with_name("ty"))

    args = sys.argv[1:] or DEFAULT_ARGS
    try:
        result = subprocess.run(  # noqa: S603
            [ty_executable, *args],
            cwd=repo_root,
            check=False,
        )
    except FileNotFoundError:
        return 127

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
