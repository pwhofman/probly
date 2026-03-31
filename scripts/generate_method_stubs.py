#!/usr/bin/env python3
"""Generate method stubs via sigx."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

INCLUDES = (
    "probly/method/*/_common.py",
    "probly/method/method.py",
    "probly/method/_sigx_transforms.py",
)
STUB_PATHSPEC = "src/probly/method/*/_common.pyi"


def build_command(src_root: Path, out_root: Path) -> list[str]:
    """Build the sigx generate command."""
    sigx_executable = shutil.which("sigx-gen")
    if sigx_executable is None:
        sigx_executable = str(Path(sys.executable).with_name("sigx-gen"))

    command = [
        sigx_executable,
        "generate",
        "--src-root",
        str(src_root),
        "--out-root",
        str(out_root),
        "--fail-on-errors",
        "--prune-unplanned",
    ]
    for include in INCLUDES:
        command.extend(["--include", include])
    return command


def has_stub_drift(repo_root: Path) -> bool:
    """Return whether generated method stubs differ from git state."""
    git_executable = shutil.which("git")
    if git_executable is None:
        return True
    result = subprocess.run(  # noqa: S603
        [
            git_executable,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            STUB_PATHSPEC,
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def main() -> int:
    """Run stub generation through sigx-gen."""
    args = sys.argv[1:]
    check_mode = False
    if args and args[0] == "check":
        check_mode = True
        args = args[1:]

    # Remaining positional args may be file paths passed by pre-commit.
    # They are intentionally ignored because the hook-level `files` filter
    # determines when this script should run.
    _ = args

    repo_root = Path(__file__).resolve().parents[1]
    src_root = (repo_root / "src").resolve()

    command = build_command(src_root, src_root)

    try:
        subprocess.run(command, cwd=repo_root, check=True)  # noqa: S603
    except FileNotFoundError:
        return 127
    except subprocess.CalledProcessError as exc:
        return exc.returncode

    if check_mode and has_stub_drift(repo_root):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
