#!/usr/bin/env python3
"""Thin wrapper around ``sigx-gen generate`` for method stubs."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

DEFAULT_SRC_ROOT = Path("src")
DEFAULT_INCLUDES = (
    "probly/method/*/_common.py",
    "probly/method/method.py",
    "probly/method/_sigx_transforms.py",
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the wrapper script."""
    parser = argparse.ArgumentParser(description="Generate stubs via sigx-gen")
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT, help="Source root path")
    parser.add_argument("--out-root", type=Path, default=None, help="Stub output root; defaults to --src-root")
    parser.add_argument("--sigx-gen", default="sigx-gen", help="sigx-gen executable")
    parser.add_argument("--check", action="store_true", help="Run in check mode without writing files")
    parser.add_argument("--fail-on-errors", action="store_true", help="Fail when sigx emits error diagnostics")
    parser.add_argument("--no-prune-unplanned", action="store_true", help="Do not delete unplanned .pyi files")
    parser.add_argument("--include", action="append", default=[], help="Additional include glob")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude glob")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    return parser


def build_command(args: argparse.Namespace, *, src_root: Path, out_root: Path) -> list[str]:
    """Build the ``sigx-gen generate`` command from parsed args."""
    sigx_executable = args.sigx_gen
    if shutil.which(sigx_executable) is None:
        fallback = Path(sys.executable).with_name("sigx-gen")
        if fallback.is_file():
            sigx_executable = str(fallback)

    command = [
        sigx_executable,
        "generate",
        "--src-root",
        str(src_root),
        "--out-root",
        str(out_root),
    ]
    if args.check:
        command.append("--check")
    if args.fail_on_errors:
        command.append("--fail-on-errors")
    if not args.no_prune_unplanned:
        command.append("--prune-unplanned")

    for include in (*DEFAULT_INCLUDES, *args.include):
        command.extend(["--include", include])
    for exclude in args.exclude:
        command.extend(["--exclude", exclude])
    return command


def main() -> int:
    """Run stub generation through sigx-gen."""
    args = build_parser().parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_root = (repo_root / args.src_root).resolve()
    out_root = src_root if args.out_root is None else (repo_root / args.out_root).resolve()

    if not src_root.is_dir():
        sys.stderr.write(f"Source root does not exist: {src_root}\n")
        return 2
    out_root.mkdir(parents=True, exist_ok=True)

    command = build_command(args, src_root=src_root, out_root=out_root)
    if args.dry_run:
        sys.stdout.write(f"[dry-run] {' '.join(command)}\n")
        return 0

    try:
        subprocess.run(command, cwd=repo_root, check=True)  # noqa: S603
    except FileNotFoundError:
        sys.stderr.write(f"Executable '{command[0]}' not found.\n")
        return 127
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
