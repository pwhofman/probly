#!/usr/bin/env python3
"""Check backend-word placement in library definitions without importing them."""

from __future__ import annotations

import argparse
import ast
import io
from pathlib import Path
import re
import sys
import tokenize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

RULE = "BKN001"
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "src" / "probly"
BACKEND_WORDS = frozenset({"torch", "jax", "flax", "numpy", "Torch", "Jax", "Flax", "Numpy"})
SEMANTIC_WORDS = frozenset({"from", "to", "is", "has", "supports", "Supports"})
ALLOWED_PREFIX_WORDS = BACKEND_WORDS | SEMANTIC_WORDS
WORD_BOUNDARY = re.compile(r"_+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
NOQA = re.compile(r"#\s*noqa:\s*([A-Z]+\d+\b(?:[\s,]+[A-Z]+\d+\b)*)", re.IGNORECASE)


def misplaced_backend(name: str) -> str | None:
    """Find a backend word in an identifier without an allowed prefix.

    Args:
        name: A definition's identifier, including any leading underscore.

    Returns:
        The first misplaced backend word, or None if the name is permitted.
        Underscores and CamelCase boundaries separate words; digits stay in
        their word. Only the explicitly listed lowercase and title-case
        backend spellings are checked.
    """
    # Remove exactly one privacy marker. Dunder methods are exempted separately.
    unprefixed = name.removeprefix("_")
    words = WORD_BOUNDARY.split(unprefixed)
    if words[0] in ALLOWED_PREFIX_WORDS:
        return None
    return next((word for word in words if word in BACKEND_WORDS), None)


def _ignored_lines(source: str) -> set[int]:
    ignored = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        if any(RULE in re.split(r"[\s,]+", match[1].upper()) for match in NOQA.finditer(token.string)):
            ignored.add(token.start[0])
    return ignored


def check_source(source: str, filename: str) -> list[str]:
    """Check function, class, and explicit type-alias definitions in Python code.

    Args:
        source: Python source to parse. No code is imported or executed.
        filename: Filename to include in diagnostics.

    Returns:
        Diagnostics ordered by source location. An explicit ``# noqa: BKN001``
        on the definition's opening line suppresses that definition only.

    Raises:
        SyntaxError: If the source cannot be parsed.
    """
    tree = ast.parse(source, filename=filename)
    ignored = _ignored_lines(source)
    diagnostics: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name.startswith("__") and name.endswith("__"):
                continue
        elif isinstance(node, ast.ClassDef):
            name = node.name
        elif isinstance(node, ast.TypeAlias):
            name = node.name.id
        else:
            continue

        backend = misplaced_backend(name)
        if backend is not None and node.lineno not in ignored:
            message = f"{RULE} Backend word '{backend}' requires an allowed prefix in '{name}'"
            diagnostics.append((node.lineno, node.col_offset + 1, message))

    return [f"{filename}:{line}:{column}: {message}" for line, column, message in sorted(diagnostics)]


def main(argv: Sequence[str] | None = None) -> int:
    """Check the library by default, or explicitly supplied Python files/directories.

    Args:
        argv: Command-line arguments. Defaults to the process arguments.

    Returns:
        Zero if all files pass, or one for naming, parsing, or file-read errors.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path, default=[DEFAULT_ROOT], help="Python files or directories to check"
    )
    args = parser.parse_args(argv)
    files = sorted({file for path in args.paths for file in (path.rglob("*.py") if path.is_dir() else [path])})
    failed = False
    for path in files:
        try:
            with tokenize.open(path) as source_file:
                diagnostics = check_source(source_file.read(), str(path))
        except SyntaxError as error:
            diagnostics = [f"{path}:{error.lineno or 1}:{error.offset or 1}: SyntaxError: {error.msg}"]
        except (OSError, UnicodeError) as error:
            diagnostics = [f"{path}:1:1: {error}"]
        for diagnostic in diagnostics:
            sys.stderr.write(f"{diagnostic}\n")
        failed |= bool(diagnostics)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
