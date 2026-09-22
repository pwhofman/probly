#!/usr/bin/env python3
"""Check backend naming and import placement without importing library code."""

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

NAMING_RULE = "BKN001"
IMPORT_RULE = "BKN002"
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "src" / "probly"
BACKEND_WORDS = frozenset({"torch", "jax", "flax", "numpy", "Torch", "Jax", "Flax", "Numpy"})
SEMANTIC_WORDS = frozenset({"from", "to", "is", "has", "supports", "Supports"})
ALLOWED_PREFIX_WORDS = BACKEND_WORDS | SEMANTIC_WORDS
IMPORT_PREFIXES = {
    "torch": ("torch", "transformers", "huggingface", "peft"),
    "jax": ("jax", "flax"),
    "flax": ("flax",),
}
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


def _ignored_lines(source: str) -> dict[int, set[str]]:
    ignored: dict[int, set[str]] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        for match in NOQA.finditer(token.string):
            ignored.setdefault(token.start[0], set()).update(re.split(r"[\s,]+", match[1].upper()))
    return ignored


class _BackendImportVisitor(ast.NodeVisitor):
    """Inspect runtime import statements and recognize imported TYPE_CHECKING guards."""

    def __init__(self, filename: str, ignored: dict[int, set[str]]) -> None:
        path = Path(filename)
        self.module = path.parent.name if path.stem == "__init__" else path.stem
        self.prefix = WORD_BOUNDARY.split(self.module.removeprefix("_"))[0]
        self.ignored = ignored
        self.diagnostics: list[tuple[int, int, str]] = []
        self.bindings: dict[str, str] = {}

    def _check_import(self, node: ast.Import | ast.ImportFrom, modules: set[str]) -> None:
        if IMPORT_RULE in self.ignored.get(node.lineno, set()):
            return
        for module in sorted(modules):
            prefixes = IMPORT_PREFIXES.get(module)
            if prefixes is not None and self.prefix not in prefixes:
                message = (
                    f"{IMPORT_RULE} Import of '{module}' requires a module prefixed with "
                    f"{', '.join(prefixes)}; found '{self.module}'"
                )
                self.diagnostics.append((node.lineno, node.col_offset + 1, message))

    def visit_Import(self, node: ast.Import) -> None:
        self._check_import(node, {alias.name.split(".")[0] for alias in node.names})
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self.bindings.pop(name, None)
            if alias.name in {"typing", "typing_extensions"}:
                self.bindings[name] = "typing"

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            self._check_import(node, {node.module.split(".")[0]})
        for alias in node.names:
            name = alias.asname or alias.name
            self.bindings.pop(name, None)
            if node.level == 0 and node.module in {"typing", "typing_extensions"} and alias.name == "TYPE_CHECKING":
                self.bindings[name] = "TYPE_CHECKING"

    def _type_checking_guard(self, node: ast.expr) -> bool | None:
        if isinstance(node, ast.Name) and self.bindings.get(node.id) == "TYPE_CHECKING":
            return True
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and self.bindings.get(node.value.id) == "typing"
            and node.attr == "TYPE_CHECKING"
        ):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            guard = self._type_checking_guard(node.operand)
            return None if guard is None else not guard
        return None

    def visit_If(self, node: ast.If) -> None:
        guard = self._type_checking_guard(node.test)
        if guard is None:
            self.visit(node.test)
            before = self.bindings.copy()
            for statement in node.body:
                self.visit(statement)
            after_body = self.bindings
            self.bindings = before
            for statement in node.orelse:
                self.visit(statement)
            # A conditional import cannot prove a guard's origin on every path.
            self.bindings = {name: value for name, value in after_body.items() if self.bindings.get(name) == value}
        else:
            # TYPE_CHECKING is false at runtime; only its runtime branch is visited.
            for statement in node.orelse if guard else node.body:
                self.visit(statement)

    def _visit_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.bindings.pop(node.name, None)
        outer_bindings = self.bindings
        self.bindings = outer_bindings.copy()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in ast.walk(node.args):
                if isinstance(argument, ast.arg):
                    self.bindings.pop(argument.arg, None)
        self.generic_visit(node)
        self.bindings = outer_bindings

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bindings.pop(node.id, None)


def check_source(source: str, filename: str) -> list[str]:
    """Check definition names and runtime backend imports in Python code.

    Args:
        source: Python source to parse. No code is imported or executed.
        filename: Filename to include in diagnostics.

    Returns:
        Diagnostics ordered by source location. Explicit ``# noqa: BKN001`` or
        ``# noqa: BKN002`` comments on a definition or import's opening line
        suppress only the specified rule on that line.

    Raises:
        SyntaxError: If the source cannot be parsed.
    """
    tree = ast.parse(source, filename=filename)
    ignored = _ignored_lines(source)
    imports = _BackendImportVisitor(filename, ignored)
    imports.visit(tree)
    diagnostics = imports.diagnostics
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
        if backend is not None and NAMING_RULE not in ignored.get(node.lineno, set()):
            message = f"{NAMING_RULE} Backend word '{backend}' requires an allowed prefix in '{name}'"
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
