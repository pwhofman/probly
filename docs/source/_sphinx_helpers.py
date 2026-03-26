"""Sphinx helper utilities for the probly documentation build."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from docutils.nodes import Element
    from sphinx.addnodes import pending_xref
    from sphinx.application import Sphinx
    from sphinx.builders import Builder
    from sphinx.domains.python import PythonDomain
    from sphinx.environment import BuildEnvironment


# -- Prevent Sphinx from hyperlinking ambiguous short names ----------------------------------------
# Workaround for https://github.com/sphinx-doc/sphinx/issues/10568
#
# Several bare names in signatures cause "more than one target found for
# cross-reference" warnings and wrong hyperlinks:
#
#   * PEP 695 type parameters (T, S, D, In, Out, …) — Sphinx resolves them
#     to unrelated ``.T`` properties (numpy transpose convention) instead of
#     treating them as type variables.
#   * Common attribute names (``type``) — many classes define these,
#     so Sphinx picks an arbitrary target.
#
# The ``missing-reference`` event cannot help because Sphinx *does* resolve
# the reference (ambiguously); the event only fires for truly unresolved refs.
#
# We monkey-patch ``PythonDomain.resolve_xref`` to short-circuit for these
# names, returning plain unlinked text before the domain ever searches.
_SKIP_XREF_NAMES = frozenset(
    {
        # PEP 695 usual type parameters
        "T",
        "S",
        "C",
        "D",
        "F",
        "V",
        "Q",
        "In",
        "Out",
        # Common attribute names with many targets
        "type",
    }
)


def setup(_app: Sphinx) -> None:
    """Patch the Python domain resolver to skip ambiguous short names."""
    from sphinx.domains.python import PythonDomain  # noqa: PLC0415

    _orig_resolve = PythonDomain.resolve_xref

    def _patched_resolve(
        self: PythonDomain,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        xref_type: str,
        target: str,
        node: pending_xref,
        contnode: Element,
    ) -> Element | None:
        if target in _SKIP_XREF_NAMES:
            return contnode  # plain text, no link, no warning
        return _orig_resolve(self, env, fromdocname, builder, xref_type, target, node, contnode)

    PythonDomain.resolve_xref = _patched_resolve


def make_linkcode_resolve(repo_root: Path) -> Callable[[str, dict[str, str]], str | None]:
    """Return a ``linkcode_resolve`` function bound to *repo_root*.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        A callable suitable for assignment to ``linkcode_resolve`` in ``conf.py``.
    """

    def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
        """Return a URL to the source for the given Python object for Sphinx linkcode.

        Args:
            domain: The domain, e.g. "py".
            info: Info dict containing keys like "module" and "fullname".

        Returns:
            The URL to the source file, or None if it cannot be resolved.
        """
        if domain != "py" or not info.get("module"):
            return None
        try:
            module = importlib.import_module(info["module"])
            obj = module
            for part in info["fullname"].split("."):
                obj = getattr(obj, part)
            fn = inspect.getsourcefile(obj)
            _src, lineno = inspect.getsourcelines(obj)
            if fn is None:
                return None
            relpath = Path(fn).relative_to(repo_root)
        except (ModuleNotFoundError, AttributeError, TypeError, OSError, ValueError):
            return None

        base = "https://github.com/n-teGruppe/probly"
        branch = "sphinx_gallery"
        return f"{base}/blob/{branch}/{relpath}#L{lineno}"

    return linkcode_resolve
