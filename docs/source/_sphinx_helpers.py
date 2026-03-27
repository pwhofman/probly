"""Sphinx helper utilities for the probly documentation build."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


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
