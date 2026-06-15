"""Sphinx helper utilities for the probly documentation build."""

from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
from typing import TYPE_CHECKING

# env.dependencies entries must be _StrPath; plain str crashes _has_doc_changed.
from sphinx.util._pathlib import _StrPath

if TYPE_CHECKING:
    from collections.abc import Callable

    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment

REPO_ROOT_HELPERS = Path(__file__).resolve().parents[2]


def _resolve_dotted(qualified_name: str) -> object | None:
    """Resolve a dotted name to a module or a module attribute, or None."""
    parts = qualified_name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None
        return obj
    return None


def _defining_file(obj: object) -> Path | None:
    """Return the repository file defining *obj*, or None if not in the repo."""
    module = obj if inspect.ismodule(obj) else inspect.getmodule(obj)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    path = Path(module_file).resolve()
    if "site-packages" in path.parts or not path.is_relative_to(REPO_ROOT_HELPERS):
        return None
    return path


def add_member_source_dependencies(app: Sphinx, env: BuildEnvironment) -> None:
    """Make every API page depend on the files defining its documented objects.

    autodoc records only the documented module's own file, but most public
    objects here live in private submodules and are re-exported, so changes
    there would never outdate the page. Runs on ``env-updated``.

    Args:
        app: The Sphinx application (unused).
        env: The build environment whose dependency map is extended.
    """
    del app
    for docname in env.found_docs:
        if not docname.startswith("api/"):
            continue
        target = _resolve_dotted(docname.removeprefix("api/"))
        if target is None:
            continue
        files = {_defining_file(target)}
        if inspect.ismodule(target):
            for name in dir(target):
                if name.startswith("_"):
                    continue
                try:
                    files.add(_defining_file(getattr(target, name)))
                except Exception:  # noqa: BLE001, S112  (lazy attributes may fail to load)
                    continue
        for path in files:
            if path is not None:
                env.dependencies[docname].add(_StrPath(path))


def scrub_external_dependencies(app: Sphinx, env: BuildEnvironment) -> None:
    """Drop recorded page dependencies on installed packages.

    CI recreates the venv each run with fresh mtimes, which would re-read
    every page depending on a site-packages file. Safe to drop: package
    upgrades change ``uv.lock``, which forces a cold build via the cache key.

    Args:
        app: The Sphinx application (unused).
        env: The build environment whose dependency map is scrubbed.
    """
    del app
    for deps in env.dependencies.values():
        external = set()
        for dep in deps:
            full = Path(os.path.normpath(os.path.join(env.srcdir, os.fspath(dep))))
            if "site-packages" in full.parts or not full.is_relative_to(REPO_ROOT_HELPERS):
                external.add(dep)
        deps.difference_update(external)


def ignore_installed_template_mtimes(app: Sphinx) -> None:
    """Keep cached pages from being rewritten because installed templates look new.

    The HTML builder rewrites pages older than the newest template, and
    theme templates live in site-packages with fresh CI mtimes. The patched
    lookup considers only templates inside this repository.

    Args:
        app: The Sphinx application whose HTML builder is patched.
    """
    templates = getattr(app.builder, "templates", None)
    if templates is None:
        print(f"template mtime patch skipped: no template loader on {type(app.builder).__name__}")  # noqa: T201
        return

    def newest_local_template_mtime() -> float:
        mtimes: list[float] = []
        for template_dir in app.config.templates_path:
            # Extensions may append absolute site-packages paths; skip those.
            base = Path(os.path.normpath(os.path.join(app.srcdir, os.fspath(template_dir))))
            if "site-packages" in base.parts or not base.is_relative_to(REPO_ROOT_HELPERS):
                continue
            mtimes.extend(path.stat().st_mtime for path in base.rglob("*") if path.is_file())
        return max(mtimes, default=0)

    templates.newest_template_mtime = newest_local_template_mtime
    print(  # noqa: T201
        f"ignoring installed template mtimes (loader: {type(templates).__name__}, "
        f"newest local template mtime: {newest_local_template_mtime()})"
    )


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
