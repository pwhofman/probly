"""Sphinx helper utilities for the probly documentation build."""

from __future__ import annotations

import functools
import importlib
import inspect
import os
from pathlib import Path
import pkgutil
import re
from typing import TYPE_CHECKING

# env.dependencies entries must be _StrPath; plain str crashes _has_doc_changed.
from sphinx.util._pathlib import _StrPath

if TYPE_CHECKING:
    from collections.abc import Callable

    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment

REPO_ROOT_HELPERS = Path(__file__).resolve().parents[2]

# Seed applied to every gallery example; see seed_gallery_rngs.
GALLERY_SEED = 0


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


@functools.cache
def _resolve_attribute_first(qualified_name: str) -> object | None:
    """Resolve a dotted name, preferring an attribute over a same-named module.

    ``_resolve_dotted`` imports the longest importable prefix, so it returns the
    module ``probly.method.dropout`` rather than the function re-exported under
    that name. sphinx-gallery records a backreference under the name an example
    writes, which for ``from probly.method import dropout`` denotes the
    function, so alias detection has to resolve the attribute first.

    Args:
        qualified_name: Dotted name to resolve.

    Returns:
        The resolved object, or None if no part of the name resolves.
    """
    parts = qualified_name.split(".")
    if len(parts) > 1:
        parent = _resolve_attribute_first(".".join(parts[:-1]))
        if parent is not None:
            attribute = getattr(parent, parts[-1], None)
            if attribute is not None:
                return attribute
    try:
        return importlib.import_module(qualified_name)
    except Exception:  # noqa: BLE001  (optional backends may be absent)
        return None


def _is_module_attribute(qualified_name: str) -> bool:
    """Whether *qualified_name* names an attribute of a module, not of a class.

    Only module-level names are alias candidates. Attribute chains through a
    class resolve to the inherited implementation -- every predictor's ``.eval``
    is ``torch.nn.Module.eval``, one object under many names -- which is
    identity, not aliasing, and must not pool their examples.
    """
    parent, _, _ = qualified_name.rpartition(".")
    return bool(parent) and inspect.ismodule(_resolve_attribute_first(parent))


@functools.cache
def _alias_index(package_names: tuple[str, ...] = ("probly", "pytraverse")) -> dict[int, list[str]]:
    """Map each public object to every documented module-level name it has.

    Keyed by ``id``: probly builds several public entry points from one
    factory (flexdispatch registers a fresh function per method), so those
    share ``__module__`` and ``__qualname__`` while being distinct objects that
    must stay distinct. ``_resolve_attribute_first`` is cached, which keeps the
    objects alive and their ids therefore valid for the build.

    Args:
        package_names: Top-level packages to walk.

    Returns:
        A mapping of ``id(object)`` to its sorted dotted names.
    """
    index: dict[int, list[str]] = {}
    for root_name in package_names:
        try:
            root = importlib.import_module(root_name)
        except Exception:  # noqa: BLE001, S112  (an unimportable package documents nothing)
            continue
        module_names = [root_name]
        module_names += [info.name for info in pkgutil.walk_packages(root.__path__, prefix=f"{root_name}.")]
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except Exception:  # noqa: BLE001, S112  (optional backends may be absent)
                continue
            if any(part.startswith("_") for part in module_name.split(".")):
                continue  # a private module documents nothing, so its names are not alias targets
            exported = getattr(module, "__all__", None)
            names = (
                exported if isinstance(exported, (list, tuple)) else [n for n in vars(module) if not n.startswith("_")]
            )
            for name in names:
                obj = getattr(module, name, None)
                if obj is None or not (inspect.isfunction(obj) or inspect.isclass(obj) or inspect.ismodule(obj)):
                    continue
                index.setdefault(id(obj), []).append(f"{module_name}.{name}")
    return {obj_id: sorted(set(names)) for obj_id, names in index.items()}


def merge_backreference_aliases(
    backrefs: dict[str, list],
    package_names: tuple[str, ...] = ("probly", "pytraverse"),
) -> dict[str, list]:
    """Union backreference entries across dotted names denoting the same object.

    probly re-exports the same function from several public namespaces, so
    ``probly.method.dropout`` and ``probly.transformation.dropout`` are one
    object under two names. sphinx-gallery keys backreferences by the name an
    example imports, which splits the examples for one method across two keys:
    a ``minigallery`` on either name then shows only part of them, and a name
    that no example happens to import -- ``probly.method.dropconnect``, whose
    examples all import from ``probly.transformation`` -- shows none at all.

    Each module-level key is resolved to an object, its entries unioned with
    those of every other name for that object, and the result written back
    under all of them, including aliases that were absent from the map. Entries
    are de-duplicated on ``(filename, gallery target dir)``. Names that do not
    resolve, name a class attribute, or resolve to an object of their own are
    left untouched.

    Args:
        backrefs: Parsed ``backreferences_all.json``, mapping a dotted name to
            its list of example entries.
        package_names: Top-level packages to search for aliases.

    Returns:
        A new mapping with alias groups unioned.
    """
    index = _alias_index(package_names)

    groups: dict[int, set[str]] = {}
    for name in sorted(backrefs):
        if not _is_module_attribute(name):
            continue
        obj = _resolve_attribute_first(name)
        if obj is None:
            continue
        groups.setdefault(id(obj), set(index.get(id(obj), []))).add(name)

    merged = {name: list(entries) for name, entries in backrefs.items()}
    for names in groups.values():
        if len(names) < 2:
            continue
        union: list = []
        seen: set[tuple[str, str]] = set()
        for name in sorted(names):
            for entry in backrefs.get(name, []):
                key = (entry[0], entry[2])  # (filename, gallery target dir)
                if key not in seen:
                    seen.add(key)
                    union.append(entry)
        if not union:
            continue
        for name in names:
            merged[name] = list(union)
    return merged


_MINIGALLERY_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+minigallery::(.*)$")


def minigallery_symbol_pages(src_dir: Path) -> dict[str, set[Path]]:
    """Map each symbol named in a ``minigallery`` directive to the pages using it.

    A ``minigallery`` renders from ``backreferences_all.json`` when Sphinx reads
    the page that contains it, so a page only picks up new examples if it is
    re-read. Narrative pages are hand-written and never change on their own,
    which is what this map is for: the caller touches them when the symbols they
    show gained or lost examples. Generated example pages are skipped.

    Args:
        src_dir: Documentation source directory to scan recursively.

    Returns:
        A mapping of symbol name to the set of ``.rst`` paths naming it.
    """
    pages: dict[str, set[Path]] = {}
    for rst in src_dir.rglob("*.rst"):
        if "auto_examples" in rst.parts:
            continue
        try:
            lines = rst.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for index, line in enumerate(lines):
            match = _MINIGALLERY_DIRECTIVE.match(line)
            if match is None:
                continue
            indent, arguments = len(match.group(1)), match.group(2).split()
            for following in lines[index + 1 :]:
                stripped = following.strip()
                if not stripped:
                    continue
                if len(following) - len(following.lstrip()) <= indent:
                    break
                if stripped.startswith(":"):  # a directive option, not an argument
                    continue
                arguments.extend(stripped.split())
            for symbol in arguments:
                pages.setdefault(symbol, set()).add(rst)
    return pages


def _gallery_roots(gallery_conf: dict) -> list[tuple[Path, Path]]:
    """Pair each gallery output directory with the examples directory it is built from."""
    src_dir = Path(gallery_conf["src_dir"])
    return [
        (src_dir / gallery_dir, Path(examples_dir))
        for examples_dir, gallery_dir in zip(gallery_conf["examples_dirs"], gallery_conf["gallery_dirs"], strict=True)
    ]


def _example_source_dir(target_dir: Path, gallery_conf: dict) -> Path | None:
    """Return the examples directory a gallery output directory is generated from."""
    for gallery_root, examples_root in _gallery_roots(gallery_conf):
        if target_dir.is_relative_to(gallery_root):
            return examples_root / target_dir.relative_to(gallery_root)
    return None


def stale_example_backreferences(stale_examples: list[str], gallery_conf: dict) -> dict[str, list]:
    """Rebuild the backreference entries of examples sphinx-gallery skipped.

    sphinx-gallery only writes entries for the examples it executes, so the
    MD5-skipped (stale) ones have to be filled in by the caller. Taking them from
    the previous build's ``backreferences_all.json`` loses any example that was
    missing from that build -- one deleted and then restored unchanged is never
    executed again, so it would stay out of every minigallery until a full
    rebuild. Instead, each entry is derived from what sphinx-gallery persisted
    next to the example's generated page: the ``.codeobj.json`` of the names it
    uses, and the copied source for its intro and title. This mirrors
    ``sphinx_gallery.gen_rst._get_backreferences``.

    Args:
        stale_examples: Generated example paths, as in ``gallery_conf["stale_examples"]``.
        gallery_conf: The sphinx-gallery configuration.

    Returns:
        A mapping of symbol name to the entries of the stale examples using it.
    """
    from sphinx_gallery.gen_rst import extract_intro_and_title  # noqa: PLC0415
    from sphinx_gallery.py_source_parser import split_code_and_text_blocks  # noqa: PLC0415
    from sphinx_gallery.utils import _read_json  # noqa: PLC0415

    doc_module = gallery_conf["doc_module"]
    exclude_regex = gallery_conf["exclude_implicit_doc_regex"]
    prefer_full_module = gallery_conf["prefer_full_module"]
    backrefs: dict[str, list] = {}
    for stale in sorted(stale_examples):
        target = Path(stale)
        codeobj_path = target.with_suffix(".codeobj.json")
        example_dir = _example_source_dir(target.parent, gallery_conf)
        if not codeobj_path.exists() or example_dir is None:
            continue  # sphinx-gallery writes no .codeobj.json when an example uses no names
        _, blocks = split_code_and_text_blocks(str(target))
        intro, title = extract_intro_and_title(str(target), blocks[0].content)
        entry = [target.name, str(example_dir), str(target.parent), intro, title]
        symbols: set[str] = set()
        for cobjs in _read_json(codeobj_path).values():
            for cobj in cobjs:
                full_name = f"{cobj['module']}.{cobj['name']}"
                if not cobj["module"].startswith(doc_module):
                    continue
                if not cobj["is_explicit"] and exclude_regex and exclude_regex.search(full_name):
                    continue
                if any(re.search(pattern, full_name) for pattern in prefer_full_module):
                    symbols.add(full_name)
                else:
                    symbols.add(f"{cobj['module_short']}.{cobj['name']}")
        for symbol in symbols:
            backrefs.setdefault(symbol, []).append(entry)
    return backrefs


_GENERATED_EXAMPLE_SUFFIXES = (".rst", ".py", ".py.md5", ".codeobj.json", ".zip", ".ipynb")


def remove_orphaned_example_pages(gallery_conf: dict) -> list[Path]:
    """Delete generated pages of examples whose source file no longer exists.

    sphinx-gallery never removes what it generated for a deleted or renamed
    example, so its page stays in the source tree and fails strict builds as a
    document not included in any toctree. An example is orphaned when the copy
    of its source in the gallery output has no counterpart in the examples
    directory. Images are left alone, since their names prefix-match other
    examples and nothing references them once the page is gone.

    Args:
        gallery_conf: The sphinx-gallery configuration.

    Returns:
        The deleted files.
    """
    removed: list[Path] = []
    for gallery_root, _ in _gallery_roots(gallery_conf):
        if not gallery_root.is_dir():
            continue
        for copy in gallery_root.rglob("*.py"):
            example_dir = _example_source_dir(copy.parent, gallery_conf)
            if example_dir is None or (example_dir / copy.name).exists():
                continue
            for suffix in _GENERATED_EXAMPLE_SUFFIXES:
                generated = copy.with_name(copy.stem + suffix)
                if generated.exists():
                    generated.unlink()
                    removed.append(generated)
    return removed


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
    every page depending on a site-packages file. Safe to drop: upgrading a
    package the docs import changes the docs build cache key, which forces a
    cold build.

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


def seed_gallery_rngs(gallery_conf: dict, fname: str | None) -> None:  # noqa: ARG001
    """Seed the global random number generators before each gallery example.

    Registered in ``sphinx_gallery_conf["reset_modules"]``, so it runs in the
    worker process that executes the example. Most examples that train a model
    never seed anything, which makes the gallery a lottery: an unlucky
    initialization can send an optimizer to NaN and fail the whole build (the
    heteroscedastic VBLL objective is the known offender). Examples that seed
    themselves are unaffected, since their own call runs after this one.

    Args:
        gallery_conf: The sphinx-gallery configuration (unused).
        fname: The example about to run, or None between directories (unused).
    """
    import random  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    random.seed(GALLERY_SEED)
    np.random.seed(GALLERY_SEED)  # noqa: NPY002, seeds the legacy global RNG the examples use
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # docs can be built without the torch dependency group
        return
    torch.manual_seed(GALLERY_SEED)


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

        # Repo deliberately pinned to the canonical repo (PR #513 review).
        # The ref is "main" for dev/PR builds (cached pages keep the URL they
        # were built with, and dependency tracking re-renders pages whose
        # source changed, so anchors track main closely). Release builds set
        # DOCS_SOURCE_REF to the tag, pinning links to that exact revision.
        base = "https://github.com/pwhofman/probly"
        ref = os.environ.get("DOCS_SOURCE_REF") or "main"
        return f"{base}/blob/{ref}/{relpath}#L{lineno}"

    return linkcode_resolve


def _is_documented(modname: str, modules: dict[str, object]) -> bool:
    """Check whether autosummary generates a page for *modname*.

    Args:
        modname: Dotted name of a module.
        modules: The importable modules found while walking the packages.

    Returns:
        True if the module was imported and every component of its dotted
        name is public; private modules such as ``_common`` get no page.
    """
    return modname in modules and not any(part.startswith("_") for part in modname.split("."))


def _reexported_candidates(modname: str, module: object, modules: dict[str, object]) -> list[tuple[str, str, int]]:
    """Return ``(name, kind, id)`` for each ``__all__`` name imported into *module*.

    Args:
        modname: Dotted name of the module being scanned.
        module: The imported module object.
        modules: All importable modules found while walking the packages, used
            to tell a documented defining module from a private one.

    Returns:
        One entry per public name that the module re-exports rather than
        defines, with *kind* being ``"classes"`` or ``"functions"``.
    """
    found: list[tuple[str, str, int]] = []
    for name in getattr(module, "__all__", ()) or ():
        value = getattr(module, name, None)
        if value is None or inspect.ismodule(value):
            continue
        # Members defined here are already in autosummary's own lists.
        defining_module = getattr(value, "__module__", None)
        if defining_module == modname:
            continue
        # A class defined in a public submodule is documented there under its
        # own ``__module__``; re-exporting it would add a second page and make
        # every short-name cross-reference ambiguous.
        if inspect.isclass(value) and defining_module is not None and _is_documented(defining_module, modules):
            continue
        if inspect.isclass(value):
            found.append((name, "classes", id(value)))
        elif callable(value):
            found.append((name, "functions", id(value)))
    return found


def build_reexported_map(package_names: tuple[str, ...] = ("probly", "pytraverse")) -> dict[str, dict[str, list[str]]]:
    """Map each module to the ``__all__`` names autosummary would otherwise skip.

    probly re-exports its public API from private ``_common`` submodules, so
    names like ``conformal_lac`` reach their package as imported members. With
    ``autosummary_imported_members`` disabled they land in no autosummary list,
    get no API page, and every cross-reference to them silently renders as
    plain text instead of a link.

    Classes need de-duplication: autodoc registers a class under its defining
    ``__module__``, so documenting one from two re-export sites collides on the
    same target ("duplicate object description"). Each class is therefore
    assigned to the deepest module that re-exports it. Functions register under
    the module they are documented from, so the same function re-exported by
    ``probly.method`` and ``probly.transformation`` yields two distinct targets
    and must NOT be de-duplicated -- both namespaces are public and the user
    guide links into either.

    The result is plain data so Sphinx can pickle it into the config cache;
    passing a callable through ``autosummary_context`` disables that cache.

    Args:
        package_names: Top-level packages to walk.

    Returns:
        A mapping of module name to ``{"classes": [...], "functions": [...]}``,
        containing only modules that have such names.
    """
    modules: dict[str, object] = {}
    for root_name in package_names:
        try:
            root = importlib.import_module(root_name)
        except Exception:  # noqa: BLE001, S112  (an unimportable package documents nothing)
            continue
        modules[root_name] = root
        for info in pkgutil.walk_packages(root.__path__, prefix=f"{root_name}."):
            try:
                modules[info.name] = importlib.import_module(info.name)
            except Exception:  # noqa: BLE001, S112  (optional backends may be absent)
                continue

    candidates: list[tuple[str, str, str, int]] = []
    owner: dict[int, str] = {}
    for modname, module in modules.items():
        for name, kind, obj_id in _reexported_candidates(modname, module, modules):
            candidates.append((modname, name, kind, obj_id))
            if kind == "classes":
                current = owner.get(obj_id)
                if current is None or modname.count(".") > current.count("."):
                    owner[obj_id] = modname

    result: dict[str, dict[str, list[str]]] = {}
    for modname, name, kind, obj_id in candidates:
        if kind == "classes" and owner[obj_id] != modname:
            continue
        result.setdefault(modname, {"classes": [], "functions": []})[kind].append(name)
    return {modname: {kind: sorted(names) for kind, names in kinds.items()} for modname, kinds in result.items()}


def build_case_collision_filename_map(reexported: dict[str, dict[str, list[str]]]) -> dict[str, str]:
    """Give a case-colliding function its own stub filename.

    A module may export a class and a factory function whose names differ only
    in case, such as ``Representer`` and ``representer``. Autosummary derives a
    stub filename from the object name, and Sphinx normalizes
    ``autosectionlabel`` labels to lowercase, so the two stubs claim the same
    label and the build fails under ``-W``. A case-insensitive filesystem hides
    this: macOS folds both stubs into a single file, so the clash appears only
    on Linux CI.

    Both pages are wanted, so the function keeps its page under a suffixed
    filename rather than being dropped from the listing. Only re-exported names
    are considered, which is where probly's public API lives.

    Args:
        reexported: The mapping returned by :func:`build_reexported_map`.

    Returns:
        A mapping of object name to stub filename for ``autosummary_filename_map``.
    """
    filename_map: dict[str, str] = {}
    for modname, kinds in reexported.items():
        lowered_classes = {name.lower() for name in kinds["classes"]}
        for name in kinds["functions"]:
            if name.lower() in lowered_classes:
                filename_map[f"{modname}.{name}"] = f"{modname}.{name}_function"
    return filename_map
