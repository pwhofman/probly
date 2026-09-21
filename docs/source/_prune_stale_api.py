"""Prune generated API pages whose documented object autosummary would no longer list.

autosummary never deletes pages for removed or renamed objects, and an
orphaned page fails strict (-W) builds. A page is also orphaned when its
name now refers to an object imported from another module, because
``autosummary_imported_members`` is off. Run before ``sphinx-build``;
deleting a still-valid page is harmless (autosummary regenerates it).
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

API_DIR = Path(__file__).resolve().parent / "api"


def is_listed(qualified_name: str) -> bool:
    """Check whether autosummary would still generate a page for a fully qualified name.

    Args:
        qualified_name: Dotted name of a module or of an attribute reachable
            from a module, e.g. ``probly.calibrator`` or
            ``probly.conformal_scores.lac.torch.torch_compute_lac_score``.

    Returns:
        True if the name resolves to a module, or to an attribute defined in
        the module the page belongs to. Attributes imported from another
        module are skipped by autosummary, so their pages count as stale.
    """
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
            return False
        if inspect.ismodule(obj):
            return True
        defined_in = getattr(obj, "__module__", None)
        return defined_in is None or defined_in == qualified_name.rpartition(".")[0]
    return False


def main() -> None:
    """Delete pages in ``docs/source/api`` that autosummary would no longer list.

    The parent module's page is deleted too: its cached autosummary still
    lists the removed object, which would fail to import in strict builds.
    """
    if not API_DIR.is_dir():
        return
    doomed: set[Path] = set()
    for page in sorted(API_DIR.glob("*.rst")):
        if not is_listed(page.stem):
            doomed.add(page)
            parent = API_DIR / (page.stem.rpartition(".")[0] + ".rst")
            if parent.is_file():
                doomed.add(parent)
    for page in sorted(doomed):
        page.unlink()
        print(f"Pruned stale API page: {page.name}")  # noqa: T201
    print(f"Pruned {len(doomed)} stale API page(s).")  # noqa: T201


if __name__ == "__main__":
    main()
