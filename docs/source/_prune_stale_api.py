"""Prune generated API pages whose documented object no longer exists.

autosummary never deletes pages for removed or renamed objects, and an
orphaned page fails strict (-W) builds. Run before ``sphinx-build``;
deleting a still-valid page is harmless (autosummary regenerates it).
"""

from __future__ import annotations

import importlib
from pathlib import Path

API_DIR = Path(__file__).resolve().parent / "api"


def resolves(qualified_name: str) -> bool:
    """Check whether a fully qualified name still refers to an importable object.

    Args:
        qualified_name: Dotted name of a module or of an attribute reachable
            from a module, e.g. ``probly.calibrator`` or
            ``probly.conformal_scores.lac.torch.compute_lac_score_torch``.

    Returns:
        True if the name resolves to a module or module attribute.
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
        return True
    return False


def main() -> None:
    """Delete pages in ``docs/source/api`` whose target cannot be resolved.

    The parent module's page is deleted too: its cached autosummary still
    lists the removed object, which would fail to import in strict builds.
    """
    if not API_DIR.is_dir():
        return
    doomed: set[Path] = set()
    for page in sorted(API_DIR.glob("*.rst")):
        if not resolves(page.stem):
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
