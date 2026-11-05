"""Variant of :mod:`paper_plots.al_curves` with a single-column legend.

Same curves and styling as ``al_curves``; the differences are:

- ``ncol=1`` legend (vs. 2 in the default).
- top-K matches the canonical paper layout in ``plot_all`` (top 3).
- output filename gains an ``_one_col`` suffix so both variants can sit
  side-by-side under ``paper_artifacts/al_methods/``.

Run::

    uv run python -m paper_plots.al_curves_one_col
"""

from __future__ import annotations

from paper_plots import al_curves

TOP_K: int | None = 3
BOTTOM_K: int | None = None


def main() -> None:
    al_curves.LEGEND_NCOL = 1
    al_curves.FILENAME_SUFFIX = "_one_col"
    al_curves.TOP_K = TOP_K
    al_curves.BOTTOM_K = BOTTOM_K
    al_curves.main()


if __name__ == "__main__":
    main()
