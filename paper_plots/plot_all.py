"""Render every paper_plots figure and table in one shot.

Reads only the local caches -- no wandb traffic. Run this after ``fetch_all``
has populated the SP/OOD/AL caches.

Each underlying plot script keeps its own edit-me constants (see the module
docstrings). This driver only overrides the few decisions that benefit from a
single, opinionated default for paper-style outputs:

- ``AL_CURVES_TOP_K``: 5 (the per-module default is 10; for paper figures we
  prefer fewer, more-readable curves).

Tweak any other defaults at the top of the individual plot script
(``ood_hist.BINS``, ``paper_tables.DECIMALS``, etc.) -- they are deliberately
left alone here so a one-off custom plot doesn't have to touch ``plot_all``.

Run::

    uv run python -m paper_plots.plot_all
"""

from __future__ import annotations

import argparse

from paper_plots import (
    al_curves,
    al_table,
    bar_al,
    bar_al_eu_tu,
    bar_ood,
    bar_sp,
    coverage_efficiency,
    ood_hist,
    paper_tables,
    sp_arc,
)

# --- opinionated defaults ----------------------------------------------------
# SP curves render the top-2 + bottom-2 methods by SP Acc-AUC -- highlights
# both the leaders and the laggards in the same panel.
# AL curves stick to a simple top-3 (no bottom selection).
AL_CURVES_TOP_K: int | None = 3
AL_CURVES_BOTTOM_K: int | None = None
SP_ARC_TOP_K: int | None = 2
SP_ARC_BOTTOM_K: int | None = 2
# -----------------------------------------------------------------------------


def main(*, skip_al: bool = False) -> None:
    print("=== bar_sp ===")
    bar_sp.main()
    print("\n=== bar_ood ===")
    bar_ood.main()
    print(f"\n=== sp_arc (top-K={SP_ARC_TOP_K}, bottom-K={SP_ARC_BOTTOM_K}) ===")
    sp_arc.TOP_K = SP_ARC_TOP_K
    sp_arc.BOTTOM_K = SP_ARC_BOTTOM_K
    sp_arc.main()
    print("\n=== ood_hist ===")
    ood_hist.main()
    print("\n=== coverage_efficiency ===")
    coverage_efficiency.main()

    if not skip_al:
        print("\n=== bar_al ===")
        bar_al.main()
        print("\n=== bar_al_eu_tu ===")
        bar_al_eu_tu.main()
        print(f"\n=== al_curves (top-K={AL_CURVES_TOP_K}, bottom-K={AL_CURVES_BOTTOM_K}) ===")
        al_curves.TOP_K = AL_CURVES_TOP_K
        al_curves.BOTTOM_K = AL_CURVES_BOTTOM_K
        al_curves.main()

    print("\n=== paper_tables ===")
    paper_tables.main()
    if not skip_al:
        print("\n=== al_table ===")
        al_table.main()
    print("\nall done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render every paper figure and table from the local caches.")
    parser.add_argument("--skip-al", action="store_true", help="Leave out the active learning figures and table.")
    main(skip_al=parser.parse_args().skip_al)
