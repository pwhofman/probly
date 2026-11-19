"""Run every fetcher in sequence: SP/OOD pipeline, then the AL pipeline.

Refresh semantics:

- SP/OOD/first_order/calibration always re-query wandb and overwrite the
  cached record. Re-running picks up freshly-finished runs and rewrites
  everything else; cheap because the summary scalars come back inside the
  paginated ``api.runs(...)`` call.
- AL is **incremental** by default: ``fetch_al`` skips raw records already on
  disk (each AL run needs its own slow ``run.history()`` call). The merged
  sink (``analyze_al``) is rebuilt from scratch every time -- no need for a
  refresh knob there.

To force a full re-pull of every AL run too, set :data:`REFRESH_AL` below to
``True``, or just run ``rm -rf paper_plots/.cache_al/_raw/`` once.

Order matters within the SP/OOD block:

- ``fetch_ood`` rewrites the same ``(method, seed)`` cache files as
  ``fetch_sp`` for runs that logged both, but with materialized OOD score
  arrays from artifacts. Running OOD second means its richer payload wins.
- ``fetch_first_order`` then merges the ``coverage`` / ``efficiency`` scalars
  into those same cache files for credal methods on cifar10.
- ``fetch_calibration`` writes synthetic method names that don't overlap
  with the others, so it sits at the end harmlessly.

The AL block is independent (separate cache root) and runs last because
``fetch_al`` is the slow one even when incremental.

Run::

    uv run python -m paper_plots.fetch_all
"""

from __future__ import annotations

from paper_plots import (
    analyze_al,
    fetch_al,
    fetch_calibration,
    fetch_first_order,
    fetch_first_order_sweep,
    fetch_ood,
    fetch_sp,
)

# Set to True to force ``fetch_al`` to re-pull every wandb run (slow); the
# default False keeps the AL fetch incremental.
REFRESH_AL: bool = False


def main() -> None:
    print("=== fetch_sp ===")
    fetch_sp.main()
    print("\n=== fetch_ood ===")
    fetch_ood.main()
    print("\n=== fetch_first_order ===")
    fetch_first_order.main()
    print("\n=== fetch_first_order_sweep ===")
    fetch_first_order_sweep.main()
    print("\n=== fetch_calibration ===")
    fetch_calibration.main()

    print("\n=== fetch_al ===")
    fetch_al.main(refresh=REFRESH_AL)
    print("\n=== analyze_al ===")
    analyze_al.main()
    print("\nall done.")


if __name__ == "__main__":
    main()
