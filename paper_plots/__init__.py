"""Lean per-paper fetch + plot pipeline.

Fetchers pull wandb run summaries (and AL history) into local caches under
``paper_plots/.cache{,_al,_sweep}/``. Plot/table scripts read only from
those caches. See ``fetch_all`` and ``plot_all`` for the canonical top-level
entry points.
"""
