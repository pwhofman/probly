"""NAUC aggregation from the merged AL cache.

Provides the per-(dataset, notion) UQ-method ranking used by the AL bar plots,
plus the per-method ``random`` / ``margin`` markers that sit alongside each bar
to ground how much of the gain comes from the UQ signal versus the model
architecture, and the three ``base`` baselines drawn as horizontal reference
lines.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from paper_plots import al_cache
from paper_plots.metadata import EXCLUDED_METHODS, method_label
from paper_plots.targets import ENTITY

SINK = "al_merged"
DEFAULT_LOSS = "cross_entropy"
DEFAULT_CAL = "none"

# The UQ methods. Order doesn't matter (bars are sorted
# descending by mean NAUC at render time), but we use it to iterate the cache.
UQ_METHODS: tuple[str, ...] = (
    "dropout",
    "dropconnect",
    "bayesian",
    "dare",
    "evidential_classification",
    "posterior_network",
    "credal_relative_likelihood",
    "ddu",
    "laplace",
    "het_net",
    "duq",
    "credal_net",
    "credal_wrapper",
    "efficient_credal_prediction",
    "natural_posterior_network",
    "ensemble",
    "credal_ensembling",
    "batchensemble",
    "subensemble",
    "vbll",
    "sngp",
    "credal_bnn",
)

# (dataset, method) pairs that must not borrow another notion. CreEns can do TU and EU, but those
# runs never happened on OpenML 6, so its AU run must not pass as them. Only methods that
# cannot expose a notion at all should fall back.
NO_FALLBACK: frozenset[tuple[str, str]] = frozenset({("openml_6", "credal_ensembling")})

# A cell with fewer seeds is not a result yet, e.g. SNGP on OpenML 6 has a single run.
MIN_SEEDS = 3

# Notion read for a column when a method has no run for the column's own notion, in order
# (same preference as scripts/al_rebuttal_table.py).
COLUMN_FALLBACKS: dict[str, tuple[str, ...]] = {
    "total": ("total", "aleatoric", "epistemic"),
    "epistemic": ("epistemic", "aleatoric", "total"),
}

# Baseline variants drawn as reference lines / curves / table rows on AL plots.
# Each variant pins a (strategy, loss, calibration) tuple. ``least_confident``
# is intentionally excluded -- it crowds the figures and behaves nearly
# identically to ``margin``. Calibration / loss variants are anchored to
# strategy=margin (the canonical AL baseline). Edit this list to add or remove
# baselines from every AL plot at once.
BASE_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "key": "base_random",
        "strategy": "random",
        "loss": "cross_entropy",
        "calibration": "none",
        "label": "base + random",
        "render": "line",  # horizontal reference line on bar_al.py
    },
    {
        "key": "base_margin",
        "strategy": "margin",
        "loss": "cross_entropy",
        "calibration": "none",
        "label": "base + margin",
        "render": "line",
    },
    {
        "key": "base_label_smoothing",
        "strategy": "margin",
        "loss": "label_smoothing",
        "calibration": "none",
        "label": "base + margin + LS",
        "render": "bar",  # full bar (green) on bar_al.py, sorted into the leaderboard
    },
    {
        "key": "base_label_relaxation",
        "strategy": "margin",
        "loss": "label_relaxation",
        "calibration": "none",
        "label": "base + margin + LR",
        "render": "bar",
    },
    {
        "key": "base_temperature_scaling",
        "strategy": "margin",
        "loss": "cross_entropy",
        "calibration": "temperature_scaling",
        "label": "base + margin + temp",
        "render": "bar",
    },
    {
        "key": "base_vector_scaling",
        "strategy": "margin",
        "loss": "cross_entropy",
        "calibration": "vector_scaling",
        "label": "base + margin + vec",
        "render": "bar",
    },
)

# Backwards-compat alias used by older imports; same canonical strategies as
# the first two ``BASE_VARIANTS`` rows.
BASE_STRATEGIES: tuple[str, ...] = ("random", "margin")


def _nauc(records: list[dict]) -> np.ndarray:
    """Pull NAUC scalars from a list of merged records into a float ndarray."""
    out: list[float] = []
    for r in records:
        v = r.get("summary", {}).get("nauc")
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return np.asarray(out, dtype=float)


def _agg(records: list[dict]) -> tuple[float | None, float, int]:
    """Return ``(mean, std, n)``. ``mean`` is ``None`` when ``records`` is empty."""
    arr = _nauc(records)
    if arr.size == 0:
        return None, 0.0, 0
    std = float(arr.std()) if arr.size > 1 else 0.0
    return float(arr.mean()), std, int(arr.size)


def uq_ranking(
    dataset: str,
    notion: str,
    *,
    loss: str = DEFAULT_LOSS,
    calibration: str = DEFAULT_CAL,
    seeds: list[int] | None = None,
    fallback: bool = False,
) -> list[dict[str, Any]]:
    """Return the descending UQ-method ranking for one ``(dataset, notion)`` panel.

    Each entry carries the ``uncertainty`` strategy mean/std (the bar) and,
    when available, the same method's ``random`` and ``margin`` strategy means
    (the markers used to ground how much of the gain comes from the UQ
    signal). Methods missing the ``uncertainty`` cell entirely are skipped,
    unless ``fallback`` lets another notion from :data:`COLUMN_FALLBACKS` fill
    it. The entry's ``notion`` names the notion the values come from.
    """
    out: list[dict[str, Any]] = []
    for method in UQ_METHODS:
        if method in EXCLUDED_METHODS:
            continue
        unc_mean = None
        used_notion = notion
        can_fall_back = fallback and (dataset, method) not in NO_FALLBACK
        for used_notion in COLUMN_FALLBACKS[notion] if can_fall_back else (notion,):
            unc_records = al_cache.load_merged(
                ENTITY,
                SINK,
                dataset,
                method,
                "uncertainty",
                notion=used_notion,
                supervised_loss=loss,
                calibration=calibration,
                seeds=seeds,
            )
            unc_mean, unc_std, unc_n = _agg(unc_records)
            if unc_mean is not None and unc_n >= MIN_SEEDS:
                break
            unc_mean = None
        if unc_mean is None:
            continue

        rand_records = al_cache.load_merged(
            ENTITY,
            SINK,
            dataset,
            method,
            "random",
            supervised_loss=loss,
            calibration=calibration,
            seeds=seeds,
        )
        marg_records = al_cache.load_merged(
            ENTITY,
            SINK,
            dataset,
            method,
            "margin",
            supervised_loss=loss,
            calibration=calibration,
            seeds=seeds,
        )
        rand_mean, _rand_std, rand_n = _agg(rand_records)
        marg_mean, _marg_std, marg_n = _agg(marg_records)

        out.append(
            {
                "method": method,
                "label": method_label(method),
                "notion": used_notion,
                "mean": unc_mean,
                "std": unc_std,
                "n": unc_n,
                "random_mean": rand_mean,
                "random_n": rand_n,
                "margin_mean": marg_mean,
                "margin_n": marg_n,
            },
        )
    out.sort(key=lambda r: r["mean"], reverse=True)
    return out


def base_variant_levels(
    dataset: str,
    *,
    seeds: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Mean NAUC of ``method=base`` for each entry in :data:`BASE_VARIANTS`.

    Returned shape: ``{variant_key: {"label", "mean", "std", "n"}}``. Variants
    with no cached runs map to ``mean=None``.
    """
    out: dict[str, dict[str, Any]] = {}
    for variant in BASE_VARIANTS:
        records = al_cache.load_merged(
            ENTITY,
            SINK,
            dataset,
            "base",
            variant["strategy"],
            supervised_loss=variant["loss"],
            calibration=variant["calibration"],
            seeds=seeds,
        )
        mean, std, n = _agg(records)
        out[variant["key"]] = {
            "label": variant["label"],
            "mean": mean,
            "std": std,
            "n": n,
        }
    return out
