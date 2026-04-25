"""Latency table builder.

Aggregates per-step records into a (method × stream) summary of detection
latency for the probly-UQ detector and the tailored baseline.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

TAILORED_LABEL = {
    "arf": "ADWIN (native, per-tree)",
    "deep_ensemble": "PageHinkley on error*",
    "mc_dropout": "PageHinkley on error*",
}


def _alarm_t(group: pd.DataFrame, col: str) -> float:
    fired = group.loc[group[col], "t"]
    if len(fired) == 0:
        return np.nan
    return float(fired.iloc[0])


def build_latency_table(records: pd.DataFrame) -> pd.DataFrame:
    """Build the (method × stream) detection-latency table.

    Latency is ``alarm_t - true_drift_t`` for each seed; missed alarms
    contribute to a count column instead of being averaged.

    Args:
        records: Tidy per-step DataFrame as produced by ``run_prequential``.

    Returns:
        Aggregated DataFrame with one row per (method, stream).
    """
    out_rows: list[dict] = []
    for key, grp in records.groupby(["method", "stream"]):
        method, stream = cast("tuple[str, str]", key)
        per_seed = []
        for seed, sg in grp.groupby("seed"):
            true_t = sg["true_drift_t"].iloc[0]
            probly_t = _alarm_t(sg, "alarm_probly")
            tailored_t = _alarm_t(sg, "alarm_tailored")
            per_seed.append(
                {
                    "seed": seed,
                    "true_t": true_t,
                    "probly_lat": (probly_t - true_t) if not np.isnan(probly_t) and true_t is not None else np.nan,
                    "tailored_lat": (tailored_t - true_t) if not np.isnan(tailored_t) and true_t is not None else np.nan,
                }
            )
        ps = pd.DataFrame(per_seed)
        out_rows.append(
            {
                "method": method,
                "stream": stream,
                "n_seeds": len(ps),
                "probly_mean": ps["probly_lat"].mean(skipna=True),
                "probly_std": ps["probly_lat"].std(skipna=True),
                "n_missed_probly": int(ps["probly_lat"].isna().sum()),
                "tailored_mean": ps["tailored_lat"].mean(skipna=True),
                "tailored_std": ps["tailored_lat"].std(skipna=True),
                "n_missed_tailored": int(ps["tailored_lat"].isna().sum()),
                "tailored_label": TAILORED_LABEL.get(method, "—"),
            }
        )
    return pd.DataFrame(out_rows)


def latency_table_to_latex(table: pd.DataFrame) -> str:
    """Render the latency table as a booktabs LaTeX tabular fragment."""
    rows: list[str] = []
    for _, r in table.iterrows():
        probly = (
            f"{r['probly_mean']:.0f} $\\pm$ {r['probly_std']:.0f}"
            if not np.isnan(r["probly_mean"])
            else f"\\textit{{missed}} ({int(r['n_missed_probly'])}/{int(r['n_seeds'])})"
        )
        tailored = (
            f"{r['tailored_mean']:.0f} $\\pm$ {r['tailored_std']:.0f}"
            if not np.isnan(r["tailored_mean"])
            else f"\\textit{{missed}} ({int(r['n_missed_tailored'])}/{int(r['n_seeds'])})"
        )
        rows.append(
            f"{r['method']} & {r['stream']} & {probly} & {tailored} & {r['tailored_label']} \\\\"
        )
    body = "\n        ".join(rows)
    return (
        "\\begin{tabular}{llrrl}\n"
        "    \\toprule\n"
        "    Method & Stream & probly-UQ latency & Tailored latency & Tailored detector \\\\\n"
        "    \\midrule\n"
        f"        {body}\n"
        "    \\bottomrule\n"
        "\\end{tabular}\n"
        "% *PageHinkley-on-error requires labels at inference; probly-UQ does not.\n"
    )
