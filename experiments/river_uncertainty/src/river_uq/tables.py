"""Latency table builder.

Aggregates per-step records into a (method × stream) summary that splits
detection-latency-on-true-positives from false-alarm rate. Tailored
detectors (PageHinkley, ADWIN) can fire before the true drift on noisy
streams; averaging negative latencies with positive ones produces
meaningless means, so we report the two quantities separately.
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


def _first_alarm_t(group: pd.DataFrame, col: str) -> float:
    fired = group.loc[group[col], "t"]
    if len(fired) == 0:
        return np.nan
    return float(fired.iloc[0])


def build_latency_table(records: pd.DataFrame) -> pd.DataFrame:
    """Build the (method x stream) detection-latency + false-alarm table.

    For each (method, stream, seed): an alarm with ``t < true_drift_t`` is
    counted as a false alarm; an alarm with ``t >= true_drift_t`` contributes
    its latency (``alarm_t - true_drift_t``) to the true-positive pool. Seeds
    with no alarm at all are missed detections. Means and stds are reported
    over true-positive latencies only; false-alarm and miss counts are reported
    as ``k/N`` summaries.

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
            probly_t = _first_alarm_t(sg, "alarm_probly")
            tailored_t = _first_alarm_t(sg, "alarm_tailored")
            per_seed.append(
                {
                    "seed": seed,
                    "true_t": true_t,
                    "probly_t": probly_t,
                    "tailored_t": tailored_t,
                }
            )
        ps = pd.DataFrame(per_seed)
        n = len(ps)

        def _split(prefix: str) -> dict:
            t_col = ps[f"{prefix}_t"]
            true_col = ps["true_t"]
            if true_col.iloc[0] is None or pd.isna(true_col.iloc[0]):
                return {
                    f"{prefix}_tp_mean": np.nan,
                    f"{prefix}_tp_std": np.nan,
                    f"{prefix}_n_tp": 0,
                    f"{prefix}_n_fa": 0,
                    f"{prefix}_n_miss": int(t_col.isna().sum()),
                }
            true_t_val = float(true_col.iloc[0])
            fired = ~t_col.isna()
            tp_mask = fired & (t_col >= true_t_val)
            fa_mask = fired & (t_col < true_t_val)
            tp_lat = t_col[tp_mask] - true_t_val
            return {
                f"{prefix}_tp_mean": float(tp_lat.mean()) if len(tp_lat) else np.nan,
                f"{prefix}_tp_std": float(tp_lat.std()) if len(tp_lat) > 1 else np.nan,
                f"{prefix}_n_tp": int(tp_mask.sum()),
                f"{prefix}_n_fa": int(fa_mask.sum()),
                f"{prefix}_n_miss": int((~fired).sum()),
            }

        row = {"method": method, "stream": stream, "n_seeds": n}
        row.update(_split("probly"))
        row.update(_split("tailored"))
        row["tailored_label"] = TAILORED_LABEL.get(method, "-")
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _fmt_cell(mean: float, std: float, n_tp: int, n_fa: int, n_seeds: int) -> str:
    if n_tp == 0:
        return f"\\textit{{none}} ({n_fa}/{n_seeds} FA)"
    if n_tp == 1 or np.isnan(std):
        body = f"{mean:.0f}"
    else:
        body = f"{mean:.0f} $\\pm$ {std:.0f}"
    return f"{body} ({n_tp}/{n_seeds} TP, {n_fa} FA)"


def latency_table_to_latex(table: pd.DataFrame) -> str:
    """Render the latency table as a booktabs LaTeX tabular fragment.

    Each cell shows ``mean +/- std (n_tp/n_seeds TP, n_fa FA)`` so that
    detection latency on true positives is reported separately from the
    pre-drift false-alarm count.
    """
    rows: list[str] = []
    for _, r in table.iterrows():
        n_seeds = int(r["n_seeds"])
        probly = _fmt_cell(
            r["probly_tp_mean"],
            r["probly_tp_std"],
            int(r["probly_n_tp"]),
            int(r["probly_n_fa"]),
            n_seeds,
        )
        tailored = _fmt_cell(
            r["tailored_tp_mean"],
            r["tailored_tp_std"],
            int(r["tailored_n_tp"]),
            int(r["tailored_n_fa"]),
            n_seeds,
        )
        rows.append(
            f"{r['method']} & {r['stream']} & {probly} & {tailored} & {r['tailored_label']} \\\\"
        )
    body = "\n        ".join(rows)
    return (
        "\\begin{tabular}{llllc}\n"
        "    \\toprule\n"
        "    Method & Stream & probly-UQ latency & Tailored latency & Tailored detector \\\\\n"
        "    \\midrule\n"
        f"        {body}\n"
        "    \\bottomrule\n"
        "\\end{tabular}\n"
        "% Cells: latency_mean $\\pm$ std (n_tp/n_seeds TP, n_fa FA).\n"
        "% TP = alarm at or after true drift; FA = alarm before true drift.\n"
        "% PageHinkley-on-error requires labels at inference; probly-UQ does not.\n"
    )
