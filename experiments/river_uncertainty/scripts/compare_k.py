"""Compare ProblyUQDetector k=4.0 (default) vs k=3.0 across all combos.

Reads the existing run_records.parquet (which already has the per-step epi
signal) and re-applies the detector with different k values. No need to
re-run models or streams.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from river_uq.detectors import ProblyUQDetector

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _replay(group: pd.DataFrame, k: float) -> int | None:
    det = ProblyUQDetector(k=k)
    for row in group.itertuples():
        fired = det.update(int(row.t), epi=float(row.epi), error=0, model_n_drifts=0)
        if fired:
            return int(row.t)
    return None


def main() -> None:
    df = pd.read_parquet(RESULTS / "run_records.parquet")
    rows = []
    for (m, s, seed), g in df.groupby(["method", "stream", "seed"]):
        true_t = g["true_drift_t"].iloc[0]
        rows.append(
            {
                "method": m,
                "stream": s,
                "seed": seed,
                "true_t": true_t,
                "alarm_k4": _replay(g, k=4.0),
                "alarm_k3": _replay(g, k=3.0),
            }
        )
    out = pd.DataFrame(rows)

    summary_rows = []
    for (m, s), g in out.groupby(["method", "stream"]):
        if s == "agrawal_stationary":
            # stationary: count any alarm as a false positive
            fa_k4 = g["alarm_k4"].notna().sum()
            fa_k3 = g["alarm_k3"].notna().sum()
            summary_rows.append(
                {"method": m, "stream": s, "k4_tp": "-", "k4_fa": int(fa_k4), "k3_tp": "-", "k3_fa": int(fa_k3)}
            )
            continue
        true_t = g["true_t"].iloc[0]
        for k_label, col in [("k4", "alarm_k4"), ("k3", "alarm_k3")]:
            tp = ((g[col].notna()) & (g[col] >= true_t)).sum()
            fa = ((g[col].notna()) & (g[col] < true_t)).sum()
            if k_label == "k4":
                k4_tp, k4_fa = int(tp), int(fa)
            else:
                k3_tp, k3_fa = int(tp), int(fa)
        summary_rows.append(
            {"method": m, "stream": s, "k4_tp": k4_tp, "k4_fa": k4_fa, "k3_tp": k3_tp, "k3_fa": k3_fa}
        )
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
