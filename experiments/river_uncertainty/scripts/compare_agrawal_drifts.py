"""Side-by-side: agrawal_drift (label-flip) vs agrawal_covariate_drift.

Hypothesis: deep methods miss label-flip drift because the input distribution
stays in-distribution -- members don't disagree. Covariate drift puts post-drift
inputs OOD, so members should disagree -> probly-UQ should fire.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from river_uq.plotting import (
    METHOD_LABEL,
    METHOD_ORDER,
    PROBLY_COLOR,
    SMOOTH_WIN,
    TAILORED_COLOR,
)
from river_uq.prequential import run_prequential

STREAMS = ["agrawal_drift", "agrawal_covariate_drift"]
STREAM_LABEL = {
    "agrawal_drift": "Agrawal label-flip (function 0->4)",
    "agrawal_covariate_drift": "Agrawal covariate (salary shift +80k)",
}
SEEDS = (0, 1, 2)
N = 3000


def _smooth(s: pd.Series, win: int = SMOOTH_WIN) -> pd.Series:
    return s.rolling(win, min_periods=1).mean()


def main() -> None:
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    for method in METHOD_ORDER:
        for stream in STREAMS:
            for seed in SEEDS:
                df = run_prequential(method=method, stream_name=stream, seed=seed, n_steps=N)
                parts.append(df)
    records: pd.DataFrame = pd.concat(parts, ignore_index=True)

    fig, axes = plt.subplots(len(METHOD_ORDER), len(STREAMS), figsize=(10, 7.5), squeeze=False)
    for j, stream in enumerate(STREAMS):
        axes[0, j].set_title(STREAM_LABEL[stream], fontsize=10)
    for i, method in enumerate(METHOD_ORDER):
        axes[i, 0].set_ylabel(METHOD_LABEL[method], fontsize=11)
    for i, method in enumerate(METHOD_ORDER):
        for j, stream in enumerate(STREAMS):
            ax = axes[i, j]
            sub = records[(records["method"] == method) & (records["stream"] == stream)]
            for seed, sg in sub.groupby("seed"):
                ax.plot(sg["t"], _smooth(sg["epi"]), alpha=0.7, lw=1.2)
            ymax = ax.get_ylim()[1]
            for _, sg in sub.groupby("seed"):
                pa = sg.loc[sg["alarm_probly"], "t"]
                ta = sg.loc[sg["alarm_tailored"], "t"]
                if len(pa):
                    ax.scatter([pa.iloc[0]], [ymax * 0.95], marker="v", color=PROBLY_COLOR, s=60, zorder=5)
                if len(ta):
                    ax.scatter([ta.iloc[0]], [ymax * 0.85], marker="^", color=TAILORED_COLOR, s=60, zorder=5)
            ax.axvline(2000, color="black", ls="--", alpha=0.4)
            ax.set_xlabel("step t" if i == len(METHOD_ORDER) - 1 else "")
            ax.grid(alpha=0.3)
    fig.suptitle(
        "Label-flip vs covariate drift on Agrawal\n"
        "v probly-UQ alarm   ^ tailored alarm   -- true drift @ t=2000",
        fontsize=10,
    )
    fig.tight_layout()
    out_pdf = out_dir / "agrawal_label_vs_covariate.pdf"
    out_png = out_dir / "agrawal_label_vs_covariate.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=120)

    print("=== alarm summary (3 seeds) ===")
    rows = []
    for key, g in records.groupby(["method", "stream", "seed"]):
        m, s, seed = cast("tuple[str, str, int]", key)
        pa = g.loc[g["alarm_probly"], "t"]
        ta = g.loc[g["alarm_tailored"], "t"]
        rows.append(
            {
                "method": m,
                "stream": s,
                "seed": seed,
                "probly_first": int(pa.iloc[0]) if len(pa) else None,
                "tailored_first": int(ta.iloc[0]) if len(ta) else None,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {out_pdf}")


if __name__ == "__main__":
    main()
