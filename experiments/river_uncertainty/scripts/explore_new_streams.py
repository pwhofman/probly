"""Exploratory run: agrawal_drift + electricity, all 3 methods, 2 seeds.

Produces a 3x2 grid (3 methods x 2 new streams) showing the epistemic signal
and alarm timings -- same template as the headline figure.

Electricity has no labeled drift point so the dashed vertical line is omitted.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("RIVER_DATA", "/tmp/claude/river_data")

from river_uq.plotting import (  # noqa: E402
    METHOD_LABEL,
    METHOD_ORDER,
    PROBLY_COLOR,
    SMOOTH_WIN,
    TAILORED_COLOR,
)
from river_uq.prequential import run_prequential  # noqa: E402

NEW_STREAMS = ["agrawal_drift", "electricity"]
NEW_STREAM_LABEL = {
    "agrawal_drift": "Agrawal (abrupt drift @ t=2000)",
    "electricity": "Elec2 (real, no labeled drift)",
}
SEEDS = (0, 1)
N_AGRAWAL = 3000
N_ELEC = 10000  # cap; full Elec2 is 45k


def _smooth(series: pd.Series, win: int = SMOOTH_WIN) -> pd.Series:
    return series.rolling(win, min_periods=1).mean()


def main() -> None:
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    all_records: list[pd.DataFrame] = []
    for method in METHOD_ORDER:
        for stream in NEW_STREAMS:
            n = N_ELEC if stream == "electricity" else N_AGRAWAL
            for seed in SEEDS:
                df = run_prequential(method=method, stream_name=stream, seed=seed, n_steps=n)
                all_records.append(df)
    elapsed = time.time() - t0
    records = pd.concat(all_records, ignore_index=True)
    records.to_parquet(out_dir / "exploratory_records.parquet")
    print(f"ran in {elapsed:.1f}s -> {len(records)} rows")

    # 3 methods x 2 streams
    fig, axes = plt.subplots(
        len(METHOD_ORDER), len(NEW_STREAMS), figsize=(11, 7.5), sharex=False, squeeze=False
    )
    for j, stream in enumerate(NEW_STREAMS):
        axes[0, j].set_title(NEW_STREAM_LABEL[stream], fontsize=11)
    for i, method in enumerate(METHOD_ORDER):
        axes[i, 0].set_ylabel(METHOD_LABEL[method], fontsize=11)
    for i, method in enumerate(METHOD_ORDER):
        for j, stream in enumerate(NEW_STREAMS):
            ax = axes[i, j]
            sub = records[(records["method"] == method) & (records["stream"] == stream)]
            # plot per-seed smoothed epistemic
            for seed, sg in sub.groupby("seed"):
                ax.plot(sg["t"], _smooth(sg["epi"]), alpha=0.7, lw=1.2, label=f"seed {seed}")
            # alarm markers (probly = down triangle, tailored = up triangle)
            for _, sg in sub.groupby("seed"):
                pa = sg.loc[sg["alarm_probly"], "t"]
                ta = sg.loc[sg["alarm_tailored"], "t"]
                ymax = ax.get_ylim()[1]
                if len(pa):
                    ax.scatter([pa.iloc[0]], [ymax * 0.95], marker="v", color=PROBLY_COLOR, s=60, zorder=5)
                if len(ta):
                    ax.scatter([ta.iloc[0]], [ymax * 0.85], marker="^", color=TAILORED_COLOR, s=60, zorder=5)
            true_t = sub["true_drift_t"].iloc[0] if len(sub) else None
            if true_t is not None and not pd.isna(true_t):
                ax.axvline(float(true_t), color="black", ls="--", alpha=0.4)
            ax.set_xlabel("step t" if i == len(METHOD_ORDER) - 1 else "")
            ax.grid(alpha=0.3)
    fig.suptitle(
        "Exploratory: probly-UQ on agrawal_drift + electricity\n"
        "v probly-UQ alarm   ^ tailored alarm   -- true drift (where labeled)",
        fontsize=10,
    )
    fig.tight_layout()
    out_pdf = out_dir / "exploratory_figure.pdf"
    out_png = out_dir / "exploratory_figure.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_pdf} and {out_png}")

    # alarm summary
    print("\n=== alarm summary ===")
    rows = []
    for key, g in records.groupby(["method", "stream", "seed"]):
        m, s, seed = cast("tuple[str, str, int]", key)
        true_t = g["true_drift_t"].iloc[0]
        pa = g.loc[g["alarm_probly"], "t"]
        ta = g.loc[g["alarm_tailored"], "t"]
        rows.append(
            {
                "method": m,
                "stream": s,
                "seed": seed,
                "true_t": true_t,
                "probly_first": int(pa.iloc[0]) if len(pa) else None,
                "n_probly_alarms": int(g["alarm_probly"].sum()),
                "tailored_first": int(ta.iloc[0]) if len(ta) else None,
                "n_tailored_alarms": int(g["alarm_tailored"].sum()),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
