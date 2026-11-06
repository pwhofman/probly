r"""Active-learning NAUC summary table (rows = methods, columns = dataset x notion).

Skips ``aleatoric`` (AU) per request: columns are EU and TU only. Rows follow
the same group structure as :mod:`paper_plots.paper_tables` (deterministic
baseline / calibration variants / second-order / credal). Cells are
``$mean_{\pm std}$`` across seeds with the column-best bolded.

Reads only the merged AL cache. Run::

    uv run python -m paper_plots.al_table

Writes ``paper_artifacts/tables/paper_table_al.tex``.
"""

from __future__ import annotations

from paper_plots import _table_layout
from paper_plots.al_aggregate import BASE_VARIANTS, base_variant_levels, uq_ranking
from paper_plots.metadata import dataset_display
from paper_plots.paths import PAPER_ARTIFACTS

# --- edit me -----------------------------------------------------------------
OUT_DIR = PAPER_ARTIFACTS / "paper_tables"
DECIMALS = 3
DATASETS: tuple[str, ...] = ("openml_6", "openml_156", "cifar10")
NOTIONS: tuple[str, ...] = ("epistemic", "total")  # AU intentionally skipped
NOTION_LABEL = {"epistemic": "EU", "aleatoric": "AU", "total": "TU"}

# Same families and order as the bar plot legend (see paper_plots.style), with full words.
GROUPS: list[dict] = [
    {
        "name": "Baselines",
        "methods": ["base_margin", "base_random"],
    },
    {
        "name": "Direct",
        "methods": ["ddu", "duq", "het_net"],
    },
    {
        "name": "2nd-order Distribution",
        "methods": [
            "dropout",
            "dropconnect",
            "bayesian",
            "laplace",
            "vbll",
            "ensemble",
            "batchensemble",
            "subensemble",
            "dare",
            "sngp",
            "evidential_classification",
            "posterior_network",
            "natural_posterior_network",
        ],
    },
    {
        "name": "Credal",
        "methods": [
            "credal_wrapper",
            "credal_ensembling",
            "credal_bnn",
            "credal_relative_likelihood",
            "efficient_credal_prediction",
        ],
    },
    {
        "name": "Calibration",
        "methods": [
            "base_label_smoothing",
            "base_label_relaxation",
            "base_temperature_scaling",
            "base_vector_scaling",
        ],
    },
]
CAPTION = (
    "Active learning (NAUC). LS denotes Label Smoothing, LR denotes Label Relaxation, temp denotes "
    "temperature scaling, and vec vector scaling. A superscript names the notion a value comes from "
    "when a method has no run for the column's notion."
)
# 8pt, as in the paper's two-dataset version, overflows now that CIFAR-10 is a third dataset.
TABCOLSEP_PT = 6
# -----------------------------------------------------------------------------


def _collect_rankings() -> dict[tuple[str, str], list[dict]]:
    """Build ``{(dataset, notion): ranking}`` for every panel we want to render.

    Baseline variants don't have a notion axis (random / margin acquisition),
    so the same ``(mean, std)`` is repeated across the EU and TU columns for a
    given dataset.
    """
    out: dict[tuple[str, str], list[dict]] = {}
    for ds in DATASETS:
        variants = base_variant_levels(ds)
        for notion in NOTIONS:
            # A method without a run for this notion shows another notion's value, marked in its cell.
            ranking = uq_ranking(ds, notion, fallback=True)
            for v in BASE_VARIANTS:
                info = variants.get(v["key"])
                if info is None or info["mean"] is None:
                    continue
                ranking.append(
                    {
                        "method": v["key"],
                        "label": info["label"],
                        "mean": info["mean"],
                        "std": info["std"],
                    },
                )
            if ranking:
                out[(ds, notion)] = ranking
    return out


def _render_table(rankings: dict[tuple[str, str], list[dict]]) -> str:
    columns = [(ds, notion) for ds in DATASETS for notion in NOTIONS if (ds, notion) in rankings]
    if not columns:
        msg = "no rankings available; run fetch_al + analyze_al first"
        raise ValueError(msg)

    method_labels = _table_layout.collect_method_labels(list(rankings.values()))
    sections = _table_layout.grouped_sections(method_labels, GROUPS)

    cells: dict[tuple[str, tuple[str, str]], dict] = {}
    for col, ranking in rankings.items():
        for entry in ranking:
            cells[(entry["method"], col)] = entry
    best_per_col: dict[tuple[str, str], str] = {}
    for col, ranking in rankings.items():
        # Only a run of the column's own notion can be the column's best.
        own_notion = [e for e in ranking if e.get("notion", col[1]) == col[1]]
        if own_notion:
            best_per_col[col] = max(own_notion, key=lambda e: e["mean"])["method"]

    n_data_cols = len(columns)
    n_total_cols = 1 + n_data_cols
    # One c-block per dataset, each block has len(NOTIONS) sub-cols.
    ds_blocks: list[tuple[str, int]] = []
    for ds in DATASETS:
        n_cols_for_ds = sum(1 for c in columns if c[0] == ds)
        if n_cols_for_ds:
            ds_blocks.append((ds, n_cols_for_ds))
    col_spec = "@{}l " + " ".join(["c" * n for _, n in ds_blocks]) + "@{}"

    lines = [
        _table_layout.PREAMBLE.rstrip(),
        *_table_layout.table_open(title=CAPTION, label_key="results-al", tabcolsep_pt=TABCOLSEP_PT),
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Top header row: dataset spans.
    spans = r"\multicolumn{1}{@{}l}{}"
    cdash: list[str] = []
    col_idx = 2  # column 1 is the method label
    for ds, n_cols_for_ds in ds_blocks:
        spans += f" & \\multicolumn{{{n_cols_for_ds}}}{{c}}{{\\textbf{{{dataset_display(ds)}}}}}"
        cdash.append(f"\\cdashline{{{col_idx}-{col_idx + n_cols_for_ds - 1}}}")
        col_idx += n_cols_for_ds
    lines.append(spans + r" \\")
    lines.append(" ".join(cdash))
    lines.append(r"\noalign{\vskip " + _table_layout.HEADER_VSKIP + "}")

    # Sub-header: Method + per-column EU/TU labels.
    sub_header = r"\multicolumn{1}{@{}l}{\textbf{Method}}"
    for _ds, notion in columns:
        sub_header += f" & \\textbf{{{NOTION_LABEL[notion]}}}"
    lines.append(sub_header + r" \\")
    lines.append(r"\hdashline\noalign{\vskip " + _table_layout.HEADER_VSKIP + "}")

    def render(method: str) -> str:
        row = method_labels[method]
        for col in columns:
            cell = cells.get((method, col))
            notion = cell.get("notion", col[1]) if cell else col[1]
            row += " & " + _table_layout.format_cell(
                cell,
                decimals=DECIMALS,
                is_best=best_per_col.get(col) == method,
                mark=NOTION_LABEL[notion] if notion != col[1] else None,
            )
        return row + r" \\"

    lines.extend(_table_layout.emit_grouped_rows(sections, n_total_cols, render))
    lines += [r"\bottomrule", *_table_layout.TABLE_CLOSE]
    return "\n".join(lines)


def main() -> None:
    rankings = _collect_rankings()
    if not rankings:
        msg = "no AL rankings cached; run paper_plots.al_all first"
        raise SystemExit(msg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = _render_table(rankings)
    out = OUT_DIR / "paper_table_al.tex"
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
