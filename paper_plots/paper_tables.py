r"""Build paper-ready LaTeX tables from cached runs.

Writes three booktabs ``tabular`` blocks under ``paper_artifacts/tables/``:

- ``paper_table_sp.tex``           — methods x ID datasets, cell = Acc-AUC.
- ``paper_table_ood.tex``          — methods x (dataset x {near, far}), cell = AUROC.
- ``paper_table_<dataset>.tex``    — combined SP + Near/Far OOD per dataset.

Cells are ``$mean_{\pm std}$`` with the column-best bolded. Methods are
sectioned by :data:`GROUPS`; missing methods fall into an "Other" trailing
section. Edit :data:`GROUPS` and :data:`CITATIONS` to adjust grouping or
citation keys. Requires ``\usepackage{booktabs, arydshln}`` and
``\usepackage[table]{xcolor}`` in your preamble.

Run::

    uv run python -m paper_plots.paper_tables
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from paper_plots import ood_groups
from paper_plots.aggregate import ood_band_ranking, sp_ranking
from paper_plots.metadata import dataset_display
from paper_plots.paths import PAPER_ARTIFACTS
from paper_plots.targets import TARGETS

if TYPE_CHECKING:
    from collections.abc import Callable

# --- edit me -----------------------------------------------------------------
# Citation keys are the ones the paper uses, so they resolve in its bib file.
OUT_DIR = PAPER_ARTIFACTS / "paper_tables"
DECIMALS = 3

# Same families and order as the bar plot legend (see paper_plots.style), with full words.
GROUPS: list[dict] = [
    {
        "name": "Deterministic baseline",
        "methods": ["base"],
    },
    {
        "name": "Direct",
        "methods": ["ddu", "duq", "het_net", "deup"],
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
            "prior_network",
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
            "base_vector_scaling",
            "base_temperature_scaling",
        ],
    },
]

CITATIONS: dict[str, str] = {
    # --- Second-order methods ---
    "dropout": "galDropout2016",
    "dropconnect": "wanDropConnect2013",
    "bayesian": "blundellWeightUncertainty2015",
    "laplace": "daxbergerLaplaceRedux2021",
    "vbll": "harrisonVBLL2024",
    "ensemble": "laks_sa17",
    "batchensemble": "wenBatchEnsemble2020",
    "subensemble": "valdenegroToroSubEnsemble2023",
    "dare": "deMathelinDARE2023",
    "duq": "vanAmersfoortDUE2021",
    "ddu": "mukhotiDDU2023",
    "sngp": "liuSNGP2020",
    "evidential_classification": "sens_ed18",
    "posterior_network": "charpentierPosteriorNetwork2020",
    "natural_posterior_network": "charpentierNaturalPosteriorNetwork2022",
    "prior_network": "malininPriorNetworks2018",
    "het_net": "collierHETNet2021",
    # --- Credal-set methods ---
    "credal_wrapper": "wangCredalWrapper2025",
    "credal_ensembling": "nguyenCredalEnsembling2025",
    "credal_bnn": "caprioCredalBNNs2024",
    "credal_net": "wangCredalIntervalNNs2025",
    "credal_relative_likelihood": "loehrCredalPrediction2025",
    "efficient_credal_prediction": "hofmanEfficientCredalPrediction2026",
    # --- Calibration variants of base ---
    "base_label_smoothing": "szegedyRethinkingInceptionArchitecture2016",
    "base_label_relaxation": "DBLP:conf/aaai/LienenH21",
    "base_temperature_scaling": "guoOnCalibration2017",
    "base_vector_scaling": "guoOnCalibration2017",
}
# -----------------------------------------------------------------------------

_PREAMBLE = (
    "% Requires in your preamble:\n"
    "%   \\usepackage{booktabs}\n"
    "%   \\usepackage{arydshln}             % for \\hdashline / \\cdashline\n"
    "%   \\usepackage[table]{xcolor}        % for \\rowcolor on group-header rows\n"
)
_TABCOLSEP_PT = 7
# Float placement per table, as tuned in the paper.
_PLACEMENT = {"imagenet": "h", "cifar10": "ht"}
_UP = r"($\uparrow$)"
_ARRAYSTRETCH = "1.15"
_HEADER_VSKIP = "0.45em"
_TABLE_CLOSE: tuple[str, ...] = (r"\end{tabular}", r"\end{table}", "")


def _table_open(*, title: str, label_key: str, placement: str = "t") -> list[str]:
    return [
        f"\\begin{{table}}[{placement}]",
        r"\centering",
        f"\\caption{{{title}}}",
        f"\\label{{tab:{label_key}}}",
        r"\footnotesize",
        f"\\setlength{{\\tabcolsep}}{{{_TABCOLSEP_PT}pt}}",
        f"\\renewcommand{{\\arraystretch}}{{{_ARRAYSTRETCH}}}",
    ]


def _strip_leading_zero(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def _format_cell(entry: dict | None, *, decimals: int, is_best: bool) -> str:
    if entry is None:
        return "--"
    mean = _strip_leading_zero(entry["mean"], decimals)
    std = _strip_leading_zero(entry["std"], decimals)
    body = f"{mean}_{{\\pm {std}}}"
    return f"$\\mathbf{{{body}}}$" if is_best else f"${body}$"


def _cite_cell(method: str) -> str:
    key = CITATIONS.get(method)
    return f"\\cite{{{key}}}" if key else "---"


def _collect_method_labels(rankings: list[list[dict]]) -> dict[str, str]:
    method_labels: dict[str, str] = {}
    for ranking in rankings:
        for entry in ranking:
            method_labels.setdefault(entry["method"], entry["label"])
    return method_labels


def _grouped_sections(method_labels: dict[str, str]) -> list[tuple[str | None, list[str]]]:
    if not GROUPS:
        return [(None, sorted(method_labels))]
    used: set[str] = set()
    sections: list[tuple[str | None, list[str]]] = []
    for group in GROUPS:
        name = str(group.get("name", "")) or None
        members = [m for m in group.get("methods", []) if m in method_labels and m not in used]
        used.update(members)
        if members:
            sections.append((name, members))
    leftover = sorted(m for m in method_labels if m not in used)
    if leftover:
        sections.append(("Other", leftover))
    return sections


def _group_header_row(name: str, n_cols: int) -> list[str]:
    return [
        r"\addlinespace[0.4em]",
        r"\rowcolor[gray]{0.92}[0pt][0pt]",
        f"\\multicolumn{{{n_cols}}}{{@{{}}l@{{}}}}{{\\textbf{{{name}}}}} \\\\",
        r"\addlinespace[0.15em]",
    ]


def _emit_grouped_rows(
    sections: list[tuple[str | None, list[str]]],
    n_total_cols: int,
    row_renderer: Callable[[str], str],
) -> list[str]:
    out: list[str] = []
    for section_idx, (section_name, section_methods) in enumerate(sections):
        if section_name is not None:
            header = _group_header_row(section_name, n_total_cols)
            out.extend(header[1:] if section_idx == 0 else header)
        out.extend(row_renderer(method) for method in section_methods)
    return out


def _sp_table(sp: dict[str, list[dict]]) -> str:
    datasets = sorted(sp)
    method_labels = _collect_method_labels(list(sp.values()))
    sections = _grouped_sections(method_labels)
    cells: dict[tuple[str, str], dict] = {}
    for ds, ranking in sp.items():
        for entry in ranking:
            cells[(entry["method"], ds)] = entry
    best_per_col = {ds: max(ranking, key=lambda e: e["mean"])["method"] for ds, ranking in sp.items() if ranking}

    n_data_cols = len(datasets)
    n_total_cols = 2 + n_data_cols
    col_spec = "@{}l@{\\hspace{4pt}}l " + " ".join(["c"] * n_data_cols) + "@{}"
    lines = [
        _PREAMBLE.rstrip(),
        *_table_open(title="Selective prediction (Acc-AUC)", label_key="results-sp"),
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]
    header = r"\multicolumn{2}{@{}l}{\textbf{Method}}"
    for ds in datasets:
        header += f" & \\textbf{{{dataset_display(ds)}}}"
    lines.append(header + r" \\")
    lines.append(r"\hdashline\noalign{\vskip " + _HEADER_VSKIP + "}")

    def render(method: str) -> str:
        row = f"{_cite_cell(method)} & {method_labels[method]}"
        for ds in datasets:
            row += " & " + _format_cell(
                cells.get((method, ds)),
                decimals=DECIMALS,
                is_best=best_per_col.get(ds) == method,
            )
        return row + r" \\"

    lines.extend(_emit_grouped_rows(sections, n_total_cols, render))
    lines += [r"\bottomrule", *_TABLE_CLOSE]
    return "\n".join(lines)


def _ood_table(ood: dict[tuple[str, str], list[dict]]) -> str:
    datasets = sorted({ds for _, ds in ood})
    bands = ("near", "far")
    method_labels = _collect_method_labels(list(ood.values()))
    sections = _grouped_sections(method_labels)
    cells: dict[tuple[str, str, str], dict] = {}
    for (band, ds), ranking in ood.items():
        for entry in ranking:
            cells[(entry["method"], band, ds)] = entry
    best_per_col: dict[tuple[str, str], str] = {}
    for (band, ds), ranking in ood.items():
        if ranking:
            best_per_col[(band, ds)] = max(ranking, key=lambda e: e["mean"])["method"]

    n_band = len(bands)
    n_data_cols = n_band * len(datasets)
    n_total_cols = 2 + n_data_cols
    col_spec = "@{}l@{\\hspace{4pt}}l " + " ".join(["c" * n_band] * len(datasets)) + "@{}"
    lines = [
        _PREAMBLE.rstrip(),
        *_table_open(title="OoD detection (AUROC)", label_key="results-ood"),
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    spans = r"\multicolumn{2}{@{}l}{}"
    cdash: list[str] = []
    col_idx = 3
    for ds in datasets:
        spans += f" & \\multicolumn{{{n_band}}}{{c}}{{\\textbf{{{dataset_display(ds)}}}}}"
        cdash.append(f"\\cdashline{{{col_idx}-{col_idx + n_band - 1}}}")
        col_idx += n_band
    lines.append(spans + r" \\")
    lines.append(" ".join(cdash))
    lines.append(r"\noalign{\vskip " + _HEADER_VSKIP + "}")

    sub_header = r"\multicolumn{2}{@{}l}{\textbf{Method}}"
    for _ in datasets:
        for band in bands:
            sub_header += f" & {band.title()}-OoD"
    lines.append(sub_header + r" \\")
    lines.append(r"\hdashline\noalign{\vskip " + _HEADER_VSKIP + "}")

    def render(method: str) -> str:
        row = f"{_cite_cell(method)} & {method_labels[method]}"
        for ds in datasets:
            for band in bands:
                row += " & " + _format_cell(
                    cells.get((method, band, ds)),
                    decimals=DECIMALS,
                    is_best=best_per_col.get((band, ds)) == method,
                )
        return row + r" \\"

    lines.extend(_emit_grouped_rows(sections, n_total_cols, render))
    lines += [r"\bottomrule", *_TABLE_CLOSE]
    return "\n".join(lines)


def _per_dataset_table(
    dataset: str,
    sp_for_ds: list[dict] | None,
    ood_for_ds: dict[str, list[dict]],
) -> str:
    bands = ("near", "far")
    columns: list[tuple[str, str, str]] = []
    if sp_for_ds:
        columns.append(("sp", "SP Acc-Rej", "sp"))
    columns.extend((band, f"{band.title()}-OoD AUROC", f"ood/{band}") for band in bands if ood_for_ds.get(band))
    if not columns:
        msg = f"no SP or OOD rankings for dataset {dataset!r}"
        raise ValueError(msg)

    rankings_for_labels: list[list[dict]] = []
    if sp_for_ds:
        rankings_for_labels.append(sp_for_ds)
    rankings_for_labels.extend(ood_for_ds.get(b, []) for b in bands)
    method_labels = _collect_method_labels(rankings_for_labels)
    sections = _grouped_sections(method_labels)

    sp_cells = {entry["method"]: entry for entry in (sp_for_ds or [])}
    ood_cells: dict[tuple[str, str], dict] = {}
    for band in bands:
        for entry in ood_for_ds.get(band, []) or []:
            ood_cells[(entry["method"], band)] = entry

    best_per_col: dict[str, str] = {}
    if sp_for_ds:
        best_per_col["sp"] = max(sp_for_ds, key=lambda e: e["mean"])["method"]
    for band in bands:
        ranking = ood_for_ds.get(band) or []
        if ranking:
            best_per_col[f"ood/{band}"] = max(ranking, key=lambda e: e["mean"])["method"]

    n_data_cols = len(columns)
    n_total_cols = 2 + n_data_cols
    col_spec = "@{}l@{\\hspace{4pt}}l " + " ".join(["c"] * n_data_cols) + "@{}"
    lines = [
        _PREAMBLE.rstrip(),
        *_table_open(
            title=(
                f"Summary Results on {dataset_display(dataset)}. For Selective prediction we report "
                "accuracy-rejection AUC (Acc-Rej) and for OoD AUROC."
            ),
            label_key=f"results-{dataset}",
            placement=_PLACEMENT.get(dataset, "t"),
        ),
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]
    header = r"\multicolumn{2}{@{}l}{\textbf{Method}}"
    for _kind, label, _key in columns:
        header += f" & \\textbf{{{label}}} {_UP}"
    lines.append(header + r" \\")
    lines.append(r"\hdashline\noalign{\vskip " + _HEADER_VSKIP + "}")

    def render(method: str) -> str:
        row = f"{_cite_cell(method)} & {method_labels[method]}"
        for kind, _label, key in columns:
            cell = sp_cells.get(method) if kind == "sp" else ood_cells.get((method, kind))
            row += " & " + _format_cell(cell, decimals=DECIMALS, is_best=best_per_col.get(key) == method)
        return row + r" \\"

    lines.extend(_emit_grouped_rows(sections, n_total_cols, render))
    lines += [r"\bottomrule", *_TABLE_CLOSE]
    return "\n".join(lines)


def _collect_rankings() -> tuple[
    dict[str, list[dict]],
    dict[tuple[str, str], list[dict]],
]:
    """Build SP / OOD ranking dicts from the cache for every TARGET."""
    sp: dict[str, list[dict]] = {}
    ood: dict[tuple[str, str], list[dict]] = {}
    for target in TARGETS:
        sp_r = sp_ranking(target)
        if sp_r:
            sp[target.dataset] = sp_r
        for band in ("near", "far"):
            band_r = [r for r in ood_band_ranking(target, band) if r["method"] not in ood_groups.EXCLUDED_METHODS]
            if band_r:
                ood[(band, target.dataset)] = band_r
    return sp, ood


def main() -> None:
    sp, ood = _collect_rankings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if sp:
        path = OUT_DIR / "paper_table_sp.tex"
        path.write_text(_sp_table(sp))
        print(f"wrote {path} (datasets: {sorted(sp)})")
    else:
        print("no SP rankings; skipping paper_table_sp.tex")

    if ood:
        path = OUT_DIR / "paper_table_ood.tex"
        path.write_text(_ood_table(ood))
        print(f"wrote {path} (datasets: {sorted({ds for _, ds in ood})})")
    else:
        print("no OOD rankings; skipping paper_table_ood.tex")

    per_dataset_ids = set(sp.keys()) | {ds for _, ds in ood}
    for ds in sorted(per_dataset_ids):
        ood_for_ds = {band: ood.get((band, ds), []) for band in ("near", "far")}
        text = _per_dataset_table(ds, sp.get(ds), ood_for_ds)
        path = OUT_DIR / f"paper_table_{ds}.tex"
        path.write_text(text)
        print(f"wrote {path} (combined SP + OOD for {ds})")


if __name__ == "__main__":
    main()
