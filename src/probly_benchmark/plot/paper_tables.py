r"""Assemble combined paper-ready LaTeX tables from per-dataset ranking JSONs.

Reads ``bar_sp_<dataset>.ranking.json`` and ``bar_ood_{near,far}_<dataset>.ranking.json``
files written by :mod:`probly_benchmark.plot.bar_ranked` across one or more
``inputs`` directories, and produces:

- ``paper_table_sp.tex``  (rows = methods, columns = ID datasets; cells = Acc-AUC)
- ``paper_table_ood.tex`` (rows = methods, columns = ID dataset x {near, far}; cells = AUROC)

Both are booktabs ``tabular`` blocks with ``mean $\pm$ std`` cells and the
column-best bolded.

Usage::

    uv run paper_tables.py \
        inputs='[/path/to/cifar10_methods, /path/to/imagenet_methods]' \
        save_path=/path/to/paper_tables
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig, OmegaConf

from probly_benchmark.plot.utils import resolve_save_path

if TYPE_CHECKING:
    from collections.abc import Callable


def _normalize_inputs(value: object) -> list[Path]:
    """Coerce ``inputs`` (string or list) to a list of absolute Paths."""
    if isinstance(value, str):
        items: list[str] = [value]
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        msg = f"`inputs` must be a string or list of strings; got {type(value).__name__}."
        raise TypeError(msg)
    return [Path(p).expanduser() for p in items]


def _load_rankings(
    inputs: list[Path],
) -> tuple[
    dict[str, list[dict]],
    dict[tuple[str, str], list[dict]],
    dict[tuple[str, str], list[dict]],
]:
    """Scan ``inputs`` for bar ranking JSONs.

    Returns:
        ``(sp_rankings, ood_rankings, al_rankings)``:

        - ``sp_rankings[dataset]`` -- per-method list (Acc-AUC).
        - ``ood_rankings[(band, dataset)]`` -- keyed by ``band in {"near", "far"}``.
        - ``al_rankings[(strategy, dataset)]`` -- keyed by AL acquisition strategy.
    """
    sp: dict[str, list[dict]] = {}
    ood: dict[tuple[str, str], list[dict]] = {}
    al: dict[tuple[str, str], list[dict]] = {}
    for root in inputs:
        if not root.exists():
            continue
        for path in sorted(root.glob("bar_sp_*.ranking.json")):
            stem = path.name[: -len(".ranking.json")]
            ds = stem[len("bar_sp_") :]
            sp[ds] = json.loads(path.read_text())
        for path in sorted(root.glob("bar_ood_*_*.ranking.json")):
            stem = path.name[: -len(".ranking.json")]
            rest = stem[len("bar_ood_") :]
            band, _, ds = rest.partition("_")
            if band in ("near", "far") and ds:
                ood[(band, ds)] = json.loads(path.read_text())
        for path in sorted(root.glob("bar_al_*_*.ranking.json")):
            # Filename shape: bar_al_<dataset>_<strategy>.ranking.json.
            # ``<dataset>`` itself can include underscores (e.g. ``openml_6``),
            # so split from the right.
            stem = path.name[: -len(".ranking.json")]
            rest = stem[len("bar_al_") :]
            ds, _, strategy = rest.rpartition("_")
            if ds and strategy:
                al[(strategy, ds)] = json.loads(path.read_text())
    return sp, ood, al


def _collect_method_labels(rankings: list[list[dict]]) -> dict[str, str]:
    """Map every method seen in any ranking to its display label."""
    method_labels: dict[str, str] = {}
    for ranking in rankings:
        for entry in ranking:
            method_labels.setdefault(entry["method"], entry["label"])
    return method_labels


def _grouped_sections(
    method_labels: dict[str, str],
    groups: list[dict],
) -> list[tuple[str | None, list[str]]]:
    """Materialize the user's group spec into ``(name, [methods])`` pairs.

    Methods are kept in the order provided by each group's ``methods:`` list.
    Methods present in the data but not mentioned in any group are appended
    in their own ``"Other"`` section. When ``groups`` is empty, returns a
    single nameless section with all methods alphabetically sorted (matching
    the prior flat-table behavior).
    """
    if not groups:
        return [(None, sorted(method_labels))]

    used: set[str] = set()
    sections: list[tuple[str | None, list[str]]] = []
    for group in groups:
        name = str(group.get("name", "")) or None
        wanted = [str(m) for m in group.get("methods", [])]
        members = [m for m in wanted if m in method_labels and m not in used]
        used.update(members)
        if members:
            sections.append((name, members))

    leftover = sorted(m for m in method_labels if m not in used)
    if leftover:
        sections.append(("Other", leftover))
    return sections


_PREAMBLE = (
    "% Requires in your preamble:\n"
    "%   \\usepackage{booktabs}\n"
    "%   \\usepackage{arydshln}             % for \\hdashline / \\cdashline (dashed sub-rules)\n"
    "%   \\usepackage[table]{xcolor}        % for \\rowcolor on group-header rows\n"
)

# Horizontal padding between columns (pt). Default tabcolsep is ~6pt; bumped
# to give Near-OOD / Far-OOD a bit of breathing room.
_TABCOLSEP_PT = 8

# Vertical row stretch — gentle scale on row height so dashed rules + bold
# headers don't feel cramped.
_ARRAYSTRETCH = "1.15"

# Vertical gap inserted below dataset-span row before the Near/Far sub-header,
# and below the sub-header before the dashed mid-rule.
_HEADER_VSKIP = "0.45em"


def _strip_leading_zero(value: float, decimals: int) -> str:
    """Format ``value`` with ``decimals`` digits, dropping a leading ``0`` if present.

    ``0.918`` becomes ``.918``; ``1.000`` stays ``1.000``. Negative values are
    not expected for AUROC / Acc-AUC bars but are passed through unchanged.
    """
    text = f"{value:.{decimals}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def _format_cell(entry: dict | None, *, decimals: int, is_best: bool) -> str:
    r"""Format one ``mean\,(std)`` cell, bolded when best.

    Numbers drop their leading ``0`` (``.918`` instead of ``0.918``) since
    every metric in these tables is a [0, 1] score; std is rendered in
    parentheses with a thin space.
    """
    if entry is None:
        return "--"
    mean = _strip_leading_zero(entry["mean"], decimals)
    std = _strip_leading_zero(entry["std"], decimals)
    body = f"{mean}\\,({std})"
    return r"\textbf{" + body + "}" if is_best else body


def _group_header_row(name: str, n_cols: int) -> list[str]:
    """Two-line booktabs-style group header (gray band + small extra space)."""
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
    """Emit grouped section headers + per-method data rows."""
    out: list[str] = []
    for section_idx, (section_name, section_methods) in enumerate(sections):
        if section_name is not None:
            header = _group_header_row(section_name, n_total_cols)
            # Skip the leading addlinespace on the very first group — the
            # column-header dashed rule already provided a gap.
            out.extend(header[1:] if section_idx == 0 else header)
        out.extend(row_renderer(method) for method in section_methods)
    return out


def _cite_cell(method: str, citations: dict[str, str]) -> str:
    r"""Render the citation sub-cell: ``\cite{key}`` or ``---`` when missing."""
    key = citations.get(method)
    return f"\\cite{{{key}}}" if key else "---"


def _sp_table(
    sp: dict[str, list[dict]],
    groups: list[dict],
    citations: dict[str, str],
    decimals: int,
) -> str:
    """Render the SP table: rows = methods, columns = ID datasets, cell = Acc-AUC."""
    datasets = sorted(sp)
    method_labels = _collect_method_labels(list(sp.values()))
    sections = _grouped_sections(method_labels, groups)

    cells: dict[tuple[str, str], dict] = {}
    for ds, ranking in sp.items():
        for entry in ranking:
            cells[(entry["method"], ds)] = entry
    best_per_col: dict[str, str] = {
        ds: max(ranking, key=lambda e: e["mean"])["method"] for ds, ranking in sp.items() if ranking
    }

    n_data_cols = len(datasets)
    n_total_cols = 2 + n_data_cols  # cite + name + datasets
    col_spec = "@{}l@{\\hspace{4pt}}l " + " ".join(["c"] * n_data_cols) + "@{}"
    lines = [
        _PREAMBLE.rstrip(),
        "{",
        f"\\setlength{{\\tabcolsep}}{{{_TABCOLSEP_PT}pt}}",
        f"\\renewcommand{{\\arraystretch}}{{{_ARRAYSTRETCH}}}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    header = r"\multicolumn{2}{@{}l}{\textbf{Method}}"
    for ds in datasets:
        header += f" & \\textbf{{{ds.upper()}}}"
    lines.append(header + r" \\")
    lines.append(r"\hdashline\noalign{\vskip " + _HEADER_VSKIP + "}")

    def render_sp_row(method: str) -> str:
        row = f"{_cite_cell(method, citations)} & {method_labels[method]}"
        for ds in datasets:
            row += " & " + _format_cell(
                cells.get((method, ds)),
                decimals=decimals,
                is_best=best_per_col.get(ds) == method,
            )
        return row + r" \\"

    lines.extend(_emit_grouped_rows(sections, n_total_cols, render_sp_row))
    lines += [r"\bottomrule", r"\end{tabular}", "}", ""]
    return "\n".join(lines)


def _ood_table(
    ood: dict[tuple[str, str], list[dict]],
    groups: list[dict],
    citations: dict[str, str],
    decimals: int,
) -> str:
    """Render the OOD table: rows = methods, columns = {dataset} x {near, far}."""
    datasets = sorted({ds for _, ds in ood})
    bands = ("near", "far")
    method_labels = _collect_method_labels(list(ood.values()))
    sections = _grouped_sections(method_labels, groups)

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
        "{",
        f"\\setlength{{\\tabcolsep}}{{{_TABCOLSEP_PT}pt}}",
        f"\\renewcommand{{\\arraystretch}}{{{_ARRAYSTRETCH}}}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Top header row: dataset spans, leaving the multicolumn(2) Method blank here.
    spans = r"\multicolumn{2}{@{}l}{}"
    cdash: list[str] = []
    col_idx = 3  # columns 1,2 are the citation+name half of the Method multicol
    for ds in datasets:
        spans += f" & \\multicolumn{{{n_band}}}{{c}}{{\\textbf{{{ds.upper()}}}}}"
        cdash.append(f"\\cdashline{{{col_idx}-{col_idx + n_band - 1}}}")
        col_idx += n_band
    lines.append(spans + r" \\")
    lines.append(" ".join(cdash))
    lines.append(r"\noalign{\vskip " + _HEADER_VSKIP + "}")

    # Sub-header row: Method multicolumn(2) + Near-OOD / Far-OOD per dataset.
    sub_header = r"\multicolumn{2}{@{}l}{\textbf{Method}}"
    for _ in datasets:
        for band in bands:
            sub_header += f" & \\textbf{{{band.title()}-OOD}}"
    lines.append(sub_header + r" \\")
    lines.append(r"\hdashline\noalign{\vskip " + _HEADER_VSKIP + "}")

    def render_ood_row(method: str) -> str:
        row = f"{_cite_cell(method, citations)} & {method_labels[method]}"
        for ds in datasets:
            for band in bands:
                row += " & " + _format_cell(
                    cells.get((method, band, ds)),
                    decimals=decimals,
                    is_best=best_per_col.get((band, ds)) == method,
                )
        return row + r" \\"

    lines.extend(_emit_grouped_rows(sections, n_total_cols, render_ood_row))
    lines += [r"\bottomrule", r"\end{tabular}", "}", ""]
    return "\n".join(lines)


def _al_table(
    al: dict[tuple[str, str], list[dict]],
    groups: list[dict],
    citations: dict[str, str],
    decimals: int,
) -> str:
    """Render the AL table: rows = methods, columns = {dataset} x {strategies}, cell = NAUC."""
    datasets = sorted({ds for _, ds in al})
    strategies = sorted({strategy for strategy, _ in al})
    method_labels = _collect_method_labels(list(al.values()))
    sections = _grouped_sections(method_labels, groups)

    cells: dict[tuple[str, str, str], dict] = {}
    for (strategy, ds), ranking in al.items():
        for entry in ranking:
            cells[(entry["method"], strategy, ds)] = entry
    best_per_col: dict[tuple[str, str], str] = {}
    for (strategy, ds), ranking in al.items():
        if ranking:
            best_per_col[(strategy, ds)] = max(ranking, key=lambda e: e["mean"])["method"]

    n_strats = len(strategies)
    n_data_cols = n_strats * len(datasets)
    n_total_cols = 2 + n_data_cols
    col_spec = "@{}l@{\\hspace{4pt}}l " + " ".join(["c" * n_strats] * len(datasets)) + "@{}"
    lines = [
        _PREAMBLE.rstrip(),
        "{",
        f"\\setlength{{\\tabcolsep}}{{{_TABCOLSEP_PT}pt}}",
        f"\\renewcommand{{\\arraystretch}}{{{_ARRAYSTRETCH}}}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Dataset spans across the strategy sub-columns.
    spans = r"\multicolumn{2}{@{}l}{}"
    cdash: list[str] = []
    col_idx = 3
    for ds in datasets:
        spans += f" & \\multicolumn{{{n_strats}}}{{c}}{{\\textbf{{{ds.upper().replace('_', '\\_')}}}}}"
        cdash.append(f"\\cdashline{{{col_idx}-{col_idx + n_strats - 1}}}")
        col_idx += n_strats
    lines.append(spans + r" \\")
    lines.append(" ".join(cdash))
    lines.append(r"\noalign{\vskip " + _HEADER_VSKIP + "}")

    sub_header = r"\multicolumn{2}{@{}l}{\textbf{Method}}"
    for _ in datasets:
        for strategy in strategies:
            sub_header += f" & \\textbf{{{strategy.replace('_', ' ').title()}}}"
    lines.append(sub_header + r" \\")
    lines.append(r"\hdashline\noalign{\vskip " + _HEADER_VSKIP + "}")

    def render_al_row(method: str) -> str:
        row = f"{_cite_cell(method, citations)} & {method_labels[method]}"
        for ds in datasets:
            for strategy in strategies:
                row += " & " + _format_cell(
                    cells.get((method, strategy, ds)),
                    decimals=decimals,
                    is_best=best_per_col.get((strategy, ds)) == method,
                )
        return row + r" \\"

    lines.extend(_emit_grouped_rows(sections, n_total_cols, render_al_row))
    lines += [r"\bottomrule", r"\end{tabular}", "}", ""]
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../plot_configs", config_name="paper_tables")
def main(cfg: DictConfig) -> dict[str, Path]:
    """Build combined SP, OOD, and AL LaTeX tables from existing ranking JSONs.

    Args:
        cfg: Hydra config. ``inputs`` is a directory or list of directories
            to scan; ``save_path`` is where the combined ``.tex`` files land.

    Returns:
        Mapping ``{"sp": Path, "ood": Path, "al": Path}`` for each table written.
    """
    inputs_resolved = OmegaConf.to_container(cfg.inputs, resolve=True) if cfg.get("inputs") is not None else None
    if inputs_resolved is None:
        msg = "`inputs` is required (directory or list of directories)."
        raise ValueError(msg)
    inputs = _normalize_inputs(inputs_resolved)
    out_dir = resolve_save_path(cfg.get("save_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    decimals = int(cfg.get("decimals", 3))
    citations_raw = cfg.get("citations") or {}
    if isinstance(citations_raw, DictConfig):
        citations_obj = OmegaConf.to_container(citations_raw, resolve=True)
    else:
        citations_obj = citations_raw
    citations: dict[str, str] = (
        {str(k): str(v) for k, v in citations_obj.items()} if isinstance(citations_obj, dict) else {}
    )

    groups_raw = cfg.get("groups")
    if groups_raw is None:
        groups_obj: object = []
    elif OmegaConf.is_config(groups_raw):
        groups_obj = OmegaConf.to_container(groups_raw, resolve=True)
    else:
        groups_obj = groups_raw
    groups: list[dict] = list(groups_obj) if isinstance(groups_obj, list) else []

    sp, ood, al = _load_rankings(inputs)
    written: dict[str, Path] = {}

    if sp:
        path = out_dir / "paper_table_sp.tex"
        path.write_text(_sp_table(sp, groups, citations, decimals))
        written["sp"] = path
        print(f"Wrote {path}  (datasets: {sorted(sp)})")
    else:
        print("No SP ranking JSONs found in inputs; skipping paper_table_sp.tex.")

    if al:
        path = out_dir / "paper_table_al.tex"
        path.write_text(_al_table(al, groups, citations, decimals))
        written["al"] = path
        ds_set = sorted({ds for _, ds in al})
        st_set = sorted({s for s, _ in al})
        print(f"Wrote {path}  (datasets: {ds_set} x strategies: {st_set})")
    else:
        print("No AL ranking JSONs found in inputs; skipping paper_table_al.tex.")

    if ood:
        path = out_dir / "paper_table_ood.tex"
        path.write_text(_ood_table(ood, groups, citations, decimals))
        written["ood"] = path
        ds_set = sorted({ds for _, ds in ood})
        print(f"Wrote {path}  (datasets: {ds_set})")
    else:
        print("No OOD ranking JSONs found in inputs; skipping paper_table_ood.tex.")

    return written


if __name__ == "__main__":
    main()
