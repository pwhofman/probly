"""Shared LaTeX layout helpers for paper-ready booktabs tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

PREAMBLE = (
    "% Requires in your preamble:\n"
    "%   \\usepackage{booktabs}\n"
    "%   \\usepackage{arydshln}             % for \\hdashline / \\cdashline\n"
    "%   \\usepackage[table]{xcolor}        % for \\rowcolor on group-header rows\n"
)
TABCOLSEP_PT = 8
ARRAYSTRETCH = "1.15"
HEADER_VSKIP = "0.45em"
TABLE_CLOSE: tuple[str, ...] = (r"\end{tabular}", r"\end{table}", "")


def table_open(*, title: str, label_key: str, tabcolsep_pt: int = TABCOLSEP_PT) -> list[str]:
    """LaTeX preamble for a numbered table float (caption + label + spacing)."""
    return [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{title}}}",
        f"\\label{{tab:{label_key}}}",
        r"\footnotesize",
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep_pt}pt}}",
        f"\\renewcommand{{\\arraystretch}}{{{ARRAYSTRETCH}}}",
    ]


def strip_leading_zero(value: float, decimals: int) -> str:
    """``0.918`` -> ``.918``; ``1.000`` stays ``1.000``."""
    text = f"{value:.{decimals}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def format_cell(entry: dict | None, *, decimals: int, is_best: bool, mark: str | None = None) -> str:
    r"""Render one ``$mean_{\pm std}$`` cell, bolded when best, ``--`` when missing."""
    if entry is None:
        return "--"
    mean = strip_leading_zero(entry["mean"], decimals)
    std = strip_leading_zero(entry["std"], decimals)
    body = f"{mean}_{{\\pm {std}}}"
    if mark:
        # The value comes from another notion than the column's; the mark says which.
        body += f"^{{\\mathrm{{{mark}}}}}"
    return f"$\\mathbf{{{body}}}$" if is_best else f"${body}$"


def collect_method_labels(rankings: list[list[dict]]) -> dict[str, str]:
    """Map every method appearing in any ranking to its label."""
    out: dict[str, str] = {}
    for ranking in rankings:
        for entry in ranking:
            out.setdefault(entry["method"], entry["label"])
    return out


def grouped_sections(
    method_labels: dict[str, str],
    groups: list[dict],
) -> list[tuple[str | None, list[str]]]:
    """Materialize ``GROUPS`` into ``(name, [methods])`` pairs.

    Methods present in the data but not mentioned in any group are appended in
    their own ``"Other"`` section. When ``groups`` is empty, returns a single
    nameless section with all methods alphabetically sorted.
    """
    if not groups:
        return [(None, sorted(method_labels))]
    used: set[str] = set()
    sections: list[tuple[str | None, list[str]]] = []
    for group in groups:
        name = str(group.get("name", "")) or None
        members = [m for m in group.get("methods", []) if m in method_labels and m not in used]
        used.update(members)
        if members:
            sections.append((name, members))
    leftover = sorted(m for m in method_labels if m not in used)
    if leftover:
        sections.append(("Other", leftover))
    return sections


def group_header_row(name: str, n_cols: int) -> list[str]:
    """Two-line booktabs-style group header (gray band + small extra space)."""
    return [
        r"\addlinespace[0.4em]",
        r"\rowcolor[gray]{0.92}[0pt][0pt]",
        f"\\multicolumn{{{n_cols}}}{{@{{}}l@{{}}}}{{\\textbf{{{name}}}}} \\\\",
        r"\addlinespace[0.15em]",
    ]


def emit_grouped_rows(
    sections: list[tuple[str | None, list[str]]],
    n_total_cols: int,
    row_renderer: Callable[[str], str],
) -> list[str]:
    """Emit grouped section headers + per-method data rows."""
    out: list[str] = []
    for section_idx, (section_name, section_methods) in enumerate(sections):
        if section_name is not None:
            header = group_header_row(section_name, n_total_cols)
            out.extend(header[1:] if section_idx == 0 else header)
        out.extend(row_renderer(method) for method in section_methods)
    return out
