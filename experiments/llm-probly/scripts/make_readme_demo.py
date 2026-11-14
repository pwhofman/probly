"""Generate the animated README demo (light and dark SVG) from saved results.

The script only reads the cache rows and the score files, no model runs. The style follows
probly's own ``docs/source/_static/readme/llm_uncertainty_demo_*.svg``: plain SVG with SMIL
``<animate>`` on opacity, one loop, no JavaScript.

Usage (from ``experiments/llm-probly``)::

    uv run python scripts/make_readme_demo.py
    uv run python scripts/make_readme_demo.py --ids 48 49 10 --run qwen0.5b --n 50
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import load_data  # noqa: E402
from common.io import RESULTS_ROOT, load_scores  # noqa: E402

WIDTH, HEIGHT = 900, 470
CHAR_W = 7.2  # width of one character at font-size 12 (monospace)
CHAT_LEFT, CHAT_RIGHT = 28, 570
PANEL_X, PANEL_W = 600, 272

# (file name in results/<run>/, label in the panel)
METHODS = [
    ("p_true_zeroshot_norm", "p(True) zero-shot"),
    ("p_true_brainstorm", "p(True) brainstorm"),
    ("likelihood_mean", "likelihood (mean)"),
    ("semantic_entropy", "semantic entropy"),
    ("discrete_se", "discrete SE"),
]

GOOD, BAD, MID = "#1e88e5", "#ff0d57", "#f5a623"  # same blue and red as probly's demo

THEMES = {
    "light": {
        "bg1": "#ffffff", "bg2": "#f7f8fa", "border": "#cccccc", "text": "#1a1a1a", "muted": "#708090",
        "faint": "#9aa4b0", "bubble": "#eef0f3", "bubble_line": "#d8dce1", "track": "#e7e9ec",
        "panel": "#f4f5f7", "dot1": "#d3d7dc", "dot2": "#dfe2e6", "dot3": "#e7e9ec",
    },
    "dark": {
        "bg1": "#0d1117", "bg2": "#161b22", "border": "#30363d", "text": "#e6edf3", "muted": "#8b949e",
        "faint": "#6e7681", "bubble": "#21262d", "bubble_line": "#30363d", "track": "#30363d",
        "panel": "#161b22", "dot1": "#484f58", "dot2": "#3b424a", "dot3": "#30363d",
    },
}


@dataclass
class Item:
    """One SVG element that is switched on at ``t_on`` and off at ``t_off`` (seconds)."""

    tag: str
    attrs: str
    inner: str
    t_on: float
    t_off: float
    attr: str = "opacity"
    on: float = 1.0
    rise: float = 0.02  # seconds the change takes (bars grow slowly, text appears at once)


def _key(value: float) -> str:
    return f"{value:.5f}".rstrip("0").rstrip(".") or "0"


def render_item(item: Item, total: float) -> str:
    """Render an item with its ``<animate>`` child for a loop of ``total`` seconds."""
    fall = 0.3
    a = item.t_on / total
    b = (item.t_on + item.rise) / total
    c = item.t_off / total
    d = (item.t_off + fall) / total
    keys = ";".join(_key(x) for x in (0, a, b, c, d, 1))
    on, off = _key(item.on), "0"
    values = ";".join((off, off, on, on, off, off))
    animate = (
        f'<animate attributeName="{item.attr}" values="{values}" keyTimes="{keys}" '
        f'dur="{_key(total)}s" repeatCount="indefinite"/>'
    )
    return f'<{item.tag} {item.attrs} {item.attr}="0">{item.inner}{animate}</{item.tag}>'


def typed(text_lines: list[str], x: float, y: float, step: float, t_start: float, t_off: float, theme: dict, size=12):
    """Per-character typing effect (like probly's demo). Returns items and the end time."""
    items: list[Item] = []
    t = t_start
    for row, line in enumerate(text_lines):
        for col, char in enumerate(line):
            if char != " ":
                attrs = f'x="{x + CHAR_W * col:.1f}" y="{y + 18 * row:.0f}" font-size="{size}" fill="{theme["text"]}"'
                items.append(Item("text", attrs, escape(char), t, t_off))
            t += step
    return items, t


def clean(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def percentile(values: np.ndarray, value: float) -> float:
    """Share of questions with a lower uncertainty (ties count half)."""
    return float((np.sum(values < value) + 0.5 * np.sum(values == value)) / len(values))


def scene_items(scene: dict, t0: float, theme: dict) -> tuple[list[Item], float]:
    """All animated elements of one question. Returns them and the end time of the scene."""
    items: list[Item] = []
    n = scene["n"]
    # timing is relative to t0, the end is known only at the end, so use a placeholder first
    t_off = 1e9
    question = textwrap.wrap(clean(scene["question"], 110), 50) or [""]
    answer = textwrap.wrap(clean(scene["answer"], 160), 62) or [""]

    # user bubble
    width = max(len(line) for line in question) * CHAR_W + 28
    bx = CHAT_RIGHT - width
    height = 30 + 18 * (len(question) - 1)
    t = t0 + 0.4
    items.append(Item("text", f'x="{bx + 4:.1f}" y="66" font-size="10" fill="{theme["muted"]}"', "you", t, t_off))
    items.append(Item("rect", f'x="{bx:.1f}" y="74" width="{width:.1f}" height="{height}" rx="12" fill="{theme["bubble"]}" stroke="{theme["bubble_line"]}"', "", t, t_off))
    typing, t = typed(question, bx + 14, 94, 0.04, t + 0.3, t_off, theme)
    items += typing

    # assistant: three pulsing dots, then the answer
    ya = 74 + height + 30
    t += 0.3
    dots = "".join(
        f'<circle cx="{36 + 14 * i}" cy="{ya + 24}" r="3.5" fill="{theme["muted"]}">'
        f'<animate attributeName="opacity" values="0.25;1;0.25" dur="0.9s" begin="{0.3 * i}s" repeatCount="indefinite"/></circle>'
        for i in range(3)
    )
    items.append(Item("g", "", dots, t, t + 1.0))
    t += 1.0
    items.append(Item("text", f'x="{CHAT_LEFT}" y="{ya}" font-size="10" fill="{theme["muted"]}"', "assistant", t, t_off))
    typing, t = typed(answer, CHAT_LEFT, ya + 28, 0.03, t + 0.1, t_off, theme)
    items += typing

    # panel rows: one per method, the bar shows the rank among all questions
    t += 0.5
    pcts = []
    y = 136
    for _name, label, uncertainty, all_values in scene["scores"]:
        pct = percentile(all_values, uncertainty)
        rank = 1 + int(np.sum(all_values > uncertainty))
        color = BAD if pct >= 0.67 else GOOD if pct <= 0.33 else MID
        pcts.append(pct)
        items.append(Item("text", f'x="{PANEL_X + 16}" y="{y}" font-size="11" fill="{theme["text"]}"', escape(label), t, t_off))
        items.append(Item("text", f'x="{PANEL_X + PANEL_W - 16}" y="{y}" font-size="10" text-anchor="end" fill="{theme["muted"]}"', f"{uncertainty:.2f} · rank {rank}/{n}", t, t_off))
        items.append(Item("rect", f'x="{PANEL_X + 16}" y="{y + 8}" width="{PANEL_W - 32}" height="9" rx="4.5" fill="{theme["track"]}"', "", t, t_off))
        items.append(Item("rect", f'x="{PANEL_X + 16}" y="{y + 8}" height="9" rx="4.5" fill="{color}"', "", t, t_off, attr="width", on=(PANEL_W - 32) * max(pct, 0.03), rise=0.6))
        y += 46
        t += 0.9

    # verdict (from the average rank) and what was really the case
    mean_pct = float(np.mean(pcts)) if pcts else 0.5
    if mean_pct >= 0.67:
        verdict, color = "⚠ high uncertainty", BAD
    elif mean_pct <= 0.33:
        verdict, color = "✓ low uncertainty", GOOD
    else:
        verdict, color = "• medium uncertainty", MID
    items.append(Item("text", f'x="{PANEL_X + 16}" y="{y + 12}" font-size="12" font-weight="bold" fill="{color}"', verdict, t, t_off))
    t += 1.0
    truth = "correct ✓" if scene["correct"] else "wrong ✗"
    items.append(Item("text", f'x="{PANEL_X + 16}" y="{y + 34}" font-size="11" fill="{theme["muted"]}"', f"actual answer: {truth}", t, t_off))
    t_end = t + 2.8

    for item in items:
        if item.t_off == t_off:
            item.t_off = t_end - 0.3
    return items, t_end


def build_svg(scenes: list[dict], theme_name: str, footer: str) -> str:
    theme = THEMES[theme_name]
    items: list[Item] = []
    t = 0.0
    for scene in scenes:
        scene_list, t = scene_items(scene, t, theme)
        items += scene_list
    total = t + 0.5

    static = [
        f'<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{theme["bg1"]}"/><stop offset="1" stop-color="{theme["bg2"]}"/></linearGradient></defs>',
        f'<rect x="1" y="1" width="{WIDTH - 2}" height="{HEIGHT - 2}" rx="14" fill="url(#bg)" stroke="{theme["border"]}" stroke-width="1.5"/>',
        f'<circle cx="28" cy="30" r="4.5" fill="{theme["dot1"]}"/><circle cx="46" cy="30" r="4.5" fill="{theme["dot2"]}"/><circle cx="64" cy="30" r="4.5" fill="{theme["dot3"]}"/>',
        f'<text x="{CHAT_RIGHT - 160}" y="34" font-size="10" fill="{theme["muted"]}">chat · monitored by probly</text>',
        f'<rect x="{PANEL_X}" y="50" width="{PANEL_W}" height="{HEIGHT - 100}" rx="12" fill="{theme["panel"]}" stroke="{theme["bubble_line"]}"/>',
        f'<text x="{PANEL_X + 16}" y="76" font-size="11" font-weight="bold" fill="{theme["text"]}">uncertainty scores</text>',
        f'<text x="{PANEL_X + 16}" y="93" font-size="9" fill="{theme["faint"]}">bar = share of questions less uncertain</text>',
        f'<text x="{PANEL_X + 16}" y="105" font-size="9" fill="{theme["faint"]}">rank 1 = the most uncertain question</text>',
        f'<text x="28" y="{HEIGHT - 20}" font-size="10" fill="{theme["faint"]}">{escape(footer)}</text>',
    ]
    body = "".join(render_item(item, total) for item in items)
    label = "Animated demo: uncertainty scores of several methods for three real questions answered by a small language model"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" role="img" aria-label="{label}">'
        + "".join(static)
        + body
        + "</svg>\n"
    )


def load_method_scores(run: str, row_ids: list[int]) -> dict:
    """Read every available score file. Returns name -> (label, uncertainty by id, all values)."""
    scores = {}
    for name, label in METHODS:
        path = RESULTS_ROOT / run / f"{name}.json"
        try:
            record = load_scores(path, expected_ids=row_ids)
        except (FileNotFoundError, ValueError) as error:
            print(f"skip {name}: {error}")
            continue
        per_id = dict(zip(record["ids"], record["uncertainty"], strict=True))
        scores[name] = (label, per_id, np.asarray(record["uncertainty"], dtype=float))
    if not scores:
        msg = "no score files found"
        raise SystemExit(msg)
    return scores


def collect_scenes(run: str, n: int, ids: list[int]) -> tuple[list[dict], str]:
    """Read cache rows and score files and build one scene per question id."""
    rows = load_data(run, n)
    row_ids = [row["id"] for row in rows]
    by_id = {row["id"]: row for row in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        msg = f"ids not in the data: {missing}"
        raise SystemExit(msg)
    scores = load_method_scores(run, row_ids)

    scenes = []
    for question_id in ids:
        row = by_id[question_id]
        scene = {
            "id": question_id,
            "question": row["question"],
            "answer": row["greedy_answer"],
            "correct": bool(row["correct"]),
            "n": len(rows),
            "scores": [(name, label, per_id[question_id], values) for name, (label, per_id, values) in scores.items()],
        }
        scenes.append(scene)
        shown = ", ".join(f"{label} {unc:.2f}" for _, label, unc, _ in scene["scores"])
        print(f"id {question_id}: correct={scene['correct']} | {clean(scene['question'], 60)} -> {clean(scene['answer'], 40)} | {shown}")
    model = rows[0].get("model", run)
    return scenes, f"{model} · TriviaQA · {len(rows)} questions · scores read from saved results"


def list_questions(run: str, n: int, limit: int = 15) -> None:
    """Print candidate questions with the rank of each method (0 = least, 1 = most uncertain).

    Wrong answers with a low mean rank are the ones the methods miss; correct answers with a
    high mean rank are false alarms.
    """
    rows = load_data(run, n)
    scores = load_method_scores(run, [row["id"] for row in rows])
    labels = [label for label, _, _ in scores.values()]
    entries = []
    for row in rows:
        ranks = [percentile(values, per_id[row["id"]]) for _, per_id, values in scores.values()]
        entries.append((float(np.mean(ranks)), row, ranks))

    def show(title: str, selected: list) -> None:
        print(f"\n{title}\n  columns: mean rank | " + " | ".join(labels))
        for mean, row, ranks in selected[:limit]:
            cols = " ".join(f"{r:.2f}" for r in ranks)
            print(f"  id {row['id']:>3}  {mean:.2f} | {cols} | {clean(row['question'], 70)} -> {clean(row['greedy_answer'], 28)}")

    wrong = sorted((e for e in entries if not e[1]["correct"]), key=lambda e: e[0])
    right = sorted((e for e in entries if e[1]["correct"]), key=lambda e: -e[0])
    show("WRONG answers the methods miss (low rank = confident):", wrong)
    show("WRONG answers the methods flag (high rank):", wrong[::-1])
    show("CORRECT answers with false alarms (high rank):", right)
    show("CORRECT answers that look certain (low rank):", right[::-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="qwen0.5b")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--ids", type=int, nargs="+", default=[48, 49, 10])
    parser.add_argument("--out", type=Path, default=ROOT / "demo")
    parser.add_argument("--list", action="store_true", help="list candidate questions with the rank of each method")
    args = parser.parse_args()

    if args.list:
        list_questions(args.run, args.n)
        return

    scenes, footer = collect_scenes(args.run, args.n, args.ids)
    args.out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = args.out / f"llm_uncertainty_demo_{theme}.svg"
        path.write_text(build_svg(scenes, theme, footer), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
