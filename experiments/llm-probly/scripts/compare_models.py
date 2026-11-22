"""Compare all model sizes for one dataset: methods x models (PIPE).

Run from ``experiments/llm-probly`` after ``run_benchmark`` has run for each model::

    uv run python -m scripts.compare_models --dataset triviaqa --n 200

Looks for every ``results/<model>/<dataset>_<n>/`` folder, evaluates all methods
in it against that model's labels, and writes
``results/comparison_<dataset>_<n>.md`` with one table per metric
(rows = methods, columns = model sizes from small to large).
"""

from __future__ import annotations

import argparse
import json

from common.data import cache_path, load_jsonl
from common.io import RESULTS_ROOT
from common.summary import compare_models, find_models, method_by_model, to_markdown

METRICS = {
    "auroc": "AUROC (higher is better, 0.5 = random)",
    "prr": "PRR (higher is better, 1 = perfect, 0 = random)",
    "aurc": "AURC (lower is better)",
    "ece": "ECE, only probability methods (lower is better)",
}


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="triviaqa")
    parser.add_argument("--n", type=int, default=200, help="number of questions")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap samples for the AUROC interval")
    return parser.parse_args()


def precision_of(model: str, n: int, dataset: str) -> str:
    """Precision from the cache's meta file, or "unknown" for old caches."""
    meta = cache_path(model, n, dataset, suffix=".meta.json")
    return json.loads(meta.read_text())["settings"]["precision"] if meta.exists() else "unknown"


def main() -> None:
    """Build and save the comparison."""
    args = parse_args()
    run_name = f"{args.dataset}_{args.n}"
    models = find_models(run_name)
    if not models:
        msg = f"no results/<model>/{run_name}/ folders yet; run run_benchmark first"
        raise SystemExit(msg)

    rows_by_model = {m: load_jsonl(cache_path(m, args.n, args.dataset)) for m in models}
    long = compare_models(rows_by_model, run_name, n_boot=args.n_boot)

    # One line per model: how many questions, how often it was right, which precision.
    info = long.groupby("model", sort=False)["accuracy"].first().reset_index()
    info.insert(1, "questions", args.n)
    info["precision"] = [precision_of(m, args.n, args.dataset) for m in info["model"]]

    parts = [f"# {args.dataset}, {args.n} questions: methods x model size\n", "## Models\n", to_markdown(info)]
    for metric, title in METRICS.items():
        wide = method_by_model(long, metric)
        if not wide.empty:  # e.g. no ECE section while there are no probability methods
            parts += [f"\n## {title}\n", to_markdown(wide)]
    out = RESULTS_ROOT / f"comparison_{run_name}.md"
    out.write_text("\n".join(parts))

    print(info.to_string(index=False), "\n")
    print("AUROC")
    print(method_by_model(long, "auroc").to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
