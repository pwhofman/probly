"""Run the whole benchmark chain for one config (PIPE).

Run from ``experiments/llm-probly``::

    uv run python -m scripts.run_benchmark --config configs/qwen0.5b_triviaqa_5.yaml

Steps, in order. Each step is skipped if its output file already exists
(use ``--force`` to recompute the method results):

1. generate  -> data/<model>/<dataset>_<n>.jsonl + .meta.json  (scripts/generate.py)
2. label     -> adds "correct" to that file           (scripts/label.py)
3. methods   -> results/<model>/<dataset>/<method>.json, one per plugin in methods/
4. evaluate  -> AUROC, AURC, bootstrap interval per method (common/summary.py)
5. table     -> results/<model>/<dataset>/summary.md

See CONTRACT.md for the plugin interface.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from common.data import DATASETS, cache_path, load_jsonl
from common.io import RESULTS_ROOT, save_scores
from common.summary import summary_table, to_markdown
import pandas as pd
import yaml

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parent.parent  # experiments/llm-probly
METHODS_DIR = ROOT / "methods"
PRECISIONS = ("float32", "float16", "bfloat16")  # same choices as scripts/generate.py


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a YAML file in configs/")
    parser.add_argument("--methods", help="comma-separated method names; overrides the config")
    parser.add_argument("--skip-model-methods", action="store_true", help="skip plugins with NEEDS_MODEL = True")
    parser.add_argument("--force", action="store_true", help="recompute method results even if they exist")
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Read the YAML config and check that the required keys are there."""
    config = yaml.safe_load(Path(path).read_text())
    missing = [key for key in ("model", "dataset", "n") if key not in config]
    if missing:
        msg = f"{path}: missing keys {missing}"
        raise ValueError(msg)
    if config["dataset"] not in DATASETS:  # the datasets DATA registered in common/data.py
        msg = f"unknown dataset {config['dataset']!r}; known: {sorted(DATASETS)}"
        raise ValueError(msg)
    config.setdefault("precision", "float32")
    if config["precision"] not in PRECISIONS:
        msg = f"precision must be one of {PRECISIONS}, got {config['precision']!r}"
        raise ValueError(msg)
    config.setdefault("device", "cpu")
    config.setdefault("methods", "all")
    return config


def run_script(module: str, *args: str) -> None:
    """Run one of the existing scripts, e.g. scripts.generate, as its own process."""
    command = [sys.executable, "-m", module, *args]
    print("   $", " ".join(command[1:]))
    subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603 - fixed module names; check=True stops on failure


# ----- steps 1 and 2: data -------------------------------------------------


def prepare_data(config: dict) -> list[dict]:
    """Make sure the cache file exists and is labelled, then return its rows."""
    model, dataset, n = config["model"], config["dataset"], config["n"]
    path = cache_path(model, n, dataset)

    if path.exists():
        print(f"1. generate: skip, {path.relative_to(ROOT)} exists")
    else:
        print("1. generate")
        run_script(
            "scripts.generate",
            *("--model", model, "--dataset", dataset, "--n", str(n)),
            *("--precision", config["precision"], "--device", config["device"]),
        )

    rows = load_jsonl(path)
    if all("correct" in row for row in rows):
        print("2. label:    skip, already labelled")
    else:
        print("2. label")
        run_script("scripts.label", "--model", model, "--dataset", dataset, "--n", str(n))
        rows = load_jsonl(path)
    return rows


def cache_precision(config: dict) -> str:
    """Read the precision the cache was made with and check it matches the config.

    DATA found that precision changes the answers (bfloat16 -> float32 changed
    9 of 50 answers on 0.5B), so a cache made with another precision than the
    config asks for must not be used silently.
    """
    meta_path = cache_path(config["model"], config["n"], config["dataset"], suffix=".meta.json")
    if not meta_path.exists():
        print(f"   note: no {meta_path.name} (cache made before cache v2), precision unknown")
        return "unknown"
    used = json.loads(meta_path.read_text())["settings"]["precision"]
    if used != config["precision"]:
        msg = (
            f"{meta_path.name} says the cache was made with {used}, but the config asks for "
            f"{config['precision']}. Change the config, or delete the cache to regenerate it."
        )
        raise ValueError(
            msg,
        )
    return used


# ----- step 3: methods -----------------------------------------------------


def find_plugins() -> dict:
    """Import every file in methods/ and return {NAME: module}."""
    plugins = {}
    for file in sorted(METHODS_DIR.glob("*.py")):
        if file.name.startswith("_"):  # skip __init__.py and helper files
            continue
        module = importlib.import_module(f"methods.{file.stem}")
        for attr in ("NAME", "FAMILY", "NEEDS_MODEL", "score"):
            if not hasattr(module, attr):
                msg = f"methods/{file.name} has no {attr} (see CONTRACT.md)"
                raise AttributeError(msg)
        plugins[module.NAME] = module
    return plugins


def choose_plugins(plugins: dict, config: dict, args: argparse.Namespace) -> dict:
    """Keep only the methods this run should use."""
    wanted = args.methods.split(",") if args.methods else config["methods"]
    if wanted != "all":
        unknown = [name for name in wanted if name not in plugins]
        if unknown:
            msg = f"unknown methods {unknown}; available: {sorted(plugins)}"
            raise ValueError(msg)
        plugins = {name: plugins[name] for name in wanted}
    if args.skip_model_methods:
        plugins = {name: p for name, p in plugins.items() if not p.NEEDS_MODEL}
    return plugins


def load_llm(config: dict) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the LLM once for all plugins that need it."""
    # Imported here, not at the top: torch and transformers are slow to import
    # and not needed when no plugin uses the model.
    from scripts.generate import MODELS, PRECISIONS as TORCH_DTYPES  # noqa: PLC0415

    from probly.representer.sampler.huggingface import load_model  # noqa: PLC0415

    # Same precision as the cache, so methods see the model that wrote the answers.
    model, tok = load_model(MODELS[config["model"]], model_kwargs={"dtype": TORCH_DTYPES[config["precision"]]})
    model = model.to(config["device"])
    model.eval()
    return model, tok


def run_methods(plugins: dict, rows: list[dict], config: dict, run: str, force: bool) -> None:
    """Call score() for every plugin and save its record."""
    ids = [row["id"] for row in rows]
    todo = {name: p for name, p in plugins.items() if force or not (RESULTS_ROOT / run / f"{name}.json").exists()}
    for name in plugins:
        if name not in todo:
            print(f"3. {name}: skip, result exists")
    if not todo:
        return

    model = tok = None
    if any(p.NEEDS_MODEL for p in todo.values()):
        print("   loading the model ...")
        model, tok = load_llm(config)

    for name, plugin in todo.items():
        print(f"3. {name}")
        start = time.time()
        record = plugin.score(rows, model, tok) if plugin.NEEDS_MODEL else plugin.score(rows)
        path = save_scores(record, run=run, expected_ids=ids)  # validates against the contract
        print(f"   saved {path.relative_to(ROOT)} ({time.time() - start:.1f} s)")


# ----- steps 4 and 5: evaluate and table -------------------------------------


def make_table(rows: list[dict], plugins: dict, run: str, precision: str) -> None:
    """Evaluate every method of this run and write the summary table."""
    print("4. evaluate")
    table = summary_table(rows, run=run, methods=tuple(plugins))
    families = pd.Series([getattr(plugins.get(m), "FAMILY", "") for m in table["method"]], index=table.index)
    table.insert(1, "family", families)

    out = RESULTS_ROOT / run / "summary.md"
    header = f"Run `{run}`, {len(rows)} questions, precision {precision}\n\n"
    out.write_text(header + to_markdown(table))
    print(f"5. table:    {out.relative_to(ROOT)}\n")
    print(f"precision: {precision}")
    print(table.to_string(index=False))


def main() -> None:
    """Run all steps for one config."""
    args = parse_args()
    config = load_config(args.config)
    run = f"{config['model']}/{config['dataset']}_{config['n']}"  # results/<run>/
    print(f"run: {run}")

    rows = prepare_data(config)
    precision = cache_precision(config)
    plugins = choose_plugins(find_plugins(), config, args)
    run_methods(plugins, rows, config, run, args.force)
    make_table(rows, plugins, run, precision)


if __name__ == "__main__":
    main()
