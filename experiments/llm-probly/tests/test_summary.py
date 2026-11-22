from __future__ import annotations

import json

from common.summary import EXPECTED_METHODS, compare_models, find_models, method_by_model, summary_table, to_markdown
import numpy as np
import pandas as pd

ROWS = [{"id": i, "correct": i < 12} for i in range(50)]


def write(folder, name, ids, uncertainty, failed=None):
    record = {
        "method": name,
        "model": "m",
        "answer": "greedy",
        "ids": ids,
        "uncertainty": uncertainty,
        "failed": failed or [False] * len(ids),
        "seconds_per_question": 0.5,
    }
    (folder / f"{name}.json").write_text(json.dumps(record))


def test_missing_methods_are_pending(tmp_path):
    (tmp_path / "run").mkdir()
    table = summary_table(ROWS, run="run", results_root=tmp_path, n_boot=50)
    assert (table["status"].str.startswith("pending")).all()
    assert len(table) == len(EXPECTED_METHODS)


def test_valid_file_is_evaluated_and_bad_file_is_pending(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    ids = [r["id"] for r in ROWS]
    perfect = [0.0 if r["correct"] else 1.0 for r in ROWS]
    write(folder, "semantic_entropy", ids, perfect)
    write(folder, "discrete_se", ids[::-1], perfect)  # wrong id order
    table = summary_table(ROWS, run="run", results_root=tmp_path, n_boot=50).set_index("method")
    assert table.loc["semantic_entropy", "status"] == "ok"
    assert table.loc["semantic_entropy", "auroc"] == 1.0
    assert table.loc["discrete_se", "status"].startswith("pending")


def test_unknown_method_file_is_included(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    ids = [r["id"] for r in ROWS]
    write(folder, "my_new_method", ids, list(np.linspace(0, 1, 50)))
    table = summary_table(ROWS, run="run", results_root=tmp_path, n_boot=50)
    assert "my_new_method" in set(table["method"])


# ----- comparison across model sizes ------------------------------------------


def test_find_models_orders_small_to_large(tmp_path):
    for model in ("qwen3b", "qwen0.5b", "qwen1.5b"):
        (tmp_path / model / "triviaqa_200").mkdir(parents=True)
    (tmp_path / "qwen7b" / "mmlu_100").mkdir(parents=True)  # other dataset: ignored
    assert find_models("triviaqa_200", results_root=tmp_path) == ["qwen0.5b", "qwen1.5b", "qwen3b"]


def test_method_by_model_has_one_column_per_model(tmp_path):
    ids = [r["id"] for r in ROWS]
    perfect = [0.0 if r["correct"] else 1.0 for r in ROWS]
    for model in ("qwen0.5b", "qwen1.5b"):
        folder = tmp_path / model / "triviaqa_50"
        folder.mkdir(parents=True)
        write(folder, "perfect", ids, perfect)
        write(folder, "p_true_zeroshot_norm", ids, perfect)  # a probability method: gets ECE
    long = compare_models({"qwen0.5b": ROWS, "qwen1.5b": ROWS}, "triviaqa_50", n_boot=20, results_root=tmp_path)
    auroc = method_by_model(long, "auroc").set_index("method")
    assert list(auroc.columns) == ["qwen0.5b", "qwen1.5b"]
    assert (auroc.loc["perfect"] == 1.0).all()
    assert set(long["accuracy"]) == {12 / 50}
    ece = method_by_model(long, "ece")
    assert list(ece["method"]) == ["p_true_zeroshot_norm"]  # "perfect" is not a probability method


def test_to_markdown_leaves_nan_empty():
    text = to_markdown(pd.DataFrame({"method": ["a"], "auroc": [float("nan")]}))
    assert text.splitlines()[2] == "| a |  |"
