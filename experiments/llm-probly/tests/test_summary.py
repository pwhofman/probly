import json

import numpy as np

from common.summary import EXPECTED_METHODS, summary_table

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
