import json

from common.summary import sanity_table, sanity_verdict


def write(folder, method, model, auroc):
    record = {"method": method, "model": model, "auroc": auroc, "gap": 0.1, "n_distinct": 5}
    (folder / f"{method}_{model.split('/')[-1]}.json").write_text(json.dumps(record))


def test_verdict_thresholds():
    assert sanity_verdict(0.95) == "works"
    assert sanity_verdict(0.8) == "works"
    assert sanity_verdict(0.7) == "weak signal"
    assert sanity_verdict(0.4) == "random"


def test_sanity_table_has_one_row_per_file(tmp_path):
    write(tmp_path, "m1", "org/ModelA", 0.9)
    write(tmp_path, "m1", "org/ModelB", 0.5)
    table = sanity_table(tmp_path)
    assert len(table) == 2
    assert set(table["model"]) == {"ModelA", "ModelB"}
    assert table.set_index("model").loc["ModelB", "verdict"] == "random"


def test_files_without_sanity_fields_are_skipped(tmp_path):
    (tmp_path / "other.json").write_text(json.dumps({"something": 1}))
    assert sanity_table(tmp_path).empty


def test_missing_folder_gives_empty_table(tmp_path):
    assert sanity_table(tmp_path / "does_not_exist").empty
