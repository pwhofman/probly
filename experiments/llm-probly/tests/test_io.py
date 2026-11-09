import math

from common import io as score_io

GOOD = {
    "method": "dummy",
    "model": "m",
    "answer": "greedy",
    "ids": [0, 1, 2],
    "uncertainty": [0.1, 0.9, 1.0],
    "failed": [False, False, True],
    "seconds_per_question": 0.5,
}


def test_valid_record_has_no_problems():
    assert score_io.validate_scores(GOOD, expected_ids=[0, 1, 2]) == []


def test_nan_is_rejected():
    bad = {**GOOD, "uncertainty": [0.1, math.nan, 1.0]}
    assert score_io.validate_scores(bad)


def test_failed_entry_must_be_one():
    bad = {**GOOD, "uncertainty": [0.1, 0.9, 0.3]}
    assert score_io.validate_scores(bad)


def test_wrong_id_order_is_rejected():
    assert score_io.validate_scores(GOOD, expected_ids=[0, 2, 1])


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(score_io, "RESULTS_ROOT", tmp_path)
    path = score_io.save_scores(GOOD, run="test")
    assert score_io.load_scores(path) == GOOD
