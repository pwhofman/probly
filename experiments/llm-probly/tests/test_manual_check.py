"""Tests for splitting the manual check and the weighted label error rate."""

from __future__ import annotations

import pytest

from scripts.manual_check import label_error_rate, part_rows


def make_rows(n: int = 200, n_correct: int = 40) -> list[dict]:
    return [{"id": i, "correct": i < n_correct} for i in range(n)]


def test_parts_do_not_overlap_and_cover_thirty_rows() -> None:
    rows = make_rows()
    one, two = part_rows(rows, 1, seed=0), part_rows(rows, 2, seed=0)
    assert len(one) == len(two) == 15
    assert not {r["id"] for r in one} & {r["id"] for r in two}


def test_parts_mix_both_labels() -> None:
    rows = make_rows()
    for part in (1, 2):
        labels = {r["correct"] for r in part_rows(rows, part, seed=0)}
        assert labels == {True, False}


def test_label_error_rate_is_weighted_by_label_share() -> None:
    # 1 of 2 "correct" labels is wrong (50 %), 0 of 2 "wrong" labels is wrong (0 %)
    items = [
        {"auto": True, "manual": True},
        {"auto": True, "manual": False},
        {"auto": False, "manual": False},
        {"auto": False, "manual": False},
    ]
    result = label_error_rate(items, share_auto_correct=0.2)
    assert result["error_when_labelled_correct"] == pytest.approx(0.5)
    assert result["error_when_labelled_wrong"] == pytest.approx(0.0)
    assert result["label_error_rate"] == pytest.approx(0.1)
