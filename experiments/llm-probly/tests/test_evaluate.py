import numpy as np
import pytest

from common.evaluate import aurc, auroc, bootstrap_auroc, evaluate

CORRECT = np.array([True] * 12 + [False] * 38)


def test_perfect_score_has_auroc_one():
    assert auroc((~CORRECT).astype(float), CORRECT) == 1.0


def test_inverted_score_has_auroc_zero():
    assert auroc(CORRECT.astype(float), CORRECT) == 0.0


def test_random_score_has_auroc_near_half():
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.24
    assert abs(auroc(rng.random(2000), correct) - 0.5) < 0.05


def test_perfect_score_has_much_lower_aurc_than_random_score():
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.24
    perfect = aurc((~correct).astype(float), correct)
    random_score = aurc(rng.random(2000), correct)
    assert perfect < random_score - 0.1


def test_aurc_of_random_score_is_close_to_error_rate():
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.24
    error_rate = 1 - correct.mean()
    assert abs(aurc(rng.random(2000), correct) - error_rate) < 0.05


def test_ties_do_not_depend_on_label_order():
    wrong_first = np.array([False] * 38 + [True] * 12)
    wrong_last = wrong_first[::-1]
    a = aurc(np.zeros(50), wrong_first)
    b = aurc(np.zeros(50), wrong_last)
    assert abs(a - b) < 0.08
    assert abs(a - 0.76) < 0.08


def test_evaluate_returns_both_metrics():
    result = evaluate((~CORRECT).astype(float), CORRECT)
    assert set(result) == {"auroc", "aurc"}


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        auroc([0.1, 0.2], [True, False, True])


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    correct = rng.random(200) < 0.5
    u = (~correct).astype(float) + rng.normal(0, 0.5, 200)
    lo, hi = bootstrap_auroc(u, correct, n_boot=200)
    assert lo < auroc(u, correct) < hi


def test_bootstrap_skips_single_class_resamples():
    lo, hi = bootstrap_auroc([0.9, 0.1, 0.2, 0.3, 0.4], [False, True, True, True, True], n_boot=200)
    assert 0.0 <= lo <= hi <= 1.0
