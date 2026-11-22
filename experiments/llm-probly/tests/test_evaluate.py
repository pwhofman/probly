from __future__ import annotations

from common.evaluate import _rejection_area, aurc, auroc, bootstrap_auroc, ece, evaluate, kept_accuracy_curve, prr
import numpy as np
import pytest

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
    assert set(result) == {"auroc", "aurc", "prr"}


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="equally long"):
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


# ----- PRR -------------------------------------------------------------------


def lm_polygraph_rejection_area(estimator, target):
    """Copy of lm-polygraph's PredictionRejectionArea.__call__ (max_rejection=1, MIT licence).

    Source: lm_polygraph/ue_metrics/pred_rej_area.py. Kept here as the reference
    our implementation must match.
    """
    ue = np.array(estimator)
    num_obs = len(ue)
    num_rej = num_obs
    ue_argsort = np.argsort(ue)
    sorted_metrics = np.array(target)[ue_argsort]
    cumsum = np.cumsum(sorted_metrics)[-num_rej:]
    scores = (cumsum / np.arange((num_obs - num_rej) + 1, num_obs + 1))[::-1]
    return np.sum(scores) / num_rej


def test_rejection_area_matches_lm_polygraph():
    rng = np.random.default_rng(1)
    correct = rng.random(200) < 0.3
    u = rng.random(200)  # continuous, so there are no ties
    quality = correct.astype(float)
    assert np.isclose(_rejection_area(u, quality), lm_polygraph_rejection_area(u, quality))


def test_prr_matches_lm_polygraph_normalized():
    rng = np.random.default_rng(2)
    correct = rng.random(200) < 0.3
    u = (~correct).astype(float) + rng.normal(0, 0.8, 200)  # some signal, no ties
    quality = correct.astype(float)
    method = lm_polygraph_rejection_area(u, quality)
    oracle = lm_polygraph_rejection_area(-quality, quality)
    shuffles = [lm_polygraph_rejection_area(rng.permutation(200), quality) for _ in range(1000)]
    expected = (method - np.mean(shuffles)) / (oracle - np.mean(shuffles))
    assert abs(prr(u, correct) - expected) < 0.02  # lm-polygraph's random area is sampled


def test_perfect_score_has_prr_one():
    assert np.isclose(prr((~CORRECT).astype(float), CORRECT), 1.0)


def test_random_score_has_prr_near_zero():
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.24
    assert abs(prr(rng.random(2000), correct)) < 0.1


def test_inverted_score_has_negative_prr():
    assert prr(CORRECT.astype(float), CORRECT) < -0.5  # worst order: correct answers rejected first


def test_constant_score_has_prr_near_zero():
    assert abs(prr(np.zeros(50), CORRECT)) < 0.1


def test_prr_is_nan_when_all_answers_are_correct():
    assert np.isnan(prr([0.1, 0.5, 0.9], [True, True, True]))


# ----- ECE -------------------------------------------------------------------


def lm_polygraph_ece(estimator, target, n_bins):
    """Copy of lm-polygraph's ECE.__call__ (normalize=False, MIT licence).

    Source: lm_polygraph/ue_metrics/ece.py. lm-polygraph passes -confidence as
    the estimator, so confidences = -estimator.
    """
    confidences = -np.asarray(estimator)
    target = np.asarray(target)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    result, n = 0.0, len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not np.any(in_bin):
            continue
        result += (np.sum(in_bin) / n) * abs(np.mean(target[in_bin]) - np.mean(confidences[in_bin]))
    return result


@pytest.mark.parametrize("n_bins", [10, 20])
def test_ece_matches_lm_polygraph(n_bins):
    rng = np.random.default_rng(3)
    confidence = rng.random(300)
    correct = rng.random(300) < confidence  # roughly calibrated
    expected = lm_polygraph_ece(-confidence, correct.astype(float), n_bins)
    assert np.isclose(ece(1 - confidence, correct, n_bins=n_bins), expected)


def test_calibrated_scores_have_small_ece():
    rng = np.random.default_rng(4)
    confidence = rng.random(5000)
    correct = rng.random(5000) < confidence  # right exactly as often as the confidence says
    assert ece(1 - confidence, correct) < 0.03


def test_overconfident_scores_have_large_ece():
    # always 95 % sure, but only 24 % right
    assert np.isclose(ece(np.full(50, 0.05), CORRECT), 0.95 - 0.24)


def test_ece_rejects_scores_outside_zero_one():
    with pytest.raises(ValueError, match=r"values in \[0, 1\]"):
        ece([1.5, 0.2], [True, False])


# ----- rejection curve ---------------------------------------------------------


def test_kept_accuracy_curve_mean_is_the_prr_area():
    rng = np.random.default_rng(5)
    correct = rng.random(100) < 0.4
    u = rng.random(100)
    rejected, accuracy = kept_accuracy_curve(u, correct)
    assert rejected[0] == 0.0
    assert len(accuracy) == 100
    assert np.isclose(accuracy[0], correct.mean())  # nothing rejected: overall accuracy
    assert np.isclose(accuracy.mean(), _rejection_area(u, correct.astype(float)))


def test_perfect_kept_accuracy_curve_reaches_full_accuracy():
    _, accuracy = kept_accuracy_curve((~CORRECT).astype(float), CORRECT)
    assert accuracy[-1] == 1.0  # only the most confident (a correct) answer is left
    assert np.all(np.diff(accuracy) >= 0)  # never drops when wrong answers go first
