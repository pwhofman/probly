"""Tests for conformal nonconformity-score callables."""

from __future__ import annotations

import numpy as np
import pytest

from probly.conformal_scores import (
    APSScore,
    NonConformityScore,
    RAPSScore,
    SAPSScore,
    aps_score,
    cqr_score,
    lac_score,
    raps_score,
    saps_score,
)
from probly.conformal_scores.aps._common import compute_aps_score_numpy
from probly.conformal_scores.raps._common import compute_raps_score_numpy
from probly.conformal_scores.saps._common import compute_saps_score_func_numpy
from probly.conformal_scores.uacqr._common import compute_uacqr_score_func_numpy


def test_non_conformity_score_protocol_accepts_functions_and_callables() -> None:
    assert isinstance(lac_score, NonConformityScore)
    assert isinstance(cqr_score, NonConformityScore)
    assert isinstance(APSScore(), NonConformityScore)


def test_aps_score_class_curries_configuration() -> None:
    probs = np.array([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]], dtype=float)
    labels = np.array([1, 2], dtype=int)

    score = APSScore(randomized=False)
    expected = compute_aps_score_numpy(probs, labels, randomized=False)
    np.testing.assert_allclose(score(probs, labels), expected)

    with pytest.raises(TypeError):
        aps_score(probs, labels, randomized=False)


def test_saps_score_class_curries_configuration() -> None:
    probs = np.array([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]], dtype=float)
    labels = np.array([0, 2], dtype=int)

    score = SAPSScore(randomized=False, lambda_val=0.3)
    expected = compute_saps_score_func_numpy(probs, labels, randomized=False, lambda_val=0.3)
    np.testing.assert_allclose(score(probs, labels), expected)

    with pytest.raises(TypeError):
        saps_score(probs, labels, lambda_val=0.3)


def test_raps_score_class_curries_configuration() -> None:
    probs = np.array([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]], dtype=float)
    labels = np.array([0, 1], dtype=int)

    score = RAPSScore(randomized=False, lambda_reg=0.2, k_reg=1, epsilon=0.05)
    expected = compute_raps_score_numpy(
        probs,
        labels,
        randomized=False,
        lambda_reg=0.2,
        k_reg=1,
        epsilon=0.05,
    )
    np.testing.assert_allclose(score(probs, labels), expected)

    with pytest.raises(TypeError):
        raps_score(probs, labels, lambda_reg=0.2)


def test_classification_scores_support_multi_axis_batch_shapes() -> None:
    probs = np.array(
        [
            [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]],
            [[0.6, 0.2, 0.2], [0.4, 0.3, 0.3]],
        ],
        dtype=float,
    )
    labels = np.array([[1, 0], [2, 1]], dtype=int)

    flat_probs = probs.reshape(-1, probs.shape[-1])
    flat_labels = labels.reshape(-1)

    aps_scores = compute_aps_score_numpy(probs, labels, randomized=False)
    aps_flat = compute_aps_score_numpy(flat_probs, flat_labels, randomized=False).reshape(labels.shape)
    np.testing.assert_allclose(aps_scores, aps_flat)

    lac_scores = lac_score(probs, labels)
    lac_flat = lac_score(flat_probs, flat_labels).reshape(labels.shape)
    np.testing.assert_allclose(lac_scores, lac_flat)

    saps_scores = compute_saps_score_func_numpy(probs, labels, randomized=False, lambda_val=0.3)
    saps_flat = compute_saps_score_func_numpy(flat_probs, flat_labels, randomized=False, lambda_val=0.3).reshape(
        labels.shape
    )
    np.testing.assert_allclose(saps_scores, saps_flat)

    raps_scores = compute_raps_score_numpy(
        probs,
        labels,
        randomized=False,
        lambda_reg=0.2,
        k_reg=1,
        epsilon=0.05,
    )
    raps_flat = compute_raps_score_numpy(
        flat_probs,
        flat_labels,
        randomized=False,
        lambda_reg=0.2,
        k_reg=1,
        epsilon=0.05,
    ).reshape(labels.shape)
    np.testing.assert_allclose(raps_scores, raps_flat)


def test_uacqr_score_supports_multi_axis_batch_shapes() -> None:
    y_true = np.array([[0.3, 0.7], [0.4, 0.5]], dtype=float)
    y_pred = np.array(
        [
            [[[0.2, 0.5], [0.4, 0.8]], [[0.3, 0.6], [0.4, 0.7]]],
            [[[0.1, 0.4], [0.5, 0.9]], [[0.2, 0.7], [0.3, 0.8]]],
            [[[0.2, 0.6], [0.3, 1.0]], [[0.4, 0.8], [0.2, 0.6]]],
        ],
        dtype=float,
    )

    scores = compute_uacqr_score_func_numpy(y_pred, y_true)
    flat_scores = compute_uacqr_score_func_numpy(
        y_pred.reshape(y_pred.shape[0], -1, y_pred.shape[-1]),
        y_true.reshape(-1),
    ).reshape(y_true.shape)

    np.testing.assert_allclose(scores, flat_scores)
