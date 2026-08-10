"""Numpy layer implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike


@runtime_checkable
class CovarianceEstimator(Protocol):
    """An estimator exposing a precision matrix after being fitted.

    Structurally satisfied by the estimators in :mod:`sklearn.covariance`, e.g.
    ``EmpiricalCovariance``, ``LedoitWolf`` or ``ShrunkCovariance``.
    """

    precision_: np.ndarray

    def fit(self, features: np.ndarray, /) -> object:
        """Fit the estimator on a feature matrix of shape ``(N, feature_dim)``."""


class ArrayMahalanobisHead:
    """Class-conditional Gaussian head with a tied covariance (Mahalanobis OOD).

    Numpy counterpart of :class:`probly.layers.torch.MahalanobisHead`, implementing
    the Gaussian-discriminant-analysis confidence of the out-of-distribution
    detector of :cite:`leeSimpleUnifiedFramework2018`. One mean is estimated per
    class, while a single covariance (its inverse, the precision matrix) is shared
    across all classes. After fitting, :meth:`score` gives each sample a per-class
    confidence (higher = closer to a class centroid = more in-distribution), of
    which downstream code typically takes the max over classes.

    Attributes:
        means: Per-class mean vectors, shape ``(num_classes, feature_dim)``.
        precision: Shared (tied) precision matrix, shape
            ``(feature_dim, feature_dim)``.
    """

    def __init__(self, num_classes: int, feature_dim: int) -> None:
        """Initialize with zero means and an identity precision.

        Args:
            num_classes: Number of classes (one mean vector per class).
            feature_dim: Dimensionality of the feature vectors.
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.means = np.zeros((num_classes, feature_dim))
        self.precision = np.eye(feature_dim)

    def fit(
        self,
        features: ArrayLike | np.ndarray,
        labels: ArrayLike | np.ndarray,
        covariance_estimator: CovarianceEstimator | None = None,
    ) -> None:
        """Estimate per-class means and the shared (tied) precision matrix.

        Each class mean is the empirical mean of its features. The shared
        covariance is the empirical covariance of all per-class-centered features
        (pooled within-class scatter). Its inverse is obtained from
        ``covariance_estimator`` when given, and otherwise from the Hermitian
        pseudo-inverse, which stays finite even when the pooled covariance is
        rank-deficient, e.g. for dead post-ReLU feature dimensions.

        Args:
            features: Feature vectors of shape ``(N, feature_dim)``.
            labels: Integer class labels of shape ``(N,)``, taking values in
                ``[0, num_classes)``.
            covariance_estimator: Optional fitted-on-demand covariance estimator
                used to derive the precision matrix, e.g.
                ``sklearn.covariance.LedoitWolf()`` for a shrunk estimate. When
                ``None`` the pseudo-inverse of the empirical covariance is used.

        Raises:
            ValueError: If no sample matches any class index in
                ``[0, num_classes)``, leaving the covariance undefined.
        """
        feature_array = np.asarray(features, dtype=float)
        label_array = np.asarray(labels)

        means = np.zeros_like(self.means)
        centered = []
        for c in range(self.num_classes):
            mask = label_array == c
            if not bool(np.any(mask)):
                continue
            z = feature_array[mask]
            mu = z.mean(axis=0)
            means[c] = mu
            centered.append(z - mu)
        if not centered:
            msg = "Cannot fit ArrayMahalanobisHead: no labelled samples were provided."
            raise ValueError(msg)

        centered_all = np.concatenate(centered, axis=0)
        if covariance_estimator is not None:
            covariance_estimator.fit(centered_all)
            precision = np.asarray(covariance_estimator.precision_, dtype=float)
        else:
            covariance = centered_all.T @ centered_all / centered_all.shape[0]
            precision = np.linalg.pinv(covariance, hermitian=True)

        self.means = means
        self.precision = precision

    def score(self, features: ArrayLike | np.ndarray) -> np.ndarray:
        """Compute per-class Mahalanobis confidence scores for each sample.

        Returns ``-0.5 * (z - mu_c)^T P (z - mu_c)`` for every class *c*.

        Args:
            features: Feature vectors of shape ``(N, feature_dim)``.

        Returns:
            Per-class confidence scores of shape ``(N, num_classes)``.
        """
        feature_array = np.asarray(features, dtype=float)
        diff = feature_array[:, None, :] - self.means[None, :, :]  # (N, C, D)
        # Per-class quadratic form via einsum, shape (N, C).
        mahalanobis = np.einsum("ncd,de,nce->nc", diff, self.precision, diff)
        return -0.5 * mahalanobis

    def __call__(self, features: ArrayLike | np.ndarray) -> np.ndarray:
        """Alias for :meth:`score`."""
        return self.score(features)
