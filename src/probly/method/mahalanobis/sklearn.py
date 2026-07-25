"""sklearn implementation of the Mahalanobis OOD transformation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, clone
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from probly.layers.array import ArrayMahalanobisHead
from probly.predictor import LogitClassifier, predict_raw
from probly.representation.distribution.array_categorical import ArrayProbabilityCategoricalDistribution

from ._common import MahalanobisPredictor, mahalanobis_generator
from .array import ArrayMahalanobisRepresentation

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from probly.layers.array import CovarianceEstimator

# The activations ``MLPClassifier`` supports. Reimplemented here rather than taken
# from ``sklearn.neural_network._base`` (a private module whose implementations
# mutate their argument in place).
_MLP_ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda z: z,
    "logistic": expit,
    "tanh": np.tanh,
    "relu": lambda z: np.maximum(z, 0.0),
}


class _IdentityFeatures:
    """Feature source returning the raw input, used for estimators without inner features."""

    def __call__(self, x: object) -> np.ndarray:
        return np.asarray(x, dtype=float)


class _MLPHiddenFeatures:
    """Feature source returning the activation of one hidden layer of an ``MLPClassifier``."""

    def __init__(self, estimator: MLPClassifier, layer_index: int) -> None:
        self.estimator = estimator
        self.layer_index = layer_index

    def __call__(self, x: object) -> np.ndarray:
        activation = _MLP_ACTIVATIONS[self.estimator.activation]
        out = np.asarray(x, dtype=float)
        for index in range(self.layer_index + 1):
            out = activation(out @ self.estimator.coefs_[index] + self.estimator.intercepts_[index])
        return out


class _PipelinePrefixFeatures:
    """Feature source returning the output of the first ``num_steps`` steps of a ``Pipeline``."""

    def __init__(self, pipeline: Pipeline, num_steps: int) -> None:
        self.pipeline = pipeline
        self.num_steps = num_steps

    def __call__(self, x: object) -> np.ndarray:
        return np.asarray(self.pipeline[: self.num_steps].transform(x), dtype=float)


type _FeatureSource = _IdentityFeatures | _MLPHiddenFeatures | _PipelinePrefixFeatures


def _node_index(name: str, available: Sequence[str]) -> int:
    """Resolve a feature-node name to its position, or raise with the valid names."""
    if name not in available:
        msg = f"Unknown feature node {name!r}. Available feature nodes are {list(available)}."
        raise ValueError(msg)
    return available.index(name)


def _resolve_feature_sources(
    predictor: BaseEstimator,
    feature_nodes: Sequence[str],
) -> tuple[_FeatureSource, Any, list[_FeatureSource]]:
    """Split an estimator into a penultimate feature source, a head and intermediate sources.

    Three estimator shapes are supported, in order of specificity: an
    ``MLPClassifier`` (whose hidden activations mirror the layers of a torch
    model), a multi-step ``Pipeline`` (whose leading steps produce features), and
    anything else (which exposes no inner features, so the raw input is the only
    feature layer).
    """
    if isinstance(predictor, MLPClassifier):
        num_hidden = predictor.n_layers_ - 2
        if num_hidden > 0:
            available = [f"hidden_{index}" for index in range(num_hidden)]
            hidden: list[_FeatureSource] = [
                _MLPHiddenFeatures(predictor, _node_index(name, available)) for name in feature_nodes
            ]
            return _MLPHiddenFeatures(predictor, num_hidden - 1), predictor, hidden

    elif isinstance(predictor, Pipeline) and len(predictor.steps) > 1:
        available = [name for name, _ in predictor.steps[:-1]]
        prefixes: list[_FeatureSource] = [
            _PipelinePrefixFeatures(predictor, _node_index(name, available) + 1) for name in feature_nodes
        ]
        encoder = _PipelinePrefixFeatures(predictor, len(predictor.steps) - 1)
        return encoder, predictor.steps[-1][1], prefixes

    if feature_nodes:
        msg = (
            f"An estimator of type {type(predictor).__name__} exposes no intermediate features, so feature_nodes "
            f"must be empty, but got {list(feature_nodes)}. Use an MLPClassifier or a multi-step Pipeline to build "
            "a multi-layer Mahalanobis ensemble."
        )
        raise ValueError(msg)
    return _IdentityFeatures(), predictor, []


@mahalanobis_generator.register(BaseEstimator)
class SklearnMahalanobisPredictor(
    BaseEstimator,
    MahalanobisPredictor[[np.ndarray], ArrayMahalanobisRepresentation],
):
    """sklearn Mahalanobis OOD predictor.

    The estimator is split into a feature encoder and a classification head, so
    that class-conditional Gaussians can be fitted on the features exactly as the
    torch backend fits them on penultimate activations. One
    :class:`~probly.layers.array.ArrayMahalanobisHead` is fitted per feature layer
    (any user-provided intermediate nodes plus the penultimate features), and the
    per-layer Mahalanobis confidences are combined into a single OOD score.

    Unlike torch models, sklearn estimators are only introspectable once trained,
    so the base estimator must already be fitted. After transforming, call
    ``fit_mahalanobis_heads(features, labels)`` to estimate the Gaussian
    parameters; optionally call ``fit_combiner(id, ood)`` to calibrate the
    multi-layer combination weights on in- versus out-of-distribution data.

    Attributes:
        encoder: The penultimate feature source.
        classification_head: The final classification stage of the estimator.
        mahalanobis_heads: One Mahalanobis head per feature layer (populated by
            ``fit_mahalanobis_heads``).
        combiner_weight: Per-layer combination weights of shape ``(num_layers,)``,
            initialised to ``-1`` and calibrated by ``fit_combiner``.
        combiner_bias: Scalar combination bias, initialised to ``0`` and
            calibrated by ``fit_combiner``.
    """

    def __init__(
        self,
        predictor: BaseEstimator,
        feature_nodes: Sequence[str] | None = None,
        input_preprocessing_eps: float = 0.0,
    ) -> None:
        """Build the Mahalanobis predictor from a fitted base classifier.

        Args:
            predictor: Fitted base classification estimator to be transformed.
            feature_nodes: Optional names of intermediate feature layers providing
                additional layers for the ensemble. Hidden layers of an
                ``MLPClassifier`` are named ``"hidden_0"``, ``"hidden_1"``, ...;
                steps of a ``Pipeline`` are named by their step name. When
                ``None`` only the penultimate features are used.
            input_preprocessing_eps: Unsupported by this backend; must be ``0``.

        Raises:
            ValueError: If ``input_preprocessing_eps`` is positive, if the base
                estimator is not fitted, or if ``feature_nodes`` names a layer the
                estimator does not expose.
        """
        # sklearn's get_params/clone contract requires constructor arguments to be
        # stored unchanged and under their own names; normalised copies are private.
        self.predictor = predictor
        self.feature_nodes = feature_nodes
        self.input_preprocessing_eps = input_preprocessing_eps

        if input_preprocessing_eps > 0:
            msg = (
                "The FGSM-style input preprocessing of the Mahalanobis detector needs gradients with respect to the "
                "input, which sklearn estimators do not provide. Use input_preprocessing_eps=0, or apply the "
                "transformation to a torch model."
            )
            raise ValueError(msg)

        classes = getattr(predictor, "classes_", None)
        if classes is None:
            msg = (
                f"The estimator of type {type(predictor).__name__} is not fitted. The sklearn Mahalanobis backend "
                "inspects the fitted estimator to expose its features and classes, so call fit() on the base "
                "estimator before applying mahalanobis()."
            )
            raise ValueError(msg)

        self._classes = np.asarray(classes)
        self._num_classes = len(self._classes)
        self._feature_nodes = list(feature_nodes) if feature_nodes is not None else []

        encoder, head, intermediate = _resolve_feature_sources(predictor, self._feature_nodes)
        self.encoder = encoder
        self.classification_head = head
        # Intermediate nodes come first and the penultimate features last, matching the torch backend.
        self._feature_sources: list[_FeatureSource] = [*intermediate, encoder]

        self.mahalanobis_heads: list[ArrayMahalanobisHead] = []
        # Default combiner: negated sum of per-layer confidences (high score => out-of-distribution).
        self.combiner_weight = -np.ones(len(self._feature_sources))
        self.combiner_bias = np.zeros(())

    def __sklearn_clone__(self) -> SklearnMahalanobisPredictor:
        """Clone the wrapper while keeping the same fitted base estimator.

        sklearn's default ``clone`` recursively unfits nested estimators, which
        would strip the base estimator this post-hoc transformation is built on.
        Keeping the base is the same convention ``sklearn.base.FrozenEstimator``
        follows; only the Mahalanobis state is dropped.
        """
        return type(self)(self.predictor, self.feature_nodes, self.input_preprocessing_eps)

    @property
    def estimator(self) -> BaseEstimator:
        """Alias to sklearn's conventional attribute name for wrapped estimators."""
        return self.predictor

    @estimator.setter
    def estimator(self, value: BaseEstimator) -> None:
        """Set the wrapped estimator via sklearn's conventional attribute alias."""
        self.predictor = value

    def _logits(self, x: object) -> np.ndarray:
        """Read raw logits from the wrapped estimator, normalised to shape ``(N, num_classes)``."""
        # The transformation tags the base as a LogitClassifier, but that registration is a weak
        # per-instance one that does not survive a plain pickle round-trip; re-assert it here.
        LogitClassifier.register_instance(self.predictor)
        logits = np.asarray(predict_raw(self.predictor, x), dtype=float)
        if logits.ndim == 1:
            # A binary decision_function returns one margin per sample; expand it to two columns.
            logits = np.stack([np.zeros_like(logits), logits], axis=-1)
        return logits

    def _features(self, x: object) -> list[np.ndarray]:
        """Extract the per-layer feature matrices for the given input."""
        return [source(x) for source in self._feature_sources]

    def _layer_scores(self, feats: list[np.ndarray]) -> np.ndarray:
        """Per-layer Mahalanobis confidences (max over classes) stacked to ``(N, num_layers)``."""
        if not self.mahalanobis_heads:
            msg = "Mahalanobis heads are not fitted. Please call fit_mahalanobis_heads() or fit() before prediction."
            raise ValueError(msg)
        scores = [head.score(feat).max(axis=-1) for head, feat in zip(self.mahalanobis_heads, feats, strict=True)]
        return np.stack(scores, axis=-1)

    def _encode_labels(self, labels: object) -> np.ndarray:
        """Map labels onto the class indices ``[0, num_classes)`` used by the Mahalanobis heads.

        Labels the base estimator never saw are mapped to ``num_classes``, i.e. out
        of range, so that fitting fails with a clear error rather than silently
        dropping them.
        """
        label_array = np.asarray(labels)
        positions = np.searchsorted(self._classes, label_array)
        clipped = np.clip(positions, 0, self._num_classes - 1)
        known = self._classes[clipped] == label_array
        return np.where(known, clipped, self._num_classes)

    def fit_mahalanobis_heads(
        self,
        features: object,
        labels: object,
        covariance_estimator: CovarianceEstimator | None = None,
    ) -> None:
        """Fit one Mahalanobis head per feature layer on the given inputs.

        Args:
            features: Input matrix (e.g. the training inputs) fed to the encoder.
            labels: Class labels of shape ``(N,)``, in the label space of the base
                estimator (they are mapped onto its ``classes_``).
            covariance_estimator: Optional covariance estimator used to derive the
                tied precision matrix. Defaults to
                ``sklearn.covariance.EmpiricalCovariance(assume_centered=True)``,
                reproducing the estimate of :cite:`leeSimpleUnifiedFramework2018`;
                pass e.g. ``LedoitWolf()`` for a shrunk alternative.
        """
        estimator = EmpiricalCovariance(assume_centered=True) if covariance_estimator is None else covariance_estimator
        feats = self._features(features)
        encoded = self._encode_labels(labels)

        heads = [ArrayMahalanobisHead(self._num_classes, feat.shape[-1]) for feat in feats]
        for head, feat in zip(heads, feats, strict=True):
            head.fit(feat, encoded, covariance_estimator=clone(estimator))
        self.mahalanobis_heads = heads

    def fit_combiner(
        self,
        id_features: object,
        ood_features: object,
        cv: int | None = None,
        **combiner_kwargs: object,
    ) -> None:
        """Calibrate the multi-layer combination weights by logistic regression.

        Fits weights and a bias over the per-layer Mahalanobis scores so that
        out-of-distribution inputs (label 1) score higher than in-distribution
        inputs (label 0), reproducing the feature-ensemble step of
        :cite:`leeSimpleUnifiedFramework2018` with the same ``LogisticRegressionCV``
        the reference implementation uses.

        Args:
            id_features: In-distribution inputs fed to the encoder.
            ood_features: Out-of-distribution inputs fed to the encoder.
            cv: Number of cross-validation folds. Defaults to at most 5, capped by
                the smaller of the two sample counts. With fewer than two samples
                per side an uncross-validated ``LogisticRegression`` is used.
            **combiner_kwargs: Extra keyword arguments forwarded to the logistic
                regression.
        """
        id_scores = self._layer_scores(self._features(id_features))
        ood_scores = self._layer_scores(self._features(ood_features))
        scores = np.concatenate([id_scores, ood_scores], axis=0)
        targets = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])

        smallest_split = min(len(id_scores), len(ood_scores))
        combiner: LogisticRegression | LogisticRegressionCV
        if smallest_split < 2:
            combiner = LogisticRegression(**combiner_kwargs)
        else:
            folds = cv if cv is not None else max(2, min(5, smallest_split))
            combiner = LogisticRegressionCV(cv=folds, **combiner_kwargs)
        combiner.fit(scores, targets)

        self.combiner_weight = np.asarray(combiner.coef_[0], dtype=float)
        self.combiner_bias = np.asarray(combiner.intercept_[0], dtype=float)

    def predict_representation(self, x: object) -> ArrayMahalanobisRepresentation:
        """Predict the Mahalanobis representation (softmax and per-layer scores)."""
        logits = self._logits(x)
        layer_scores = self._layer_scores(self._features(x))

        return ArrayMahalanobisRepresentation(
            ArrayProbabilityCategoricalDistribution(softmax(logits, axis=-1)),
            layer_scores,
            self.combiner_weight,
            self.combiner_bias,
        )

    def fit(self, x: object, y: object, **fit_kwargs: object) -> SklearnMahalanobisPredictor:
        """Fit the Mahalanobis heads, using sklearn's argument order and return convention."""
        self.fit_mahalanobis_heads(x, y, **fit_kwargs)  # ty:ignore[invalid-argument-type]
        return self

    def predict(self, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        """Forward ``predict`` to the wrapped estimator."""
        predict_method = getattr(self.predictor, "predict", None)
        if predict_method is None or not callable(predict_method):
            msg = f"Wrapped predictor {type(self.predictor)} has no predict method."
            raise AttributeError(msg)
        return predict_method(*args, **kwargs)


@predict_raw.register(SklearnMahalanobisPredictor)
def predict_raw_sklearn_mahalanobis(
    predictor: SklearnMahalanobisPredictor,
    /,
    *args: object,
    **kwargs: object,
) -> ArrayMahalanobisRepresentation:
    """Return the Mahalanobis representation for the wrapper.

    Without this registration the wrapper would inherit the generic
    ``BaseEstimator`` handler in :mod:`probly.predictor.sklearn`, which returns
    class labels and never reaches ``predict_representation``.
    """
    return predictor.predict_representation(*args, **kwargs)
