"""Tests for the sklearn Mahalanobis OOD implementation."""

from __future__ import annotations

import numpy as np
import pytest

from probly.decider import categorical_from_mean
from probly.method.mahalanobis import mahalanobis
from probly.method.mahalanobis.array import ArrayMahalanobisRepresentation
from probly.predictor import predict

pytest.importorskip("sklearn")

from flextype import registry_pickle
from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from probly.method.mahalanobis.sklearn import SklearnMahalanobisPredictor

NUM_SAMPLES = 300
NUM_CLASSES = 3
HIDDEN_SIZES = (16, 8)
MAX_ITER = 800


@pytest.fixture
def train_data() -> tuple[np.ndarray, np.ndarray]:
    """A three-class two-moons batch large enough to estimate a covariance."""
    x, y = make_moons(n_samples=NUM_SAMPLES, noise=0.1, random_state=0)
    return x, (x[:, 0] > 0).astype(int) + y


@pytest.fixture
def ood_data() -> np.ndarray:
    """Inputs far from the training distribution."""
    return np.random.default_rng(0).normal(size=(120, 2)) * 5 + 20


@pytest.fixture
def mlp(train_data: tuple[np.ndarray, np.ndarray]) -> MLPClassifier:
    """A fitted multi-class MLP with two hidden layers."""
    x, y = train_data
    return MLPClassifier(hidden_layer_sizes=HIDDEN_SIZES, max_iter=MAX_ITER, random_state=0).fit(x, y)


@pytest.fixture
def pipeline(train_data: tuple[np.ndarray, np.ndarray]) -> Pipeline:
    """A fitted two-step pipeline ending in a classifier."""
    x, y = train_data
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]).fit(x, y)


@pytest.fixture
def logistic_regression(train_data: tuple[np.ndarray, np.ndarray]) -> LogisticRegression:
    """A fitted bare classifier exposing no intermediate features."""
    x, y = train_data
    return LogisticRegression(max_iter=500).fit(x, y)


# ---------------------------------------------------------------------------
# Transformation structure
# ---------------------------------------------------------------------------


class TestTransformation:
    """The mahalanobis transformation splits the estimator into an encoder and a head."""

    def test_mlp_hidden_layers_become_features(self, mlp: MLPClassifier) -> None:
        """An MLP exposes its hidden activations as feature layers."""
        out = mahalanobis(mlp, feature_nodes=["hidden_0"])
        assert isinstance(out, SklearnMahalanobisPredictor)
        assert out.combiner_weight.shape == (2,)
        assert out.estimator is mlp
        # A monolithic MLP has no separable head, so the whole classifier stands in as the head.
        assert out.classification_head is mlp

    def test_pipeline_splits_at_the_final_step(self, pipeline: Pipeline) -> None:
        """A pipeline keeps its final estimator as the classification head."""
        out = mahalanobis(pipeline)
        assert out.classification_head is pipeline.steps[-1][1]

    def test_bare_estimator_uses_raw_input(self, logistic_regression: LogisticRegression) -> None:
        """An estimator without inner features scores on the raw input as a single layer."""
        out = mahalanobis(logistic_regression)
        assert out.classification_head is logistic_regression
        assert out.combiner_weight.shape == (1,)

    def test_unfitted_estimator_raises(self) -> None:
        """The backend inspects the fitted estimator, so an unfitted base is rejected."""
        with pytest.raises(ValueError, match="not fitted or not a classifier"):
            mahalanobis(MLPClassifier())

    def test_fitted_regressor_raises(self, train_data: tuple[np.ndarray, np.ndarray]) -> None:
        """A fitted regressor exposes no classes_ and is rejected, not reported as unfitted."""
        x, y = train_data
        regressor = LinearRegression().fit(x, y.astype(float))
        with pytest.raises(ValueError, match="not fitted or not a classifier"):
            mahalanobis(regressor)

    def test_input_preprocessing_raises(self, mlp: MLPClassifier) -> None:
        """FGSM preprocessing needs input gradients, which sklearn cannot provide."""
        with pytest.raises(ValueError, match="input preprocessing"):
            mahalanobis(mlp, input_preprocessing_eps=0.01)

    def test_unknown_feature_node_raises(self, mlp: MLPClassifier) -> None:
        """An unknown node name is reported together with the available ones."""
        with pytest.raises(ValueError, match="Unknown feature node"):
            mahalanobis(mlp, feature_nodes=["hidden_9"])

    def test_feature_nodes_on_featureless_estimator_raise(self, logistic_regression: LogisticRegression) -> None:
        """Naming feature nodes on an estimator without inner features is an error."""
        with pytest.raises(ValueError, match="no intermediate features"):
            mahalanobis(logistic_regression, feature_nodes=["anything"])

    def test_single_step_pipeline_has_no_feature_nodes(self, train_data: tuple[np.ndarray, np.ndarray]) -> None:
        """A one-step pipeline cannot transform, so it falls back to the raw input."""
        x, y = train_data
        single = Pipeline([("clf", LogisticRegression(max_iter=500))]).fit(x, y)
        assert mahalanobis(single).combiner_weight.shape == (1,)
        with pytest.raises(ValueError, match="no intermediate features"):
            mahalanobis(single, feature_nodes=["clf"])


# ---------------------------------------------------------------------------
# Fitting and prediction
# ---------------------------------------------------------------------------


class TestFitAndPredict:
    """Fitting the Mahalanobis heads and predicting representations."""

    def test_single_layer_shapes(self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Default (no feature nodes) yields a single feature layer over the penultimate activations."""
        x, y = train_data
        out = mahalanobis(mlp)
        out.fit_mahalanobis_heads(x, y)
        assert len(out.mahalanobis_heads) == 1
        assert out.mahalanobis_heads[0].feature_dim == HIDDEN_SIZES[-1]
        assert predict(out, x).layer_scores.shape == (len(x), 1)

    def test_multi_layer_shapes(self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]) -> None:
        """An extra feature node adds a layer to the ensemble, ordered before the penultimate one."""
        x, y = train_data
        out = mahalanobis(mlp, feature_nodes=["hidden_0"])
        out.fit_mahalanobis_heads(x, y)
        assert [head.feature_dim for head in out.mahalanobis_heads] == list(HIDDEN_SIZES)
        assert predict(out, x).layer_scores.shape == (len(x), 2)

    def test_predict_returns_a_representation(
        self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """predict() must reach predict_representation, not the generic sklearn label path."""
        x, y = train_data
        out = mahalanobis(mlp)
        out.fit_mahalanobis_heads(x, y)
        assert isinstance(predict(out, x), ArrayMahalanobisRepresentation)

    def test_categorical_from_mean_matches_predict_proba(
        self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The categorical mean decider reproduces the base estimator's probabilities."""
        x, y = train_data
        out = mahalanobis(mlp)
        out.fit_mahalanobis_heads(x, y)
        single = categorical_from_mean(predict(out, x))
        np.testing.assert_allclose(single.probabilities, mlp.predict_proba(x), atol=1e-10)

    def test_binary_estimator_keeps_two_classes(self, train_data: tuple[np.ndarray, np.ndarray]) -> None:
        """A binary MLP has a single output unit but still two classes."""
        x, y = train_data
        binary = MLPClassifier(hidden_layer_sizes=HIDDEN_SIZES, max_iter=MAX_ITER, random_state=0)
        out = mahalanobis(binary.fit(x, y > 0))
        out.fit_mahalanobis_heads(x, y > 0)
        assert out.mahalanobis_heads[0].means.shape[0] == 2
        assert predict(out, x).softmax.probabilities.shape == (len(x), 2)

    def test_binary_decision_function_is_expanded(self, train_data: tuple[np.ndarray, np.ndarray]) -> None:
        """A binary estimator's 1-D decision_function is expanded into a two-column softmax."""
        x, y = train_data
        binary = LogisticRegression(max_iter=500).fit(x, y > 0)
        out = mahalanobis(binary)
        out.fit_mahalanobis_heads(x, y > 0)

        probabilities = predict(out, x).softmax.probabilities
        assert probabilities.shape == (len(x), 2)
        np.testing.assert_allclose(probabilities, binary.predict_proba(x), atol=1e-10)

    def test_pipeline_and_bare_estimator_predict(
        self,
        pipeline: Pipeline,
        logistic_regression: LogisticRegression,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Both remaining feature sources produce finite scores of the expected shape."""
        x, y = train_data
        for base, num_layers in (
            (mahalanobis(pipeline, feature_nodes=["scaler"]), 2),
            (mahalanobis(logistic_regression), 1),
        ):
            base.fit_mahalanobis_heads(x, y)
            scores = predict(base, x).layer_scores
            assert scores.shape == (len(x), num_layers)
            assert np.isfinite(scores).all()

    def test_string_labels_are_encoded(self, train_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Non-integer labels are mapped onto the base estimator's classes_."""
        x, y = train_data
        labels = np.array(["a", "b", "c"])[y]
        base = MLPClassifier(hidden_layer_sizes=HIDDEN_SIZES, max_iter=MAX_ITER, random_state=0).fit(x, labels)
        out = mahalanobis(base)
        out.fit_mahalanobis_heads(x, labels)
        assert np.any(out.mahalanobis_heads[0].means != 0)

    def test_unknown_labels_raise(self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Labels the base estimator never saw are rejected."""
        x, _ = train_data
        out = mahalanobis(mlp)
        with pytest.raises(ValueError, match="never seen"):
            out.fit_mahalanobis_heads(x, np.full(len(x), NUM_CLASSES))

    def test_partially_unknown_labels_raise(
        self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """A single unknown label raises instead of being silently dropped."""
        x, y = train_data
        labels = y.copy()
        labels[0] = NUM_CLASSES
        with pytest.raises(ValueError, match=rf"Labels \[{NUM_CLASSES}\] were never seen"):
            mahalanobis(mlp).fit_mahalanobis_heads(x, labels)

    def test_predict_before_fit_raises(self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Predicting before the heads are fitted reports what is missing."""
        x, _ = train_data
        with pytest.raises(ValueError, match="not fitted"):
            predict(mahalanobis(mlp), x)

    def test_fit_alias_matches_generic_op(self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]) -> None:
        """The sklearn-style fit() alias is equivalent to fit_mahalanobis_heads()."""
        x, y = train_data
        via_alias = mahalanobis(mlp).fit(x, y)
        direct = mahalanobis(mlp)
        direct.fit_mahalanobis_heads(x, y)
        np.testing.assert_allclose(predict(via_alias, x).layer_scores, predict(direct, x).layer_scores)

    def test_duck_typed_covariance_estimator(
        self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """A protocol-satisfying non-sklearn covariance estimator is accepted."""

        class DuckCovariance:
            precision_: np.ndarray

            def fit(self, features: np.ndarray, /) -> DuckCovariance:
                self.precision_ = np.eye(features.shape[1])
                return self

        x, y = train_data
        out = mahalanobis(mlp)
        out.fit_mahalanobis_heads(x, y, covariance_estimator=DuckCovariance())
        np.testing.assert_allclose(out.mahalanobis_heads[0].precision, np.eye(HIDDEN_SIZES[-1]))


# ---------------------------------------------------------------------------
# Combiner calibration
# ---------------------------------------------------------------------------


class TestCombiner:
    """The logistic-regression combiner over the per-layer scores."""

    def test_default_weights_are_negative_unit(self, mlp: MLPClassifier) -> None:
        """Before calibration the combiner negates the sum of the per-layer confidences."""
        out = mahalanobis(mlp, feature_nodes=["hidden_0"])
        np.testing.assert_allclose(out.combiner_weight, -np.ones(2))
        np.testing.assert_allclose(out.combiner_bias, 0.0)

    def test_fit_combiner_separates_in_and_out(
        self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray], ood_data: np.ndarray
    ) -> None:
        """After calibration, out-of-distribution inputs score higher than in-distribution."""
        x, y = train_data
        out = mahalanobis(mlp, feature_nodes=["hidden_0"])
        out.fit_mahalanobis_heads(x, y)
        out.fit_combiner(x, ood_data)

        assert np.isfinite(out.combiner_weight).all()
        id_score = predict(out, x).layer_scores @ out.combiner_weight + out.combiner_bias
        ood_score = predict(out, ood_data).layer_scores @ out.combiner_weight + out.combiner_bias
        assert ood_score.mean() > id_score.mean()

    def test_fit_combiner_falls_back_without_cross_validation(
        self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray], ood_data: np.ndarray
    ) -> None:
        """A single out-of-distribution sample is too few to cross-validate, but still fits."""
        x, y = train_data
        out = mahalanobis(mlp)
        out.fit_mahalanobis_heads(x, y)
        out.fit_combiner(x, ood_data[:1])
        assert out.combiner_weight.shape == (1,)
        assert np.isfinite(out.combiner_bias)

    def test_fit_combiner_honours_explicit_cv(
        self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray], ood_data: np.ndarray
    ) -> None:
        """An explicitly given cv reaches the cross-validated combiner instead of the fallback."""
        x, y = train_data
        out = mahalanobis(mlp)
        out.fit_mahalanobis_heads(x, y)
        # Only the cross-validation split warns about the 1-member class; the
        # uncross-validated fallback would fit silently.
        with pytest.warns(UserWarning, match="least populated class"):
            out.fit_combiner(x, ood_data[:1], cv=2)
        assert np.isfinite(out.combiner_weight).all()


# ---------------------------------------------------------------------------
# sklearn estimator contracts
# ---------------------------------------------------------------------------


class TestEstimatorContracts:
    """The wrapper behaves like a well-formed sklearn estimator."""

    def test_get_params_exposes_constructor_arguments(self, mlp: MLPClassifier) -> None:
        """Constructor arguments are stored verbatim, as get_params requires."""
        out = mahalanobis(mlp, feature_nodes=["hidden_0"])
        params = out.get_params(deep=False)
        assert params["predictor"] is mlp
        assert params["feature_nodes"] == ["hidden_0"]
        assert params["input_preprocessing_eps"] == 0.0

    def test_clone_keeps_the_fitted_base(self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Cloning drops the Mahalanobis state but keeps the fitted base estimator."""
        x, y = train_data
        out = mahalanobis(mlp)
        out.fit_mahalanobis_heads(x, y)
        cloned = clone(out)
        assert cloned.predictor is mlp
        assert cloned.mahalanobis_heads == []

    def test_set_params_rebuilds_derived_state(
        self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """set_params re-derives the feature sources and drops the fitted heads."""
        x, y = train_data
        out = mahalanobis(mlp)
        out.fit_mahalanobis_heads(x, y)
        out.set_params(feature_nodes=["hidden_0"])
        assert out.combiner_weight.shape == (2,)
        assert out.mahalanobis_heads == []

    def test_set_params_revalidates(self, mlp: MLPClassifier) -> None:
        """set_params runs the same validation as the constructor."""
        out = mahalanobis(mlp)
        with pytest.raises(ValueError, match="input preprocessing"):
            out.set_params(input_preprocessing_eps=0.1)

    def test_registry_pickle_roundtrip_reproduces_predictions(
        self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """A serialization round-trip preserves the fitted Mahalanobis state."""
        x, y = train_data
        out = mahalanobis(mlp, feature_nodes=["hidden_0"])
        out.fit_mahalanobis_heads(x, y)
        restored = registry_pickle.loads(registry_pickle.dumps(out))
        np.testing.assert_allclose(predict(restored, x).layer_scores, predict(out, x).layer_scores)

    def test_predict_forwards_to_the_base_estimator(
        self, mlp: MLPClassifier, train_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The sklearn predict() surface still yields the base estimator's labels."""
        x, y = train_data
        out = mahalanobis(mlp)
        out.fit_mahalanobis_heads(x, y)
        np.testing.assert_array_equal(out.predict(x), mlp.predict(x))
