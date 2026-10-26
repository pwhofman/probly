"""Tests for the flax Mahalanobis OOD implementation."""

from __future__ import annotations

import numpy as np
import pytest

from probly.decider import categorical_from_mean
from probly.method.mahalanobis import mahalanobis
from probly.predictor import predict
from probly.quantification import decompose

pytest.importorskip("flax")

from flax import nnx
import jax
import jax.numpy as jnp

from probly.layers.flax import Identity, MahalanobisHead
from probly.layers.numpy import NumpyMahalanobisHead
from probly.representation.distribution.jax_categorical import JaxCategoricalDistribution

# ``flax_custom_model``: Linear(10, 20) -> ReLU -> Linear(20, 4) -> softmax.
CUSTOM_IN_FEATURES = 10
CUSTOM_NUM_CLASSES = 4
CUSTOM_FEATURE_DIM = 20
# ``conv_model``: Conv(3, 5, (3, 3)) -> ReLU -> spatial mean -> Linear(5, 2).
CONV_NUM_CLASSES = 2
CONV_FEATURE_DIM = 5


# ---------------------------------------------------------------------------
# Local models and data (the shared sample is too small to estimate a covariance,
# and the shared conv fixture is structural only -- ``nnx.flatten`` is a graph
# utility, not an array reshape, so it cannot run a forward pass.)
# ---------------------------------------------------------------------------


@pytest.fixture
def conv_model(flax_rngs: nnx.Rngs) -> nnx.Module:
    """A channels-last conv classifier that pools its spatial axes before the head."""

    class ConvModel(nnx.Module):
        def __init__(self, rngs: nnx.Rngs) -> None:
            super().__init__()
            self.conv = nnx.Conv(3, CONV_FEATURE_DIM, (3, 3), rngs=rngs)
            self.head = nnx.Linear(CONV_FEATURE_DIM, CONV_NUM_CLASSES, rngs=rngs)

        def __call__(self, x: jax.Array) -> jax.Array:
            x = nnx.relu(self.conv(x))
            return self.head(x.mean(axis=(1, 2)))

    return ConvModel(flax_rngs)


@pytest.fixture
def linear_train_data() -> tuple[jax.Array, jax.Array]:
    """A multi-class training batch shaped for ``flax_custom_model``."""
    inputs = jax.random.normal(jax.random.key(0), (80, CUSTOM_IN_FEATURES))
    targets = jax.random.randint(jax.random.key(1), (80,), 0, CUSTOM_NUM_CLASSES)
    return inputs, targets


@pytest.fixture
def conv_train_data() -> tuple[jax.Array, jax.Array]:
    """A channels-last training batch shaped for ``conv_model``."""
    inputs = jax.random.normal(jax.random.key(2), (60, 5, 5, 3))
    targets = jax.random.randint(jax.random.key(3), (60,), 0, CONV_NUM_CLASSES)
    return inputs, targets


@pytest.fixture
def ood_data() -> jax.Array:
    """Inputs far from the in-distribution training batch."""
    return jax.random.normal(jax.random.key(4), (80, CUSTOM_IN_FEATURES)) * 6.0 + 4.0


# ---------------------------------------------------------------------------
# Transformation structure
# ---------------------------------------------------------------------------


class TestTransformation:
    """The mahalanobis transformation rewires the model into an encoder + head."""

    def test_head_replaced_with_identity(self, flax_custom_model: nnx.Module) -> None:
        """The last Linear is stripped from the encoder and kept as classification head."""
        out = mahalanobis(flax_custom_model, predictor_type="logit_classifier")
        assert isinstance(out.encoder.linear2, Identity)
        assert isinstance(out.classification_head, nnx.Linear)
        assert out.classification_head.out_features == CUSTOM_NUM_CLASSES

    def test_head_found_in_sequential(self, flax_model_small_2d_2d: nnx.Module) -> None:
        """In a plain Linear stack the final Linear becomes the classification head."""
        out = mahalanobis(flax_model_small_2d_2d, predictor_type="logit_classifier")
        assert isinstance(out.encoder.layers[2], Identity)
        assert isinstance(out.classification_head, nnx.Linear)

    def test_no_linear_raises(self) -> None:
        """A model without a Linear layer cannot identify a classification head."""
        with pytest.raises(ValueError, match=r"No nnx\.Linear"):
            mahalanobis(nnx.Sequential(nnx.relu), predictor_type="logit_classifier")

    def test_unknown_feature_node_raises(self, flax_custom_model: nnx.Module) -> None:
        """An unresolvable feature node names the paths that are actually available."""
        with pytest.raises(ValueError, match="not a submodule of the encoder"):
            mahalanobis(flax_custom_model, feature_nodes=["missing"], predictor_type="logit_classifier")


# ---------------------------------------------------------------------------
# Fitting and forward pass
# ---------------------------------------------------------------------------


class TestFitAndForward:
    """Fitting the Mahalanobis heads and running the forward pass."""

    def test_single_layer_shapes(
        self, flax_custom_model: nnx.Module, linear_train_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """Default (no feature nodes) yields a single feature layer."""
        x, y = linear_train_data
        out = mahalanobis(flax_custom_model, predictor_type="logit_classifier")
        out.fit_mahalanobis_heads(x, y)
        assert len(out.mahalanobis_heads) == 1
        logits, scores = out(x)
        assert logits.shape == (len(x), CUSTOM_NUM_CLASSES)
        assert scores.shape == (len(x), 1)

    def test_multi_layer_shapes(
        self, flax_custom_model: nnx.Module, linear_train_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """An extra feature node adds a layer to the ensemble."""
        x, y = linear_train_data
        out = mahalanobis(flax_custom_model, feature_nodes=["linear1"], predictor_type="logit_classifier")
        out.fit_mahalanobis_heads(x, y)
        assert len(out.mahalanobis_heads) == 2
        _, scores = out(x)
        assert scores.shape == (len(x), 2)
        assert out.combiner_weight.shape == (2,)

    def test_sequential_feature_node(self, flax_model_small_2d_2d: nnx.Module) -> None:
        """A feature node inside an ``nnx.Sequential`` is addressed by its indexed path."""
        x = jax.random.normal(jax.random.key(5), (40, 2))
        y = jax.random.randint(jax.random.key(6), (40,), 0, 2)
        out = mahalanobis(flax_model_small_2d_2d, feature_nodes=["layers.0"], predictor_type="logit_classifier")
        out.fit_mahalanobis_heads(x, y)
        assert len(out.mahalanobis_heads) == 2
        _, scores = out(x)
        assert scores.shape == (len(x), 2)

    def test_feature_capture_preserves_the_forward_pass(
        self, flax_custom_model: nnx.Module, linear_train_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """Wrapping a node in a capture module does not change what the encoder computes."""
        x, _ = linear_train_data
        plain = mahalanobis(flax_custom_model, predictor_type="logit_classifier")
        tapped = mahalanobis(flax_custom_model, feature_nodes=["linear1"], predictor_type="logit_classifier")
        assert jnp.allclose(plain.encoder(x), tapped.encoder(x))

    def test_fit_populates_parameters(
        self, flax_custom_model: nnx.Module, linear_train_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """Fitting sets non-trivial class means and a non-identity precision."""
        x, y = linear_train_data
        out = mahalanobis(flax_custom_model, predictor_type="logit_classifier")
        out.fit_mahalanobis_heads(x, y)
        head = out.mahalanobis_heads[0]
        assert head.feature_dim == CUSTOM_FEATURE_DIM
        assert bool(jnp.any(head.means != 0))
        assert not bool(jnp.allclose(head.precision, jnp.eye(head.feature_dim)))

    def test_categorical_from_mean_returns_softmax(
        self, flax_custom_model: nnx.Module, linear_train_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """The categorical mean decider reduces the representation to its softmax."""
        x, y = linear_train_data
        out = mahalanobis(flax_custom_model, predictor_type="logit_classifier")
        out.fit_mahalanobis_heads(x, y)
        single = categorical_from_mean(predict(out, x))
        logits, _ = out(x)
        assert isinstance(single, JaxCategoricalDistribution)
        assert jnp.allclose(single.probabilities, jax.nn.softmax(logits, axis=-1))

    def test_decomposition_combines_layer_scores(
        self, flax_custom_model: nnx.Module, linear_train_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """The epistemic score is the default negated sum of the per-layer confidences."""
        x, y = linear_train_data
        out = mahalanobis(flax_custom_model, feature_nodes=["linear1"], predictor_type="logit_classifier")
        out.fit_mahalanobis_heads(x, y)
        _, scores = out(x)
        assert jnp.allclose(decompose(predict(out, x)).epistemic, -scores.sum(axis=-1), atol=1e-4)

    def test_fitting_runs_in_eval_mode_and_restores_flags(self, flax_dropout_model: nnx.Module) -> None:
        """Fitting disables dropout, then restores each submodule flag to its prior value.

        Repeating the fit must give identical parameters: if the encoder were left
        in training mode, dropout would resample and the estimates would differ.
        """
        x = jax.random.normal(jax.random.key(7), (40, 2))
        y = jax.random.randint(jax.random.key(8), (40,), 0, 2)
        out = mahalanobis(flax_dropout_model, predictor_type="logit_classifier")
        dropout = out.encoder.layers[1]
        assert dropout.deterministic is False

        out.fit_mahalanobis_heads(x, y)
        first = out.mahalanobis_heads[0].means
        assert dropout.deterministic is False

        out.fit_mahalanobis_heads(x, y)
        assert jnp.array_equal(first, out.mahalanobis_heads[0].means)

    def test_fit_without_matching_labels_raises(
        self, flax_custom_model: nnx.Module, linear_train_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """Fitting a head when no sample matches any class index raises a clear error."""
        x, _ = linear_train_data
        out = mahalanobis(flax_custom_model, predictor_type="logit_classifier")
        out_of_range = jnp.full((len(x),), CUSTOM_NUM_CLASSES)
        with pytest.raises(ValueError, match="no labelled samples"):
            out.fit_mahalanobis_heads(x, out_of_range)


# ---------------------------------------------------------------------------
# Convolutional features and global-average pooling
# ---------------------------------------------------------------------------


class TestConvFeatures:
    """The conv path: channels-last feature maps are global-average-pooled to (N, C)."""

    def test_feature_node_pools_spatial_map(
        self, conv_model: nnx.Module, conv_train_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """A feature node on the conv layer pools its ``(N, H, W, C)`` output to ``(N, C)``.

        Flax convolutions are channels-last, so a correct pool reduces the middle
        axes; reducing the trailing ones (as the torch backend does) would leave
        the head with a spatial rather than a channel dimension.
        """
        x, y = conv_train_data
        out = mahalanobis(conv_model, feature_nodes=["conv"], predictor_type="logit_classifier")
        out.fit_mahalanobis_heads(x, y)
        assert len(out.mahalanobis_heads) == 2
        assert out.mahalanobis_heads[0].feature_dim == CONV_FEATURE_DIM
        logits, scores = out(x)
        assert logits.shape == (len(x), CONV_NUM_CLASSES)
        assert scores.shape == (len(x), 2)


# ---------------------------------------------------------------------------
# Combiner calibration and input preprocessing
# ---------------------------------------------------------------------------


class TestCombinerAndPreprocessing:
    """The logistic-regression combiner and the FGSM input preprocessing."""

    def test_fit_combiner_separates_in_and_out(
        self,
        flax_custom_model: nnx.Module,
        linear_train_data: tuple[jax.Array, jax.Array],
        ood_data: jax.Array,
    ) -> None:
        """After calibration, out-of-distribution inputs score higher than in-distribution."""
        x, y = linear_train_data
        out = mahalanobis(flax_custom_model, predictor_type="logit_classifier")
        out.fit_mahalanobis_heads(x, y)
        out.fit_combiner(x, ood_data, steps=300)
        id_score = decompose(predict(out, x)).epistemic.mean()
        ood_score = decompose(predict(out, ood_data)).epistemic.mean()
        assert float(id_score) < float(ood_score)

    def test_fit_combiner_updates_the_defaults(
        self,
        flax_custom_model: nnx.Module,
        linear_train_data: tuple[jax.Array, jax.Array],
        ood_data: jax.Array,
    ) -> None:
        """Calibration replaces the ``-1``/``0`` defaults with fitted values."""
        x, y = linear_train_data
        out = mahalanobis(flax_custom_model, predictor_type="logit_classifier")
        out.fit_mahalanobis_heads(x, y)
        out.fit_combiner(x, ood_data, steps=100)
        assert out.combiner_weight.shape == (1,)
        assert not bool(jnp.allclose(out.combiner_weight, -jnp.ones(1)))
        assert not bool(jnp.allclose(out.combiner_bias, jnp.zeros(())))

    def test_preprocessing_raises_the_confidence(
        self, flax_custom_model: nnx.Module, linear_train_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """FGSM nudges the input along the confidence gradient, so scores do not drop."""
        x, y = linear_train_data
        out = mahalanobis(flax_custom_model, input_preprocessing_eps=0.01, predictor_type="logit_classifier")
        out.fit_mahalanobis_heads(x, y)
        _, plain = out(x)
        preprocessed = out.predict_representation(x).layer_scores
        assert preprocessed.shape == plain.shape
        assert not bool(jnp.allclose(preprocessed, plain))
        assert bool((preprocessed >= plain - 1e-4).all())

    def test_zero_eps_disables_preprocessing(
        self, flax_custom_model: nnx.Module, linear_train_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """With ``eps == 0`` the representation carries the plain confidences."""
        x, y = linear_train_data
        out = mahalanobis(flax_custom_model, predictor_type="logit_classifier")
        out.fit_mahalanobis_heads(x, y)
        _, plain = out(x)
        assert jnp.allclose(out.predict_representation(x).layer_scores, plain)


# ---------------------------------------------------------------------------
# Cross-backend agreement
# ---------------------------------------------------------------------------


class TestHeadEquivalence:
    """The flax head estimates the same Gaussian parameters as the numpy backend."""

    @pytest.fixture
    def labelled_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Well-conditioned features with every class populated."""
        rng = np.random.default_rng(123)
        return rng.standard_normal((200, 5)), rng.integers(0, 3, size=200)

    def test_flax_and_numpy_heads_agree(self, labelled_features: tuple[np.ndarray, np.ndarray]) -> None:
        """Both backends produce the same means, precision and confidences.

        Tolerances are loose because the flax head runs in float32 while the numpy
        head runs in float64, and ``pinv`` picks its rank cutoff from the dtype.
        """
        features, labels = labelled_features

        array_head = NumpyMahalanobisHead(3, 5)
        array_head.fit(features, labels)

        flax_head = MahalanobisHead(3, 5)
        flax_head.fit(jnp.asarray(features), jnp.asarray(labels))

        np.testing.assert_allclose(np.asarray(flax_head.means), array_head.means, atol=1e-5)
        np.testing.assert_allclose(np.asarray(flax_head.precision), array_head.precision, atol=1e-4)
        np.testing.assert_allclose(np.asarray(flax_head(jnp.asarray(features))), array_head.score(features), atol=1e-3)

    def test_out_of_range_labels_are_ignored(self, labelled_features: tuple[np.ndarray, np.ndarray]) -> None:
        """Samples labelled outside ``[0, num_classes)`` do not contribute to the fit."""
        features, labels = labelled_features
        mixed = labels.copy()
        mixed[:50] = 99

        flax_head = MahalanobisHead(3, 5)
        flax_head.fit(jnp.asarray(features), jnp.asarray(mixed))

        valid_only = NumpyMahalanobisHead(3, 5)
        valid_only.fit(features[50:], labels[50:])

        np.testing.assert_allclose(np.asarray(flax_head.means), valid_only.means, atol=1e-5)
