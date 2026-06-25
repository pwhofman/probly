"""Tests for the torch Mahalanobis OOD implementation."""

from __future__ import annotations

import pytest

from probly.decider import categorical_from_mean
from probly.method.mahalanobis import mahalanobis
from probly.predictor import predict

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution  # noqa: E402

# ``torch_custom_model``: Linear(10, 20) -> ReLU -> Linear(20, 4) -> Softmax.
CUSTOM_IN_FEATURES = 10
CUSTOM_NUM_CLASSES = 4
# ``torch_conv_linear_model``: Conv2d(3, 5, 5) -> ReLU -> Flatten -> Linear(5, 2).
CONV_NUM_CLASSES = 2
CONV_FEATURE_DIM = 5


# ---------------------------------------------------------------------------
# Local data (the shared sample is too small to estimate a covariance)
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_train_data() -> tuple[Tensor, Tensor]:
    """A multi-class training batch shaped for ``torch_custom_model``."""
    torch.manual_seed(0)
    inputs = torch.randn(80, CUSTOM_IN_FEATURES)
    targets = torch.randint(0, CUSTOM_NUM_CLASSES, (80,))
    return inputs, targets


@pytest.fixture
def conv_train_data() -> tuple[Tensor, Tensor]:
    """A training batch shaped for ``torch_conv_linear_model``."""
    torch.manual_seed(0)
    inputs = torch.randn(60, 3, 5, 5)
    targets = torch.randint(0, CONV_NUM_CLASSES, (60,))
    return inputs, targets


# ---------------------------------------------------------------------------
# Transformation structure
# ---------------------------------------------------------------------------


class TestTransformation:
    """The mahalanobis transformation rewires the model into an encoder + head."""

    def test_head_replaced_with_identity(self, torch_custom_model: nn.Module) -> None:
        """The last Linear is stripped from the encoder and kept as classification head."""
        out = mahalanobis(torch_custom_model)
        assert isinstance(out.encoder.get_submodule("linear2"), nn.Identity)
        assert isinstance(out.classification_head, nn.Linear)
        assert out.classification_head.out_features == CUSTOM_NUM_CLASSES

    def test_head_found_in_sequential(self, torch_model_small_2d_2d: nn.Module) -> None:
        """In a plain Linear stack the final Linear becomes the classification head."""
        out = mahalanobis(torch_model_small_2d_2d)
        assert isinstance(out.encoder.get_submodule("2"), nn.Identity)
        assert isinstance(out.classification_head, nn.Linear)

    def test_no_spectral_normalisation(self, torch_custom_model: nn.Module) -> None:
        """Unlike DDU, the plain feature extractor adds no weight parametrizations."""
        out = mahalanobis(torch_custom_model)
        assert not any(hasattr(m, "parametrizations") and "weight" in m.parametrizations for m in out.modules())

    def test_no_linear_raises(self) -> None:
        """A model without a Linear layer cannot identify a classification head."""
        with pytest.raises(ValueError, match=r"No nn\.Linear"):
            mahalanobis(nn.Sequential(nn.ReLU(), nn.Tanh()), predictor_type="logit_classifier")


# ---------------------------------------------------------------------------
# Fitting and forward pass
# ---------------------------------------------------------------------------


class TestFitAndForward:
    """Fitting the Mahalanobis heads and running the forward pass."""

    def test_single_layer_shapes(self, torch_custom_model: nn.Module, linear_train_data: tuple[Tensor, Tensor]) -> None:
        """Default (no feature nodes) yields a single feature layer."""
        x, y = linear_train_data
        out = mahalanobis(torch_custom_model)
        out.fit_mahalanobis_heads(x, y)
        assert len(out.mahalanobis_heads) == 1
        logits, scores = out(x)
        assert logits.shape == (len(x), CUSTOM_NUM_CLASSES)
        assert scores.shape == (len(x), 1)

    def test_multi_layer_shapes(self, torch_custom_model: nn.Module, linear_train_data: tuple[Tensor, Tensor]) -> None:
        """An extra feature node adds a layer to the ensemble."""
        x, y = linear_train_data
        out = mahalanobis(torch_custom_model, feature_nodes=["linear1"])
        out.fit_mahalanobis_heads(x, y)
        assert len(out.mahalanobis_heads) == 2
        _, scores = out(x)
        assert scores.shape == (len(x), 2)
        assert out.combiner_weight.shape == (2,)

    def test_fit_populates_parameters(
        self, torch_custom_model: nn.Module, linear_train_data: tuple[Tensor, Tensor]
    ) -> None:
        """Fitting sets non-trivial class means and a non-identity precision."""
        x, y = linear_train_data
        out = mahalanobis(torch_custom_model)
        out.fit_mahalanobis_heads(x, y)
        head = out.mahalanobis_heads[0]
        assert torch.any(head.means != 0)
        assert not torch.allclose(head.precision, torch.eye(head.feature_dim))

    def test_categorical_from_mean_returns_softmax(
        self, torch_custom_model: nn.Module, linear_train_data: tuple[Tensor, Tensor]
    ) -> None:
        """The categorical mean decider reduces the representation to its softmax."""
        x, y = linear_train_data
        out = mahalanobis(torch_custom_model)
        out.fit_mahalanobis_heads(x, y)
        single = categorical_from_mean(predict(out, x))
        logits, _ = out(x)
        assert isinstance(single, TorchCategoricalDistribution)
        assert torch.allclose(single.probabilities, torch.softmax(logits, dim=-1))

    def test_fit_without_matching_labels_raises(
        self, torch_custom_model: nn.Module, linear_train_data: tuple[Tensor, Tensor]
    ) -> None:
        """Fitting a head when no sample matches any class index raises a clear error."""
        x, _ = linear_train_data
        out = mahalanobis(torch_custom_model)
        out_of_range = torch.full((len(x),), CUSTOM_NUM_CLASSES)
        with pytest.raises(ValueError, match="no labelled samples"):
            out.fit_mahalanobis_heads(x, out_of_range)


# ---------------------------------------------------------------------------
# Convolutional features and global-average pooling
# ---------------------------------------------------------------------------


class TestConvFeatures:
    """The conv path: spatial feature maps are global-average-pooled to (N, C)."""

    def test_pooled_penultimate_shapes(
        self, torch_conv_linear_model: nn.Module, conv_train_data: tuple[Tensor, Tensor]
    ) -> None:
        """Conv-derived penultimate features fit a head of the channel dimension."""
        x, y = conv_train_data
        out = mahalanobis(torch_conv_linear_model)
        out.fit_mahalanobis_heads(x, y)
        head = out.mahalanobis_heads[0]
        assert head.feature_dim == CONV_FEATURE_DIM
        logits, scores = out(x)
        assert logits.shape == (len(x), CONV_NUM_CLASSES)
        assert scores.shape == (len(x), 1)

    def test_feature_node_pools_spatial_map(
        self, torch_conv_linear_model: nn.Module, conv_train_data: tuple[Tensor, Tensor]
    ) -> None:
        """A feature node on the conv layer pools its 4-D output to (N, channels)."""
        x, y = conv_train_data
        out = mahalanobis(torch_conv_linear_model, feature_nodes=["0"])
        out.fit_mahalanobis_heads(x, y)
        assert len(out.mahalanobis_heads) == 2
        # A successful fit on the conv node proves the spatial map was pooled to (N, C).
        assert out.mahalanobis_heads[0].feature_dim == CONV_FEATURE_DIM
        _, scores = out(x)
        assert scores.shape == (len(x), 2)

    def test_sample_classification_data_smoke(
        self, torch_conv_linear_model: nn.Module, sample_classification_data: tuple[Tensor, Tensor]
    ) -> None:
        """The shared two-sample batch flows through fit and forward without error."""
        x, y = sample_classification_data
        out = mahalanobis(torch_conv_linear_model)
        out.fit_mahalanobis_heads(x, y)
        logits, scores = out(x)
        assert logits.shape == (len(x), CONV_NUM_CLASSES)
        assert scores.shape == (len(x), 1)


# ---------------------------------------------------------------------------
# Combiner calibration and input preprocessing
# ---------------------------------------------------------------------------


class TestCombinerAndPreprocessing:
    """The logistic-regression combiner and the FGSM input preprocessing."""

    def test_fit_combiner_separates_in_and_out(
        self, torch_custom_model: nn.Module, linear_train_data: tuple[Tensor, Tensor]
    ) -> None:
        """After calibration, out-of-distribution inputs score higher than in-distribution."""
        x, y = linear_train_data
        out = mahalanobis(torch_custom_model, feature_nodes=["linear1"])
        out.fit_mahalanobis_heads(x, y)
        ood = torch.randn(80, CUSTOM_IN_FEATURES) * 6 + 25
        out.fit_combiner(x, ood, steps=300)

        id_score = predict(out, x).layer_scores @ out.combiner_weight + out.combiner_bias
        ood_score = predict(out, ood).layer_scores @ out.combiner_weight + out.combiner_bias
        assert ood_score.mean() > id_score.mean()

    def test_input_preprocessing_runs(
        self, torch_conv_linear_model: nn.Module, conv_train_data: tuple[Tensor, Tensor]
    ) -> None:
        """A positive preprocessing epsilon produces finite, correctly shaped scores."""
        x, y = conv_train_data
        out = mahalanobis(torch_conv_linear_model, input_preprocessing_eps=0.01)
        out.fit_mahalanobis_heads(x, y)
        rep = predict(out, x)
        assert rep.layer_scores.shape == (len(x), 1)
        assert torch.isfinite(rep.layer_scores).all()

    def test_input_preprocessing_changes_scores(
        self, torch_conv_linear_model: nn.Module, conv_train_data: tuple[Tensor, Tensor]
    ) -> None:
        """The FGSM perturbation actually moves the scores; eps>0 differs from eps=0 on identical inputs."""
        x, y = conv_train_data
        out = mahalanobis(torch_conv_linear_model, input_preprocessing_eps=0.01)
        out.fit_mahalanobis_heads(x, y)
        # Toggle eps on the same fitted predictor so encoder and heads are identical:
        # any difference is attributable to the FGSM preprocessing alone.
        perturbed = predict(out, x).layer_scores
        out.input_preprocessing_eps = 0.0
        plain = predict(out, x).layer_scores
        assert not torch.allclose(perturbed, plain)

    def test_fit_combiner_with_preprocessing(
        self, torch_custom_model: nn.Module, linear_train_data: tuple[Tensor, Tensor]
    ) -> None:
        """With preprocessing on, the combiner calibrates on the same FGSM scores used at inference."""
        x, y = linear_train_data
        out = mahalanobis(torch_custom_model, input_preprocessing_eps=0.01)
        out.fit_mahalanobis_heads(x, y)
        ood = torch.randn(80, CUSTOM_IN_FEATURES) * 6 + 25
        out.fit_combiner(x, ood, steps=200)

        assert torch.isfinite(out.combiner_weight).all()
        id_score = predict(out, x).layer_scores @ out.combiner_weight + out.combiner_bias
        ood_score = predict(out, ood).layer_scores @ out.combiner_weight + out.combiner_bias
        assert ood_score.mean() > id_score.mean()


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


class TestRepresentation:
    """The TorchMahalanobisRepresentation dataclass."""

    def test_indexing_preserves_combiner(
        self, torch_custom_model: nn.Module, linear_train_data: tuple[Tensor, Tensor]
    ) -> None:
        """Indexing batches the per-sample fields but keeps the shared combiner weights."""
        x, y = linear_train_data
        out = mahalanobis(torch_custom_model, feature_nodes=["linear1"])
        out.fit_mahalanobis_heads(x, y)
        rep = predict(out, x)
        sub = rep[:4]
        assert sub.layer_scores.shape == (4, 2)
        assert torch.equal(sub.weight, rep.weight)
        assert torch.equal(sub.bias, rep.bias)
