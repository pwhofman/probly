"""Tests for flax SNGP transformation."""

from __future__ import annotations

from typing import cast

import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly.layers.flax import SNGPLayer, SpectralNormWithMultiplier  # noqa: E402
from probly.method.sngp import sngp  # noqa: E402
from probly.quantification import decompose  # noqa: E402
from probly.quantification.decomposition.entropy import SecondOrderEntropyDecomposition  # noqa: E402
from probly.representation.distribution.jax_categorical import JaxCategoricalDistributionSample  # noqa: E402
from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: E402
from probly.representer import representer  # noqa: E402
from tests.probly.flax_utils import count_layers  # noqa: E402


class TestSNGPLayerReplacement:
    """Tests for the structural changes ``sngp`` applies to a flax model."""

    def test_replaces_last_linear_with_sngp_layer(self, flax_model_small_2d_2d: nnx.Module) -> None:
        """The final ``nnx.Linear`` is replaced and earlier ones are wrapped."""
        original = flax_model_small_2d_2d
        original_linear_count = count_layers(original, nnx.Linear)

        new_model = cast("nnx.Module", sngp(original, num_inducing=16))

        sngp_count = count_layers(new_model, SNGPLayer)
        spectral_norm_count = count_layers(new_model, SpectralNormWithMultiplier)

        assert sngp_count == 1
        # All ``Linear`` layers but the last one are wrapped with the spectral-
        # norm parameterization. The last is replaced by the SNGPLayer.
        assert spectral_norm_count == original_linear_count - 1

    def test_returns_a_clone(self, flax_model_small_2d_2d: nnx.Module) -> None:
        """``sngp`` does not mutate the input model."""
        new_model = sngp(flax_model_small_2d_2d, num_inducing=16)
        assert new_model is not flax_model_small_2d_2d

    def test_layer_order_earlier_wrapped_last_replaced(self, flax_model_small_2d_2d: nnx.Module) -> None:
        """Earlier layers are spectral-norm-wrapped; the trailing layer is replaced."""
        new_model = cast("nnx.Sequential", sngp(flax_model_small_2d_2d, num_inducing=16))
        layers = list(new_model.layers)

        assert isinstance(layers[-1], SNGPLayer)
        wrapped = [layer for layer in layers[:-1] if isinstance(layer, SpectralNormWithMultiplier)]
        # All earlier ``nnx.Linear`` layers (before activations etc.) are wrapped.
        assert len(wrapped) == count_layers(flax_model_small_2d_2d, nnx.Linear) - 1

    def test_wraps_conv_layers_with_spectral_norm(self, flax_conv_linear_model: nnx.Module) -> None:
        """``nnx.Conv`` layers are spectrally normalized."""
        new_model = cast("nnx.Module", sngp(flax_conv_linear_model, num_inducing=16))

        sngp_count = count_layers(new_model, SNGPLayer)
        spectral_norm_count = count_layers(new_model, SpectralNormWithMultiplier)

        assert sngp_count == 1
        # The Conv layer plus any non-final Linear layer is wrapped.
        assert spectral_norm_count >= 1


class TestSNGPForwardPass:
    """Tests that the transformed model produces ``(logits, variance)`` of the expected shape."""

    def test_raw_forward_shapes(self, flax_regression_model_2d: nnx.Module) -> None:
        """The raw forward returns ``(logits, variance)`` each of shape ``(batch, num_classes)``."""
        new_model = cast("nnx.Module", sngp(flax_regression_model_2d, num_inducing=16))
        out = new_model(jnp.ones((5, 4)))

        assert isinstance(out, tuple)
        logits, variance = out
        assert logits.shape == (5, 2)
        assert variance.shape == (5, 2)

    def test_predict_returns_gaussian_distribution(self, flax_regression_model_2d: nnx.Module) -> None:
        """``predict()`` on the wrapped predictor returns a Gaussian distribution."""
        from probly.predictor import predict  # noqa: PLC0415

        predictor = sngp(flax_regression_model_2d, num_inducing=16)
        distribution = predict(predictor, jnp.ones((5, 4)))

        assert isinstance(distribution, JaxGaussianDistribution)
        assert distribution.mean.shape == (5, 2)
        assert distribution.var.shape == (5, 2)


class TestSNGPSoftmaxTailStripping:
    """Tests that ``sngp`` strips a trailing softmax callable from a ``Sequential``."""

    def test_trailing_jax_softmax_is_removed(self, flax_rngs: nnx.Rngs) -> None:
        """A trailing ``jax.nn.softmax`` is dropped so the SNGPLayer is the tail."""
        model = nnx.Sequential(
            nnx.Linear(4, 2, rngs=flax_rngs),
            jax.nn.softmax,
        )
        new_model = cast("nnx.Sequential", sngp(model, num_inducing=16))

        layers = list(new_model.layers)
        assert isinstance(layers[-1], SNGPLayer)
        assert jax.nn.softmax not in layers


class TestSNGPRngsParameter:
    """Tests that distinct ``rngs`` arguments produce distinct ``SNGPLayer`` params."""

    def _find_sngp_layer(self, model: nnx.Module) -> SNGPLayer:
        """Locate the single ``SNGPLayer`` in a transformed model."""
        seq = cast("nnx.Sequential", model)
        for layer in seq.layers:
            if isinstance(layer, SNGPLayer):
                return layer
        msg = "no SNGPLayer found in transformed model"
        raise AssertionError(msg)

    def test_distinct_rngs_yield_distinct_layers(self, flax_regression_model_2d: nnx.Module) -> None:
        """Different ``rngs`` seeds produce distinct RFF projection weights."""
        m1 = sngp(flax_regression_model_2d, num_inducing=16, rngs=nnx.Rngs(0))
        m2 = sngp(flax_regression_model_2d, num_inducing=16, rngs=nnx.Rngs(42))

        layer_1 = self._find_sngp_layer(cast("nnx.Module", m1))
        layer_2 = self._find_sngp_layer(cast("nnx.Module", m2))

        assert not jnp.allclose(layer_1.W_L.value, layer_2.W_L.value)


class TestSNGPInferenceMode:
    """Tests that ``update_covariance=False`` freezes the precision-matrix EMA."""

    def test_precision_matrix_unchanged_when_update_false(self, flax_regression_model_2d: nnx.Module) -> None:
        """``update_covariance=False`` leaves ``precision_matrix`` untouched between calls."""
        new_model = cast("nnx.Sequential", sngp(flax_regression_model_2d, num_inducing=16))
        sngp_layer = next(layer for layer in new_model.layers if isinstance(layer, SNGPLayer))

        # Call the SNGPLayer directly with ``update_covariance=False`` since
        # ``nnx.Sequential.__call__`` does not propagate keyword arguments.
        precision_before = jnp.array(sngp_layer.precision_matrix.value)
        sngp_layer(jnp.ones((4, sngp_layer.in_features)), update_covariance=False)
        precision_after = jnp.array(sngp_layer.precision_matrix.value)

        assert jnp.array_equal(precision_before, precision_after)

    def test_precision_matrix_changes_when_update_true(self, flax_regression_model_2d: nnx.Module) -> None:
        """``update_covariance=True`` mutates ``precision_matrix`` between calls."""
        new_model = cast("nnx.Sequential", sngp(flax_regression_model_2d, num_inducing=16))
        sngp_layer = next(layer for layer in new_model.layers if isinstance(layer, SNGPLayer))

        precision_before = jnp.array(sngp_layer.precision_matrix.value)
        sngp_layer(jnp.ones((4, sngp_layer.in_features)), update_covariance=True)
        precision_after = jnp.array(sngp_layer.precision_matrix.value)

        assert not jnp.array_equal(precision_before, precision_after)


class TestSNGPRepresenter:
    """Smoke tests for the SNGP representer/decomposition chain on the flax backend."""

    def test_representer_returns_categorical_sample(self, flax_rngs: nnx.Rngs) -> None:
        """Sampling through the representer yields a ``JaxCategoricalDistributionSample``."""
        model = nnx.Sequential(
            nnx.Linear(2, 4, rngs=flax_rngs),
            nnx.Linear(4, 3, rngs=flax_rngs),
        )
        predictor = sngp(model, num_inducing=16)
        sample = representer(predictor, num_samples=3).represent(jnp.ones((2, 2)))

        assert isinstance(sample, JaxCategoricalDistributionSample)

    def test_decompose_dispatches_to_second_order_entropy(self, flax_rngs: nnx.Rngs) -> None:
        """``decompose`` of an SNGP sample routes to ``SecondOrderEntropyDecomposition``."""
        model = nnx.Sequential(
            nnx.Linear(2, 4, rngs=flax_rngs),
            nnx.Linear(4, 3, rngs=flax_rngs),
        )
        predictor = sngp(model, num_inducing=16)
        sample = representer(predictor, num_samples=3).represent(jnp.ones((2, 2)))
        decomposition = decompose(sample)

        assert isinstance(decomposition, SecondOrderEntropyDecomposition)
