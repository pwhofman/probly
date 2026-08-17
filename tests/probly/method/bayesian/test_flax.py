"""Test for flax bayesian models."""

from __future__ import annotations

from typing import cast

from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from probly.layers.flax import BayesConv, BayesLinear, _inverse_softplus, _kl_divergence_gaussian
from probly.transformation.bayesian import bayesian
from tests.probly.flax_utils import count_layers

use_base_weights = False
posterior_std = 0.05
prior_mean = 0.0
prior_std = 1.0


class TestBayesianAttributes:
    def test_bayesian_linear_attributes(self) -> None:
        rngs = nnx.Rngs(0, params=1)
        linear_layer = nnx.Linear(2, 2, rngs=rngs)

        bayes_layer = BayesLinear(
            linear_layer,
            use_base_weights=use_base_weights,
            posterior_std=posterior_std,
            prior_mean=prior_mean,
            prior_std=prior_std,
            rngs=rngs,
        )

        assert bayes_layer.precision == linear_layer.precision
        assert bayes_layer.dtype == linear_layer.dtype
        assert bayes_layer.param_dtype == linear_layer.param_dtype
        assert bayes_layer.in_features == linear_layer.in_features
        assert bayes_layer.out_features == linear_layer.out_features
        assert bayes_layer.use_bias == linear_layer.use_bias
        assert bayes_layer.dot_general == linear_layer.dot_general
        assert bayes_layer.promote_dtype == linear_layer.promote_dtype
        assert bayes_layer.rng_collection == "bayesian"

        assert isinstance(bayes_layer.weight_rho, nnx.Param)

        # `bias` is a presence flag (bool), not the Param itself like on `nnx.Linear`.
        assert bayes_layer.bias is not None
        assert bayes_layer.bias == (linear_layer.bias is not None)

        weight_shape = (linear_layer.in_features, linear_layer.out_features)
        rho = _inverse_softplus(jnp.array(posterior_std))
        assert bayes_layer.weight_mu.shape == weight_shape
        assert bayes_layer.weight_rho.shape == weight_shape
        assert jnp.allclose(bayes_layer.weight_rho[...], jnp.full(weight_shape, rho))
        assert jnp.allclose(bayes_layer.weight_prior_mu[...], jnp.full(weight_shape, prior_mean))
        assert jnp.allclose(bayes_layer.weight_prior_sigma[...], jnp.full(weight_shape, prior_std))

        bias_shape = (linear_layer.out_features,)
        assert bayes_layer.bias_mu.shape == bias_shape
        assert bayes_layer.bias_rho.shape == bias_shape
        assert jnp.allclose(bayes_layer.bias_mu[...], jnp.zeros(bias_shape))
        assert jnp.allclose(bayes_layer.bias_rho[...], jnp.full(bias_shape, rho))
        assert jnp.allclose(bayes_layer.bias_prior_mu[...], jnp.full(bias_shape, prior_mean))
        assert jnp.allclose(bayes_layer.bias_prior_sigma[...], jnp.full(bias_shape, prior_std))

    def test_bayesian_linear_attributes_use_base_weights(self) -> None:
        rngs = nnx.Rngs(0, params=1)
        linear_layer = nnx.Linear(2, 2, rngs=rngs)

        bayes_layer = BayesLinear(linear_layer, use_base_weights=True, rngs=rngs)

        # With use_base_weights=True, both the posterior mean and the prior mean are
        # copied from the base layer's parameters instead of being freshly initialized.
        assert jnp.array_equal(bayes_layer.weight_mu[...], linear_layer.kernel[...])
        assert jnp.array_equal(bayes_layer.weight_prior_mu[...], linear_layer.kernel[...])

        assert linear_layer.bias is not None
        assert jnp.array_equal(bayes_layer.bias_mu[...], linear_layer.bias[...])
        assert jnp.array_equal(bayes_layer.bias_prior_mu[...], linear_layer.bias[...])

    def test_bayesian_linear_attributes_no_bias(self) -> None:
        rngs = nnx.Rngs(0, params=1)
        linear_layer = nnx.Linear(2, 2, use_bias=False, rngs=rngs)

        bayes_layer = BayesLinear(linear_layer, use_base_weights=False, rngs=rngs)

        assert bayes_layer.bias is False
        assert bayes_layer.bias_mu is None
        assert bayes_layer.bias_rho is None
        assert bayes_layer.bias_prior_mu is None
        assert bayes_layer.bias_prior_sigma is None


class TestRngs:
    def test_init_rngs(self) -> None:
        rngs = nnx.Rngs(0, params=1)
        linear_layer = nnx.Linear(1, 2, rngs=rngs)
        rng_stream = nnx.RngStream(key=1, tag="bayesian")

        bayesian_layer_rng_stream = BayesLinear(linear_layer, rngs=rng_stream)

        new_rng_stream = nnx.RngStream(key=1, tag="bayesian")
        used_rngs = new_rng_stream.fork()
        assert bayesian_layer_rng_stream.rngs.key.value == used_rngs.key.value
        assert isinstance(bayesian_layer_rng_stream.rngs, nnx.rnglib.RngStream)

        msg = f"rngs must be a RNGS, RngStream or None, but got {str}"
        with pytest.raises(TypeError, match=msg):
            BayesLinear(linear_layer, rngs="test")

    def test_rngs_none_without_base_weights_raises(self) -> None:
        rngs = nnx.Rngs(0, params=1)
        linear_layer = nnx.Linear(1, 2, rngs=rngs)

        msg = "rngs must be provided when use_base_weights is False, to initialize new weights."
        with pytest.raises(ValueError, match=msg):
            BayesLinear(linear_layer, use_base_weights=False, rngs=None)

    def test_rngs_none_with_base_weights_defers_error_to_call(self) -> None:
        rngs = nnx.Rngs(0, params=1)
        linear_layer = nnx.Linear(1, 2, rngs=rngs)

        bayesian_layer = BayesLinear(linear_layer, use_base_weights=True, rngs=None)
        assert bayesian_layer.rngs is None

        msg = """No `rngs` argument was provided to BayesianLinear
                as either a __call__ argument or class attribute."""
        with pytest.raises(ValueError, match=msg):
            bayesian_layer(jnp.ones(1))

    def test_init_rngs_stream(self) -> None:
        rngs = nnx.Rngs(0, params=1, bayesian=2)
        linear_layer = nnx.Linear(1, 2, rngs=rngs)
        bayesian_layer = BayesLinear(linear_layer, rngs=rngs)

        new_rngs = nnx.Rngs(0, params=1, bayesian=2)
        bayesian_rngs = new_rngs["bayesian"].fork()
        assert bayesian_layer.rngs.tag == "bayesian"
        assert bayesian_layer.rngs.key.value == bayesian_rngs.key.value
        assert isinstance(bayesian_layer.rngs, nnx.rnglib.RngStream)


class TestHelperFunctions:
    def test_inverse_softplus_return_type(self) -> None:
        x = jnp.ones((1,))
        inverse_softplus = _inverse_softplus(x)
        assert isinstance(inverse_softplus, jax.Array)

    def test_kl_divergence_gaussian_return_type(self) -> None:
        mu1 = jnp.array([0.0])
        var1 = jnp.array([1.0])
        mu2 = jnp.array([2.0])
        var2 = jnp.array([1.0])

        kl = _kl_divergence_gaussian(mu1, var1, mu2, var2)

        assert isinstance(kl, jax.Array)
        assert kl.shape == (1,)

    def test_kl_divergence_gaussian_zero_when_distributions_equal(self) -> None:
        mu = jnp.array([0.0, 1.0])
        var = jnp.array([1.0, 2.0])

        kl = _kl_divergence_gaussian(mu, var, mu, var)

        assert jnp.allclose(kl, jnp.zeros_like(kl), atol=1e-6)

    def test_kl_divergence_gaussian_exact_value(self) -> None:
        # KL(N(0, 1) || N(2, 1)) = 0.5*log(1/1) + (1 + (0-2)**2)/(2*1) - 0.5 = 2.0 (exact).
        mu1 = jnp.array([0.0])
        var1 = jnp.array([1.0])
        mu2 = jnp.array([2.0])
        var2 = jnp.array([1.0])

        kl = _kl_divergence_gaussian(mu1, var1, mu2, var2)

        assert jnp.allclose(kl, jnp.array([2.0]))


class TestNetworkArchitecture:
    def test_replace_linear_layer(self, flax_model_small_2d_2d: nnx.Module) -> None:
        model = cast("nnx.Module", bayesian(flax_model_small_2d_2d))

        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        count_bayes_modified = count_layers(model, BayesLinear)
        count_linear_modified = count_layers(model, nnx.Linear)

        assert isinstance(model, type(flax_model_small_2d_2d))
        assert count_linear_modified == 0
        assert count_bayes_modified == count_linear_original

    def test_replace_conv_layer(self, flax_conv_linear_model: nnx.Module) -> None:
        """Tests that plain ``nnx.Conv`` layers are replaced by ``BayesConv``.

        ``BayesConv`` subclasses ``nnx.Conv`` (to reuse its ``__call__``), so counting
        ``nnx.Conv`` instances after the transform also counts the replaced ``BayesConv``
        layers; the original layer count is used as the point of comparison instead.
        """
        count_conv_original = count_layers(flax_conv_linear_model, nnx.Conv)
        count_linear_original = count_layers(flax_conv_linear_model, nnx.Linear)

        model = cast("nnx.Module", bayesian(flax_conv_linear_model))

        count_bayes_conv_modified = count_layers(model, BayesConv)
        count_bayes_linear_modified = count_layers(model, BayesLinear)
        count_linear_modified = count_layers(model, nnx.Linear)

        assert isinstance(model, type(flax_conv_linear_model))
        assert count_linear_modified == 0
        assert count_bayes_conv_modified == count_conv_original
        assert count_bayes_linear_modified == count_linear_original


class TestBayesConvCall:
    """Test class for the ``BayesConv`` forward pass."""

    def test_call(self, flax_rngs: nnx.Rngs) -> None:
        conv = nnx.Conv(3, 5, (5, 5), rngs=flax_rngs)
        bayes_conv = BayesConv(conv, rngs=nnx.Rngs(1))

        x = jnp.ones((2, 16, 16, 3))
        y = bayes_conv(x, rngs=nnx.Rngs(2))

        assert y is not None
        assert y.shape == (2, 16, 16, 5)

    def test_kl_divergence(self, flax_rngs: nnx.Rngs) -> None:
        conv = nnx.Conv(3, 5, (5, 5), rngs=flax_rngs)
        bayes_conv = BayesConv(conv, rngs=nnx.Rngs(1))

        kl = bayes_conv.kl_divergence

        assert kl is not None
        assert kl.shape == ()
        assert kl >= 0
