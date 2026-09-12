"""Test for flax bayesian models."""

from __future__ import annotations

from typing import cast

import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly.layers.flax import (  # noqa: E402
    BayesConv,
    BayesLinear,
    _copy_conv_attrs,
    _inverse_softplus,
    _kl_divergence_gaussian,
)
from probly.transformation.bayesian import bayesian  # noqa: E402
from tests.probly.flax_utils import count_layers  # noqa: E402

use_base_weights = False
posterior_std = 0.05
prior_mean = 0.0
prior_std = 1.0


class TestBayesianAttributes:
    """Test calls for Bayesian attributes."""

    def test_bayesian_linear_attributes(self) -> None:
        """Tests the BayesianLinear layer attributes."""
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
        """Tests BayesianLinear layer attributes for use_base_weights True."""
        rngs = nnx.Rngs(0, params=1)
        linear_layer = nnx.Linear(2, 2, rngs=rngs)

        bayes_layer = BayesLinear(linear_layer, use_base_weights=True, rngs=rngs)

        # Posterior mean dna prior mean are copied from base layer's attributes
        assert jnp.array_equal(bayes_layer.weight_mu[...], linear_layer.kernel[...])
        assert jnp.array_equal(bayes_layer.weight_prior_mu[...], linear_layer.kernel[...])

        assert linear_layer.bias is not None
        assert jnp.array_equal(bayes_layer.bias_mu[...], linear_layer.bias[...])
        assert jnp.array_equal(bayes_layer.bias_prior_mu[...], linear_layer.bias[...])

    def test_bayesian_linear_attributes_no_bias(self) -> None:
        """Tests BayesianLinear layer attributes for use_bias False."""
        rngs = nnx.Rngs(0, params=1)
        linear_layer = nnx.Linear(2, 2, use_bias=False, rngs=rngs)

        bayes_layer = BayesLinear(linear_layer, use_base_weights=False, rngs=rngs)

        assert bayes_layer.bias is False
        assert bayes_layer.bias_mu is None
        assert bayes_layer.bias_rho is None
        assert bayes_layer.bias_prior_mu is None
        assert bayes_layer.bias_prior_sigma is None

    def test_bayesian_conv_attributes(self) -> None:
        """Tests BayesianConv layer attributes."""
        rngs = nnx.Rngs(0, params=1)
        conv_layer = nnx.Conv(3, 4, (3,), padding="VALID", rngs=rngs)
        bayes_layer = BayesConv(
            conv_layer,
            use_base_weights=False,
            posterior_std=posterior_std,
            prior_mean=prior_mean,
            prior_std=prior_std,
            rngs=rngs,
        )
        assert bayes_layer.rng_collection == "bayesian"

        # Attributes are set by `_copy_conv_attrs`
        assert bayes_layer.kernel_shape == conv_layer.kernel_shape
        assert bayes_layer.in_features == conv_layer.in_features
        assert bayes_layer.out_features == conv_layer.out_features
        assert bayes_layer.kernel_size == conv_layer.kernel_size
        assert bayes_layer.strides == conv_layer.strides
        assert bayes_layer.padding == conv_layer.padding
        assert bayes_layer.input_dilation == conv_layer.input_dilation
        assert bayes_layer.kernel_dilation == conv_layer.kernel_dilation
        assert bayes_layer.feature_group_count == conv_layer.feature_group_count
        assert bayes_layer.use_bias == conv_layer.use_bias
        assert bayes_layer.mask == conv_layer.mask
        assert bayes_layer.dtype == conv_layer.dtype
        assert bayes_layer.param_dtype == conv_layer.param_dtype
        assert bayes_layer.precision == conv_layer.precision
        assert bayes_layer.conv_general_dilated is conv_layer.conv_general_dilated
        assert bayes_layer.promote_dtype is conv_layer.promote_dtype
        assert bayes_layer.preferred_element_type == conv_layer.preferred_element_type

        assert bayes_layer.bias is not None

        assert bayes_layer.kernel_shape == conv_layer.kernel_shape
        weight_shape = bayes_layer.kernel_shape

        rho = _inverse_softplus(jnp.array(posterior_std))
        assert bayes_layer.weight_mu.shape == weight_shape
        assert bayes_layer.weight_rho.shape == weight_shape
        assert jnp.allclose(bayes_layer.weight_rho[...], jnp.full(weight_shape, rho))
        assert jnp.allclose(bayes_layer.weight_prior_mu[...], jnp.full(weight_shape, prior_mean))
        assert jnp.allclose(bayes_layer.weight_prior_sigma[...], jnp.full(weight_shape, prior_std))

        bias_shape = (conv_layer.out_features,)
        assert bayes_layer.bias_mu.shape == bias_shape
        assert bayes_layer.bias_rho.shape == bias_shape
        assert jnp.allclose(bayes_layer.bias_mu[...], jnp.zeros(bias_shape))
        assert jnp.allclose(bayes_layer.bias_rho[...], jnp.full(bias_shape, rho))
        assert jnp.allclose(bayes_layer.bias_prior_mu[...], jnp.full(bias_shape, prior_mean))
        assert jnp.allclose(bayes_layer.bias_prior_sigma[...], jnp.full(bias_shape, prior_std))

        assert isinstance(bayes_layer.bias, nnx.Param)

    def test_bayesian_conv_attributes_use_base_weights(self) -> None:
        """Tests BayesianConv layer attributes for use_base_weights True."""
        rngs = nnx.Rngs(0, params=1)
        conv_layer = nnx.Conv(3, 4, (3,), padding="VALID", rngs=rngs)

        bayes_layer = BayesConv(conv_layer, use_base_weights=True, rngs=rngs)

        # Posterior mean dna prior mean are copied from base layer's attributes
        assert jnp.array_equal(bayes_layer.weight_mu[...], conv_layer.kernel[...])
        assert jnp.array_equal(bayes_layer.weight_prior_mu[...], conv_layer.kernel[...])

        assert conv_layer.bias is not None
        assert jnp.array_equal(bayes_layer.bias_mu[...], conv_layer.bias[...])
        assert jnp.array_equal(bayes_layer.bias_prior_mu[...], conv_layer.bias[...])

    def test_bayesian_conv_attributes_no_bias(self) -> None:
        rngs = nnx.Rngs(0, params=1)
        conv_layer = nnx.Conv(3, 4, (3,), use_bias=False, padding="VALID", rngs=rngs)

        bayes_layer = BayesConv(conv_layer, use_base_weights=False, rngs=rngs)

        # Takes use_bias as bool flag instead of bias
        assert bayes_layer.use_bias is False
        assert bayes_layer.bias is None
        assert bayes_layer.bias_mu is None
        assert bayes_layer.bias_rho is None
        assert bayes_layer.bias_prior_mu is None
        assert bayes_layer.bias_prior_sigma is None

    def test_copy_conv_attrs_direct_call(self) -> None:
        """Exercises `_copy_conv_attrs` directly, independent of `BayesConv.__init__`."""
        rngs = nnx.Rngs(0, params=1)
        conv_layer = nnx.Conv(3, 4, (3,), padding="VALID", rngs=rngs)

        # `_copy_conv_attrs` only assigns attributes (no reliance on prior state), so an
        # already-constructed BayesConv can be safely used as the target `module` too.
        bayes_layer = BayesConv(conv_layer, use_base_weights=True, rngs=rngs)
        _copy_conv_attrs(bayes_layer, conv_layer)

        assert bayes_layer.kernel_shape == conv_layer.kernel_shape
        assert bayes_layer.strides == conv_layer.strides
        assert bayes_layer.padding == conv_layer.padding
        assert bayes_layer.feature_group_count == conv_layer.feature_group_count
        assert bayes_layer.conv_general_dilated is conv_layer.conv_general_dilated


class TestRngs:
    """Test class for rng cases."""

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
    """Test class for helper functions."""

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
        mu1 = jnp.array([0.0])
        var1 = jnp.array([1.0])
        mu2 = jnp.array([2.0])
        var2 = jnp.array([1.0])

        kl = _kl_divergence_gaussian(mu1, var1, mu2, var2)

        assert jnp.allclose(kl, jnp.array([2.0]))


class TestNetworkArchitecture:
    """Test class for network architecture."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "flax_model_small_2d_2d",
            "flax_conv_linear_model",
        ],
    )
    def test_fixtures(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        """Tests if a model replaces the linear/conv layers respectively correctly with BayesLinear or BayesConv layers.

        Parameters:
            request: pytest.FixtureRequest, the request for a fixture.
            model_fixture: str, the name of the model fixture.

        Raises:
            AssertionError If the structure of the model differs in an unexpected manner or if the layers are not
            replaced correctly.
        """
        model_base = request.getfixturevalue(model_fixture)
        model = cast("nnx.Module", model_base)

        modified_model = cast(
            "nnx.Module",
            bayesian(
                model,
                use_base_weights=use_base_weights,
                posterior_std=posterior_std,
                prior_mean=prior_mean,
                prior_std=prior_std,
            ),
        )
        # Linear Model
        count_linear_original = count_layers(model, nnx.Linear)
        count_linear_modified = count_layers(modified_model, nnx.Linear)
        count_bayes_linear_modified = count_layers(modified_model, BayesLinear)

        # Conv Linear Model
        count_conv_original = count_layers(model, nnx.Conv)
        count_conv_modified = count_layers(modified_model, nnx.Conv)
        count_bayes_conv_modified = count_layers(modified_model, BayesConv)

        assert isinstance(modified_model, type(model))
        assert count_linear_modified == 0
        assert count_bayes_linear_modified == count_linear_original

        # BayesConv inherits of nnx.Conv, the number of layers stays the same
        assert count_conv_modified == count_conv_original
        assert count_bayes_conv_modified == count_conv_original


class TestCall:
    """Test class for Bayesian calls."""

    def test_call_linear(self) -> None:
        """Tests the call function with rngs at initialization."""
        linear = nnx.Linear(1, 4, rngs=nnx.Rngs(0))
        bayes_linear = BayesLinear(linear, rngs=nnx.Rngs(0))
        x = jnp.ones(1)
        y = bayes_linear(x)
        assert y is not None
        assert y.shape == (4,)

    def test_calls_linear_with_rngs(self) -> None:
        """Tests calls with rngs at call time."""
        linear = nnx.Linear(1, 4, rngs=nnx.Rngs(0))
        bayes_linear = BayesLinear(linear, rngs=nnx.Rngs(0))

        x = jnp.ones(1)
        y1 = bayes_linear(x, rngs=nnx.Rngs(0))
        y2 = bayes_linear(x, rngs=nnx.Rngs(1))

        assert y1.shape == y2.shape
        assert not jnp.equal(y1, y2).all()

        y_rngs_jax_array = bayes_linear(x, rngs=jax.random.key(1))
        assert y_rngs_jax_array is not None
        assert y_rngs_jax_array.shape == (4,)

        msg = f"rngs must be Rngs, RngStream or jax.Array, but got {str}"
        with pytest.raises(TypeError, match=msg):
            bayes_linear(x, rngs="test")

    def test_kl_divergence_linear(self) -> None:
        linear = nnx.Linear(1, 4, rngs=nnx.Rngs(0))
        bayes_linear = BayesLinear(linear, rngs=nnx.Rngs(0))

        kl = bayes_linear.kl_divergence

        assert kl is not None
        assert kl.shape == ()
        assert kl >= 0

    def test_call_conv(self, flax_rngs: nnx.Rngs) -> None:
        """Tests the call function with rngs at initialization."""
        conv = nnx.Conv(3, 4, (3,), rngs=flax_rngs)
        bayes_conv = BayesConv(conv, rngs=nnx.Rngs(1))

        x = jnp.ones((1, 8, 3))
        y = bayes_conv(x, rngs=nnx.Rngs(2))

        assert y is not None
        assert y.shape == (1, 8, 4)

    def test_calls_conv_with_rngs(self) -> None:
        """Tests calls with rngs at call time."""
        conv = nnx.Conv(3, 4, (3,), rngs=nnx.Rngs(0))
        bayes_conv = BayesConv(conv, rngs=nnx.Rngs(0))

        x = jnp.ones((1, 8, 3))
        y1 = bayes_conv(x, rngs=nnx.Rngs(0))
        y2 = bayes_conv(x, rngs=nnx.Rngs(1))

        assert y1.shape == y2.shape
        assert not jnp.equal(y1, y2).all()

        y_rngs_jax_array = bayes_conv(x, rngs=jax.random.key(1))
        assert y_rngs_jax_array is not None
        assert y_rngs_jax_array.shape == (1, 8, 4)

        msg = f"rngs must be Rngs, RngStream or jax.Array, but got {str}"
        with pytest.raises(TypeError, match=msg):
            bayes_conv(x, rngs="test")

    def test_kl_divergence_conv(self) -> None:
        conv = nnx.Conv(3, 4, (3,), rngs=nnx.Rngs(0))
        bayes_conv = BayesConv(conv, rngs=nnx.Rngs(0))

        kl = bayes_conv.kl_divergence

        assert kl is not None
        assert kl.shape == ()
        assert kl >= 0
