"""Test for flax dropconnect models."""

from __future__ import annotations

import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly.layers.flax import DropConnectLinear  # noqa: E402
from probly.transformation import dropconnect  # noqa: E402
from tests.probly.flax_utils import count_layers  # noqa: E402


class TestDropConnectAttributes:
    """Test class for DropConnectLinear attributes."""

    def test_dropconnect_attributes(self) -> None:
        """Tests the DropConnectLinear layer attributes.

        This function verifies that:
        - A DropConnectLinear layer is correctly constructed from an existing Linear layer.
        - All relevant attributes are inherited (e.g. kernel, bias).
        - DropConnect specific attributes are correctly set (rate, deterministic, rng_collection, rngs).
        """
        rngs = nnx.Rngs(0, params=1)
        linear_layer = nnx.Linear(2, 2, rngs=rngs)
        p = 0.2
        dropconnect_layer = DropConnectLinear(linear_layer, rate=p, rngs=rngs)

        assert dropconnect_layer.kernel == linear_layer.kernel
        assert dropconnect_layer.kernel.shape == linear_layer.kernel.shape
        assert dropconnect_layer.bias == linear_layer.bias
        assert dropconnect_layer.in_features == linear_layer.in_features
        assert dropconnect_layer.out_features == linear_layer.out_features
        assert dropconnect_layer.use_bias == linear_layer.use_bias
        assert dropconnect_layer.dtype == linear_layer.dtype
        assert dropconnect_layer.param_dtype == linear_layer.param_dtype
        assert dropconnect_layer.precision == linear_layer.precision
        assert dropconnect_layer.kernel_init == linear_layer.kernel_init
        assert dropconnect_layer.bias_init == linear_layer.bias_init
        assert dropconnect_layer.dot_general == linear_layer.dot_general
        assert dropconnect_layer.promote_dtype == linear_layer.promote_dtype
        assert dropconnect_layer.preferred_element_type == linear_layer.preferred_element_type
        assert dropconnect_layer.rate == p
        assert not dropconnect_layer.deterministic
        assert dropconnect_layer.rng_collection == "dropconnect"

        # simulate rngs, default stream used
        new_rngs = nnx.Rngs(0, params=1)
        dropconnect_rngs = new_rngs["dropconnect"].fork()
        assert dropconnect_layer.rngs.key.value == dropconnect_rngs.key.value

    def test_dropconnect_rngs(self) -> None:
        rngs = nnx.Rngs(0, params=1)
        linear_layer = nnx.Linear(1, 2, rngs=rngs)
        rng_stream = nnx.RngStream(key=1, tag="dropconnect")
        p = 0.2
        dropconnect_layer_rng_stream = DropConnectLinear(linear_layer, rate=p, rngs=rng_stream)

        new_rng_stream = nnx.RngStream(key=1, tag="dropconnect")
        used_rngs = new_rng_stream.fork()
        assert dropconnect_layer_rng_stream.rngs.key.value == used_rngs.key.value

        msg = f"rngs must be a RNGS, RngStream or None, but got {str}"
        with pytest.raises(TypeError, match=msg):
            DropConnectLinear(linear_layer, rate=p, rngs="test")

    def test_dropconnect_rng_stream(self) -> None:
        """Tests the DropConnectLinear rngs rng stream.

        This function verifies that:
        - A DropConnectLinear layer uses the 'dropconnect' rng stream if provided.
        """
        rngs = nnx.Rngs(0, params=1, dropconnect=2)
        linear_layer = nnx.Linear(2, 2, rngs=rngs)
        dropconnect_layer = DropConnectLinear(linear_layer, rate=2, rngs=rngs)

        # simulate rngs, dropconnect stream used
        new_rngs = nnx.Rngs(0, params=1, dropconnect=2)
        dropconnect_rngs = new_rngs["dropconnect"].fork()
        assert dropconnect_layer.rngs.tag == "dropconnect"
        assert dropconnect_layer.rngs.key.value == dropconnect_rngs.key.value


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "flax_model_small_2d_2d",
            "flax_conv_linear_model",
            "flax_regression_model_1d",
            "flax_regression_model_2d",
            "flax_dropout_model",
            "flax_dropconnect_model",
        ],
    )
    def test_fixtures(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        """Tests if a model replaces linear layers correctly with DropConnectLinear layers.

        This function verifies that:
        - Linear layers are replaced by DropConnectLinear layers, except for the first layer.
        - The structure of the model remains unchanged except for the replaced layers.
        - Only the specified probability parameter is applied in dropconnect modifications.

        It performs counts and asserts to ensure the modified model adheres to expectations.

        Parameters:
            request: pytest.FixtureRequest, the request for a fixture.
            model_fixture: str, the name of the model fixture.

        Raises:
            AssertionError If the structure of the model differs in an unexpected manner or if the layers are not
            replaced correctly.
        """
        model = request.getfixturevalue(model_fixture)
        p = 0.5

        modified_model = dropconnect(model, p)

        # count number of layer type layers in fixture model
        count_linear_original = count_layers(model, nnx.Linear)
        count_conv_original = count_layers(model, nnx.Conv)
        if isinstance(model.layers[0], nnx.Linear):
            count_linear_original_modifiable = count_layers(model, nnx.Linear) - 1
        else:
            count_linear_original_modifiable = count_layers(model, nnx.Linear)
        count_dropconnect_original = count_layers(model, DropConnectLinear)
        count_sequential_original = count_layers(model, nnx.Sequential)

        # count number of layer type layers in modified model
        count_linear_modified = count_layers(modified_model, nnx.Linear)
        count_conv_modified = count_layers(modified_model, nnx.Conv)
        count_dropconnect_modified = count_layers(modified_model, DropConnectLinear)
        count_sequential_modified = count_layers(modified_model, nnx.Sequential)

        # check that the model is modified as expected
        assert modified_model is not None
        assert isinstance(modified_model, type(model))
        assert count_conv_modified == count_conv_original
        assert count_dropconnect_modified == count_linear_original_modifiable + count_dropconnect_original
        assert count_linear_modified == count_linear_original - count_linear_original_modifiable
        assert count_linear_original + count_dropconnect_original == count_linear_modified + count_dropconnect_modified
        assert count_sequential_original == count_sequential_modified

        # check p value in dropconnect layer
        if model_fixture != "flax_dropconnect_model":
            for m in model.iter_modules():
                if isinstance(m, DropConnectLinear):
                    assert m.rate == p

    def test_custom_model(self, flax_custom_model: nnx.Module) -> None:
        """Tests the custom TinyModel modification with DropConnectLinear layers."""
        p = 0.5
        model = dropconnect(flax_custom_model, p)

        count_linear_original = count_layers(flax_custom_model, nnx.Linear)

        count_linear_modified = count_layers(model, nnx.Linear)
        count_dropconnect_modified = count_layers(model, DropConnectLinear)

        # check if model type is correct
        assert isinstance(model, type(flax_custom_model))
        assert not isinstance(model, nnx.Sequential)
        assert count_dropconnect_modified == count_linear_original - 1
        assert count_linear_modified == count_linear_original - 1

        # check p value in dropconnect layer
        for m in flax_custom_model.iter_modules():
            if isinstance(m, DropConnectLinear):
                assert m.rate == p


class TestCall:
    """Test class for DropConnectLinear calls."""

    def test_call(self) -> None:
        """Tests the call function with rngs at initialization."""
        dropconnect_layer = DropConnectLinear(nnx.Linear(1, 4, rngs=nnx.Rngs(0)), rate=0.25, rngs=nnx.Rngs(0))
        x = jnp.ones(1)
        y = dropconnect_layer(x)
        assert y is not None
        assert y.shape == (4,)

    def test_calls_with_rngs(self) -> None:
        """Tests calls with rngs at call time."""
        dropconnect_layer = DropConnectLinear(nnx.Linear(1, 4, rngs=nnx.Rngs(0)), rate=0.25)
        x = jnp.ones(1)
        y1 = dropconnect_layer(x, rngs=nnx.Rngs(0))
        y2 = dropconnect_layer(x, rngs=nnx.Rngs(1))

        assert y1.shape == y2.shape
        assert not jnp.equal(y1, y2).all()

        y_rngs_jax_array = dropconnect_layer(x, rngs=jax.random.key(1))
        assert y_rngs_jax_array is not None
        assert y_rngs_jax_array.shape == (4,)

        msg = f"rngs must be Rngs, RngStream or jax.Array, but got {str}"
        with pytest.raises(TypeError, match=msg):
            dropconnect_layer(x, rngs="test")

    def test_call_without_rngs(self) -> None:
        """Tests calls without call and init rngs."""
        dropconnect_layer = DropConnectLinear(nnx.Linear(1, 4, rngs=nnx.Rngs(0)), rate=0.25)
        x = jnp.ones(1)

        msg = """No `rngs` argument was provided to DropConnect
                as either a __call__ argument or class attribute."""
        with pytest.raises(ValueError, match=msg):
            dropconnect_layer(x)

    def test_determinstic_call(self) -> None:
        rngs = nnx.Rngs(0, params=1)
        dropconnect_layer = DropConnectLinear(nnx.Linear(1, 2, rngs=rngs), rate=0.25, rngs=rngs)
        dropconnect_layer.set_mode(deterministic=True)
        assert dropconnect_layer.deterministic is True
        x = jnp.ones(1)
        y = dropconnect_layer(x)
        assert y is not None
        assert y.shape == (2,)
