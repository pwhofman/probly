from __future__ import annotations

from collections.abc import Callable
import math

import pytest

from probly.traverse_nn import nn_compose, reset_traverser
from pytraverse import CLONE, traverse

torch = pytest.importorskip("torch")

from scipy import stats  # noqa: E402
from torch import nn  # noqa: E402

from probly.layers.torch import BatchEnsembleConv2d, BatchEnsembleLinear, DropConnectLinear  # noqa: E402


def reset[T: nn.Module](model: T) -> T:
    """Clone ``model`` and reset the parameters of the clone."""
    return traverse(model, nn_compose(reset_traverser), init={CLONE: True})


def assert_same_parameters(actual: nn.Module, expected: nn.Module) -> None:
    """Assert that two modules hold bit-identical parameters under the same names."""
    actual_params = dict(actual.named_parameters())
    expected_params = dict(expected.named_parameters())
    assert actual_params.keys() == expected_params.keys()
    for name, value in expected_params.items():
        torch.testing.assert_close(actual_params[name], value, rtol=0, atol=0, msg=f"parameter {name} differs")


def fill_parameters(model: nn.Module, value: float) -> None:
    """Overwrite every parameter, so that any parameter a reset misses keeps an unmistakable value."""
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(value)


class TokenModel(nn.Module):
    """A vision-transformer-like head whose tokens are parameters of the module itself."""

    def __init__(self, dim: int = 64, num_patches: int = 32) -> None:
        """Create the tokens with the scales vision transformers commonly use."""
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(0.02 * torch.randn(1, num_patches + 1, dim))
        self.fc = nn.Linear(dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        return self.fc(tokens[:, 0])


# Factories for torch modules whose parameters are initialized by the module itself. A reset under a
# given seed must draw exactly what the constructor draws under that seed.
SELF_INITIALIZING_MODULES: dict[str, Callable[..., nn.Module]] = {
    "linear": lambda **kw: nn.Linear(4, 3, **kw),
    "multihead_attention": lambda **kw: nn.MultiheadAttention(8, 2, **kw),
    "multihead_attention_kdim_vdim": lambda **kw: nn.MultiheadAttention(8, 2, kdim=6, vdim=5, **kw),
    "multihead_attention_bias_kv": lambda **kw: nn.MultiheadAttention(8, 2, add_bias_kv=True, **kw),
    "multihead_attention_no_bias": lambda **kw: nn.MultiheadAttention(8, 2, bias=False, **kw),
    "transformer_encoder_layer": lambda **kw: nn.TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, batch_first=True, **kw
    ),
    "transformer_decoder_layer": lambda **kw: nn.TransformerDecoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, batch_first=True, **kw
    ),
    "transformer": lambda **kw: nn.Transformer(
        d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16, batch_first=True, **kw
    ),
}


class TestModuleInitializationScheme:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
    @pytest.mark.parametrize("name", list(SELF_INITIALIZING_MODULES))
    def test_reset_reproduces_the_constructor(self, name: str, dtype: torch.dtype) -> None:
        """The reference is torch's own constructor: under one seed, a reset must draw the same values.

        Children are reset before their parent, as they are constructed before the parent initializes
        itself. ``nn.MultiheadAttention`` and ``nn.Transformer`` initialize through a private
        ``_reset_parameters``; the transformer re-draws every matrix of its layers with Xavier.
        """
        factory = SELF_INITIALIZING_MODULES[name]
        torch.manual_seed(0)
        model = factory(dtype=dtype)
        torch.manual_seed(1)
        expected = factory(dtype=dtype)

        torch.manual_seed(1)
        actual = reset(model)

        assert_same_parameters(actual, expected)
        assert all(param.dtype == dtype for param in actual.parameters())

    def test_attention_follows_the_xavier_scheme(self) -> None:
        """Checks the documented torch scheme with hand-derived bounds, independent of the RNG order."""
        torch.manual_seed(0)
        embed_dim = 32
        model = nn.MultiheadAttention(embed_dim, 4)
        fill_parameters(model, 7.0)

        new_model = reset(model)

        # xavier_uniform_ on the (3E, E) in-projection: bound sqrt(6 / (fan_in + fan_out)) = sqrt(6 / 4E).
        bound = math.sqrt(6.0 / (4 * embed_dim))
        in_proj = new_model.in_proj_weight.detach()
        assert in_proj.abs().max() <= bound
        assert in_proj.std().item() == pytest.approx(bound / math.sqrt(3.0), rel=0.1)
        # The projection biases start at exactly zero.
        assert torch.equal(new_model.in_proj_bias, torch.zeros(3 * embed_dim))
        assert torch.equal(new_model.out_proj.bias, torch.zeros(embed_dim))
        # The output projection keeps the nn.Linear default, kaiming_uniform_(a=sqrt(5)): bound 1 / sqrt(E).
        out_proj = new_model.out_proj.weight.detach()
        assert out_proj.abs().max() <= 1.0 / math.sqrt(embed_dim)
        assert out_proj.std().item() == pytest.approx(1.0 / math.sqrt(3.0 * embed_dim), rel=0.15)

    def test_private_reset_method_of_a_custom_module_is_used(self) -> None:
        class CustomAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty(4, 4))
                self._reset_parameters()

            def _reset_parameters(self) -> None:
                nn.init.constant_(self.weight, 0.5)

        model = CustomAttention()
        fill_parameters(model, 7.0)

        assert torch.equal(reset(model).weight, torch.full((4, 4), 0.5))

    def test_public_reset_method_is_preferred_and_trusted(self) -> None:
        """A module's own ``reset_parameters`` decides what is reset; no fallback redraws the rest."""

        class ScaledLinear(nn.Linear):
            def __init__(self) -> None:
                super().__init__(4, 4)
                self.scale = nn.Parameter(torch.randn(4))
                self.private_calls = 0

            def _reset_parameters(self) -> None:
                self.private_calls += 1

        model = ScaledLinear()
        scale = model.scale.detach().clone()
        with torch.no_grad():
            model.weight.fill_(7.0)

        new_model = reset(model)

        assert (new_model.weight != 7.0).all()
        assert torch.equal(new_model.scale, scale)
        assert new_model.private_calls == 0


class TestOwnParameterFallback:
    """Parameters of modules without any reset method: redrawn with their current mean and std."""

    def test_tokens_are_redrawn(self) -> None:
        torch.manual_seed(0)
        model = TokenModel()

        new_model = reset(model)

        assert not torch.equal(new_model.cls_token, model.cls_token)
        assert not torch.equal(new_model.pos_embed, model.pos_embed)

    def test_redrawn_values_keep_the_scale_of_the_original(self) -> None:
        torch.manual_seed(0)
        model = TokenModel(dim=64, num_patches=63)
        original = model.pos_embed.detach().clone()

        values = reset(model).pos_embed.detach().flatten()

        assert not torch.equal(values, original.flatten())
        # 4096 values drawn as 0.02 * N(0, 1): the redraw has the same scale, not a default one.
        assert values.std().item() == pytest.approx(0.02, rel=0.1)
        assert abs(values.mean().item()) < 0.002
        assert (values != 0).all()
        # The fallback is a normal distribution with the original's mean and standard deviation.
        reference = stats.norm(loc=original.mean().item(), scale=original.std().item())
        assert stats.kstest(values.double().numpy(), reference.cdf).pvalue > 0.01

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
    def test_redrawn_values_keep_dtype_and_mean(self, dtype: torch.dtype) -> None:
        class Offset(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.offset = nn.Parameter(3.0 + 0.5 * torch.randn(2048, dtype=dtype))

        torch.manual_seed(0)
        model = Offset()

        values = reset(model).offset.detach()

        assert values.dtype == dtype
        assert not torch.equal(values, model.offset.detach())
        assert values.mean().item() == pytest.approx(3.0, abs=0.05)
        assert values.std().item() == pytest.approx(0.5, rel=0.1)

    def test_constant_parameters_keep_their_value(self) -> None:
        """A zero-initialized token and a unit gain are their own initialization."""

        class Constants(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.zero_token = nn.Parameter(torch.zeros(1, 1, 8))
                self.gain = nn.Parameter(torch.ones(8))

        new_model = reset(Constants())

        assert torch.equal(new_model.zero_token, torch.zeros(1, 1, 8))
        assert torch.equal(new_model.gain, torch.ones(8))

    def test_single_element_parameters_keep_their_value(self) -> None:
        """One value carries no spread to redraw from, e.g. a learned temperature."""

        class Temperature(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.temperature = nn.Parameter(torch.tensor([1.5]))

        assert torch.equal(reset(Temperature()).temperature, torch.tensor([1.5]))

    def test_frozen_parameters_and_buffers_are_left_alone(self) -> None:
        """Only trainable parameters are redrawn; frozen ones (e.g. random features) and buffers stay."""

        class Frozen(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Parameter(torch.randn(16, 16), requires_grad=False)
                self.index = nn.Parameter(torch.arange(4), requires_grad=False)
                self.register_buffer("running", torch.randn(16))

        model = Frozen()

        new_model = reset(model)

        assert torch.equal(new_model.features, model.features)
        assert torch.equal(new_model.index, model.index)
        assert torch.equal(new_model.running, model.running)

    def test_parametrized_weights_are_redrawn(self) -> None:
        """Spectral norm keeps the raw weight on a parametrization container, which has no reset method.

        ``nn.Linear.reset_parameters`` only writes to the computed weight of a parametrized layer, so the
        raw weight has to be redrawn by the container itself.
        """
        torch.manual_seed(0)
        layer = nn.utils.parametrizations.spectral_norm(nn.Linear(64, 64))
        original = layer.get_parameter("parametrizations.weight.original").detach().clone()

        new_layer = reset(layer)

        redrawn = new_layer.get_parameter("parametrizations.weight.original").detach()
        assert not torch.equal(redrawn, original)
        # The raw weight came from nn.Linear's kaiming_uniform_(a=sqrt(5)): std = 1 / sqrt(3 * fan_in).
        assert redrawn.std().item() == pytest.approx(1.0 / math.sqrt(3.0 * 64), rel=0.1)
        assert torch.isfinite(new_layer(torch.randn(2, 64))).all()

    def test_parameters_of_submodules_are_reset_by_their_own_module(self) -> None:
        """The traversal resets every module once: a fallback on the parent leaves the children alone."""
        torch.manual_seed(0)
        model = TokenModel()
        fill_parameters(model.fc, 7.0)

        new_model = reset(model)

        # The child linear layer was reset with its own nn.Linear scheme, not redrawn around 7.
        bound = 1.0 / math.sqrt(model.fc.in_features)
        assert new_model.fc.weight.abs().max() <= bound
        assert new_model.fc.bias.abs().max() <= bound

    def test_original_model_is_left_untouched(self) -> None:
        torch.manual_seed(0)
        model = TokenModel()
        before = {name: param.detach().clone() for name, param in model.named_parameters()}

        reset(model)

        for name, param in model.named_parameters():
            assert torch.equal(param, before[name])

    def test_reset_model_stays_usable(self) -> None:
        torch.manual_seed(0)
        model = reset(TokenModel(dim=8, num_patches=4))

        out = model(torch.randn(2, 4, 8))

        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()


class TestProblyLayers:
    """probly's own BatchEnsemble and DropConnect layers reset with their own construction scheme."""

    def test_dropconnect_linear_resets_within_nn_linear_bounds(self) -> None:
        """The reset must follow nn.Linear's kaiming-uniform scheme, not a Gaussian redraw.

        A Gaussian redraw with a matching standard deviation would almost certainly put at
        least one of the 2048 weight values outside the uniform bound, so this pins the
        distribution family, not just its scale.
        """
        torch.manual_seed(0)
        layer = DropConnectLinear(nn.Linear(64, 32))

        new_layer = reset(layer)

        assert not torch.equal(new_layer.weight, layer.weight)
        assert not torch.equal(new_layer.bias, layer.bias)
        # The base nn.Linear used kaiming_uniform_(a=sqrt(5)): std = 1 / sqrt(3 * fan_in).
        weight_bound = 1.0 / math.sqrt(64)
        assert new_layer.weight.abs().max().item() <= weight_bound
        assert new_layer.weight.std().item() == pytest.approx(1.0 / math.sqrt(3.0 * 64), rel=0.1)
        assert abs(new_layer.weight.mean().item()) < 0.01
        # The bias bound comes from the same fan_in, via nn.Linear's bias scheme.
        bias_bound = 1.0 / math.sqrt(64)
        assert new_layer.bias.abs().max().item() <= bias_bound

    def test_dropconnect_linear_without_bias_stays_biasless_after_reset(self) -> None:
        torch.manual_seed(0)
        layer = DropConnectLinear(nn.Linear(64, 32, bias=False))

        new_layer = reset(layer)

        assert new_layer.bias is None

    @pytest.mark.parametrize(
        "make_layer",
        [
            lambda: BatchEnsembleLinear(nn.Linear(64, 32), num_members=4),
            lambda: BatchEnsembleConv2d(nn.Conv2d(64, 32, 3), num_members=4),
        ],
        ids=["linear", "conv2d"],
    )
    def test_batchensemble_parameters_are_redrawn(self, make_layer: Callable[[], nn.Module]) -> None:
        torch.manual_seed(0)
        layer = make_layer()

        new_layer = reset(layer)

        for name in ("weight", "bias", "r", "s"):
            assert not torch.equal(getattr(new_layer, name), getattr(layer, name)), name
        # The fast weights r and s were drawn from N(1, 0.5**2), the BatchEnsemble default.
        for name in ("r", "s"):
            values = getattr(new_layer, name).detach()
            assert values.mean().item() == pytest.approx(1.0, abs=0.2), name
            assert values.std().item() == pytest.approx(0.5, rel=0.25), name

    @pytest.mark.parametrize(
        "make_layer",
        [
            lambda **kw: BatchEnsembleLinear(nn.Linear(64, 32), num_members=4, **kw),
            lambda **kw: BatchEnsembleConv2d(nn.Conv2d(64, 32, 3), num_members=4, **kw),
        ],
        ids=["linear", "conv2d"],
    )
    def test_batchensemble_random_sign_fast_weights_stay_binary_after_reset(
        self, make_layer: Callable[..., nn.Module]
    ) -> None:
        """random_sign fast weights must stay in {-1, +1}, not become Gaussian, after a reset."""
        torch.manual_seed(0)
        layer = make_layer(init="random_sign")

        new_layer = reset(layer)

        for name in ("r", "s"):
            before = getattr(layer, name).detach()
            after = getattr(new_layer, name).detach()
            assert torch.isin(after, torch.tensor([-1.0, 1.0])).all(), name
            # The reset must have actually redrawn the values, not left the old ones in place.
            assert not torch.equal(after, before), name
            # Independent per-member rows: two members are not forced to draw the same signs.
            assert not torch.equal(after[0], after[1]), name

    @pytest.mark.parametrize(
        "make_layer",
        [
            lambda **kw: BatchEnsembleLinear(nn.Linear(64, 32), num_members=4, **kw),
            lambda **kw: BatchEnsembleConv2d(nn.Conv2d(64, 32, 3), num_members=4, **kw),
        ],
        ids=["linear", "conv2d"],
    )
    def test_batchensemble_normal_fast_weights_use_the_stored_mean_and_std(
        self, make_layer: Callable[..., BatchEnsembleLinear | BatchEnsembleConv2d]
    ) -> None:
        """A non-default mean/std given at construction must still be used after a reset."""
        torch.manual_seed(0)
        layer = make_layer(init="normal", r_mean=3.0, r_std=0.2, s_mean=-1.0, s_std=0.1)
        # Overwrite the already-N(mean, std) values so a no-op reset could not pass by accident.
        fill_parameters(layer, 7.0)

        new_layer = reset(layer)

        assert new_layer.r.mean().item() == pytest.approx(3.0, abs=0.1)
        assert new_layer.r.std().item() == pytest.approx(0.2, rel=0.3)
        assert new_layer.s.mean().item() == pytest.approx(-1.0, abs=0.1)
        assert new_layer.s.std().item() == pytest.approx(0.1, rel=0.3)

    @pytest.mark.parametrize(
        "make_layer",
        [
            lambda: BatchEnsembleLinear(nn.Linear(64, 32), num_members=4),
            lambda: BatchEnsembleConv2d(nn.Conv2d(64, 32, 3), num_members=4),
        ],
        ids=["linear", "conv2d"],
    )
    def test_batchensemble_bias_rows_are_identical_across_members_after_reset(
        self, make_layer: Callable[[], BatchEnsembleLinear | BatchEnsembleConv2d]
    ) -> None:
        """Construction copies one base bias into every member row; a reset must keep that."""
        torch.manual_seed(0)
        layer = make_layer()

        new_layer = reset(layer)

        bias = new_layer.bias.detach()
        for member in range(1, bias.shape[0]):
            assert torch.equal(bias[member], bias[0])
        assert not torch.equal(bias, layer.bias.detach())

    @pytest.mark.parametrize(
        "make_layer",
        [
            lambda: BatchEnsembleLinear(nn.Linear(64, 32, bias=False), num_members=4),
            lambda: BatchEnsembleConv2d(nn.Conv2d(64, 32, 3, bias=False), num_members=4),
        ],
        ids=["linear", "conv2d"],
    )
    def test_batchensemble_without_bias_stays_zero_after_reset(
        self, make_layer: Callable[[], BatchEnsembleLinear | BatchEnsembleConv2d]
    ) -> None:
        torch.manual_seed(0)
        layer = make_layer()
        # A construction-time zero bias would trivially "stay zero" even without a real reset.
        fill_parameters(layer, 7.0)

        new_layer = reset(layer)

        assert torch.equal(new_layer.bias.detach(), torch.zeros_like(new_layer.bias))

    @pytest.mark.parametrize(
        ("make_layer", "fan_in"),
        [
            (lambda: BatchEnsembleLinear(nn.Linear(64, 32), num_members=4, use_base_weights=True), 64),
            (lambda: BatchEnsembleConv2d(nn.Conv2d(64, 32, 3), num_members=4, use_base_weights=True), 64 * 3 * 3),
        ],
        ids=["linear", "conv2d"],
    )
    def test_batchensemble_use_base_weights_still_redraws_weight_after_reset(
        self, make_layer: Callable[[], BatchEnsembleLinear | BatchEnsembleConv2d], fan_in: int
    ) -> None:
        """A reset is what makes ensemble members differ, so it must redraw the weight even here.

        With use_base_weights=True, the weight starts as a copy of the base layer's weight.
        Leaving it alone on reset would make every ensemble member share that one weight.
        """
        torch.manual_seed(0)
        layer = make_layer()

        new_layer = reset(layer)

        assert not torch.equal(new_layer.weight, layer.weight)
        bound = 1.0 / math.sqrt(fan_in)
        assert new_layer.weight.abs().max().item() <= bound


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the device check.")
def test_reset_keeps_parameters_on_their_device() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    attention = nn.MultiheadAttention(8, 2, device=device)
    tokens = TokenModel(dim=8, num_patches=4).to(device)

    new_attention = reset(attention)
    new_tokens = reset(tokens)

    assert all(param.device.type == "cuda" for param in [*new_attention.parameters(), *new_tokens.parameters()])
    assert not torch.equal(new_attention.in_proj_weight, attention.in_proj_weight)
    assert not torch.equal(new_tokens.cls_token, tokens.cls_token)
