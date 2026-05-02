"""Tests for torch SNGP quantification dispatch."""

from __future__ import annotations

import pytest

from probly.method.sngp import sngp
from probly.quantification import decompose, measure, quantify
from probly.quantification.decomposition.entropy import SecondOrderEntropyDecomposition
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representer import representer

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


def _sngp_sample() -> TorchCategoricalDistributionSample:
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 3),
    )
    predictor = sngp(model, num_random_features=128)
    # Run a couple of training-mode forward passes so the precision matrix is
    # populated, then switch to eval mode to exercise the actual inference path.
    predictor.train()
    for _ in range(2):
        _ = predictor(torch.ones(2, 2))
    predictor.eval()
    return representer(predictor, num_samples=3).represent(torch.ones(2, 2))


def test_sngp_representer_returns_categorical_sample() -> None:
    sample = _sngp_sample()

    assert isinstance(sample, TorchCategoricalDistributionSample)


def test_decompose_dispatches_sngp_sample_to_entropy_decomposition() -> None:
    sample = _sngp_sample()

    decomposition = decompose(sample)

    assert isinstance(decomposition, SecondOrderEntropyDecomposition)


def test_quantify_dispatches_sngp_sample_to_entropy_decomposition() -> None:
    sample = _sngp_sample()

    quantification = quantify(sample)

    assert isinstance(quantification, SecondOrderEntropyDecomposition)


def test_sngp_decomposition_contains_all_uncertainties() -> None:
    sample = _sngp_sample()

    decomposition = decompose(sample)

    assert hasattr(decomposition, "aleatoric")
    assert hasattr(decomposition, "epistemic")
    assert hasattr(decomposition, "total")

    assert torch.allclose(decomposition.total, decomposition.aleatoric + decomposition.epistemic)


def test_measure_sngp_sample_returns_total_uncertainty() -> None:
    sample = _sngp_sample()

    uncertainty = measure(sample)
    decomposition = decompose(sample)

    assert torch.allclose(uncertainty, decomposition.total)
    assert uncertainty.shape == (2,)


from probly.layers.torch import _SpectralNormParametrization  # noqa: E402


def _register_sn(layer: nn.Module, *, n_power_iterations: int = 1, norm_multiplier: float = 1.0) -> None:
    """Helper: register an `_SpectralNormParametrization` on `layer.weight`."""
    nn.utils.parametrize.register_parametrization(
        layer,
        "weight",
        _SpectralNormParametrization(
            layer.weight,
            n_power_iterations=n_power_iterations,
            norm_multiplier=norm_multiplier,
        ),
    )


def _spectral_param(layer: nn.Module) -> _SpectralNormParametrization:
    """Helper: pull the SpectralNorm parametrization out of a parametrized layer."""
    parametrization = layer.parametrizations.weight[0]  # ty: ignore[unresolved-attribute,not-subscriptable]
    assert isinstance(parametrization, _SpectralNormParametrization)
    return parametrization


def test_spectral_norm_does_not_run_power_iteration_in_eval_mode() -> None:
    layer = nn.Linear(4, 8)
    _register_sn(layer, n_power_iterations=1)
    layer.eval()
    param = _spectral_param(layer)
    u_before = param.u.clone()
    v_before = param.v.clone()

    _ = layer(torch.randn(3, 4))

    assert torch.equal(param.u, u_before)
    assert torch.equal(param.v, v_before)


def test_spectral_norm_runs_power_iteration_in_train_mode() -> None:
    layer = nn.Linear(4, 8)
    _register_sn(layer, n_power_iterations=1)
    layer.train()
    param = _spectral_param(layer)
    u_before = param.u.clone()

    _ = layer(torch.randn(3, 4))

    # u should change after one power iteration step (with overwhelming probability).
    assert not torch.equal(param.u, u_before)


def test_spectral_norm_warmup_initial_u_v_are_dominant_singular_vectors() -> None:
    """After 15-iter warmup, sigma_estimate should be close to the true top singular value."""
    torch.manual_seed(0)
    weight = torch.randn(8, 16)
    layer = nn.Linear(16, 8, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    # norm_multiplier set high so the factor is 1 and we just inspect sigma.
    _register_sn(layer, n_power_iterations=1, norm_multiplier=1e9)

    param = _spectral_param(layer)
    sigma_estimate = torch.dot(param.u, torch.mv(weight, param.v)).item()
    sigma_true = torch.linalg.svdvals(weight)[0].item()

    # Warmup of 15 iterations should give a very tight estimate (relative tolerance 1%).
    assert abs(sigma_estimate - sigma_true) / sigma_true < 0.01


from probly.layers.torch import SNGPLayer  # noqa: E402


def test_sngp_layer_constructor_uses_imagenet_defaults_for_optional_args() -> None:
    # `num_random_features` is required (no default) - the sngp(...) factory
    # passes the replaced Linear's `in_features`. ridge_penalty and momentum
    # carry the imagenet defaults.
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8)
    assert layer.num_random_features == 8
    assert layer.ridge_penalty == 1.0
    assert layer.momentum == -1.0


def test_sngp_layer_constructor_requires_num_random_features() -> None:
    with pytest.raises(TypeError, match="num_random_features"):
        SNGPLayer(in_features=4, num_classes=3)  # ty: ignore[missing-argument]


def test_sngp_layer_initializes_precision_to_zeros() -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8)
    assert torch.equal(layer.precision_matrix, torch.zeros(8, 8))


def test_sngp_layer_initializes_covariance_buffer_and_stale_flag() -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8, ridge_penalty=1.0)
    assert layer.covariance_matrix.shape == (8, 8)
    assert bool(layer.covariance_is_stale) is True


def test_sngp_layer_output_classifier_has_no_bias() -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8)
    assert layer.sngp.bias is None


def test_sngp_layer_w_l_and_b_l_are_frozen() -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8)
    assert layer.W_L.requires_grad is False
    assert layer.b_L.requires_grad is False
    assert layer.W_L.shape == (8, 4)
    assert layer.b_L.shape == (8,)


def test_sngp_layer_train_mode_returns_placeholder_variance() -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8, momentum=-1.0)
    layer.train()
    x = torch.randn(2, 4)

    logits, variance = layer(x)

    assert logits.shape == (2, 3)
    assert variance.shape == (2, 3)
    assert torch.all(variance > 0)
    assert torch.all(variance < 1e-6)  # placeholder is 1e-12


def test_sngp_layer_train_mode_accumulates_precision_matrix() -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8, momentum=-1.0)
    layer.train()
    assert torch.equal(layer.precision_matrix, torch.zeros(8, 8))

    _ = layer(torch.randn(2, 4))
    after_first = layer.precision_matrix.clone()
    assert not torch.equal(after_first, torch.zeros(8, 8))

    _ = layer(torch.randn(2, 4))
    after_second = layer.precision_matrix.clone()
    # Second call ADDS another phi^T phi (accumulate, no momentum).
    diff = after_second - after_first
    assert not torch.allclose(diff, torch.zeros_like(diff))


def test_sngp_layer_train_mode_marks_covariance_stale() -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8, momentum=-1.0)
    layer.train()
    layer.covariance_is_stale.fill_(False)

    _ = layer(torch.randn(2, 4))

    assert bool(layer.covariance_is_stale) is True


def test_sngp_layer_eval_mode_returns_meaningful_variance() -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8, ridge_penalty=1.0)
    # Run a few training steps so the precision matrix is populated.
    layer.train()
    for _ in range(3):
        _ = layer(torch.randn(5, 4))

    layer.eval()
    logits, variance = layer(torch.randn(2, 4))

    assert logits.shape == (2, 3)
    assert variance.shape == (2, 3)
    assert torch.all(variance > 1e-9)  # not the 1e-12 placeholder


def test_sngp_layer_eval_mode_clears_stale_flag() -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8)
    layer.eval()
    assert bool(layer.covariance_is_stale) is True

    _ = layer(torch.randn(2, 4))

    assert bool(layer.covariance_is_stale) is False


def test_sngp_layer_eval_mode_lazy_covariance_refresh(monkeypatch) -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8)
    call_counter = {"n": 0}
    real_inv = torch.linalg.inv

    def counting_inv(m: torch.Tensor) -> torch.Tensor:
        call_counter["n"] += 1
        return real_inv(m)

    monkeypatch.setattr(torch.linalg, "inv", counting_inv)

    layer.eval()
    _ = layer(torch.randn(2, 4))
    after_first = call_counter["n"]
    _ = layer(torch.randn(2, 4))
    after_second = call_counter["n"]

    assert after_first == 1
    assert after_second == 1  # no re-inversion when stale flag is False


def test_sngp_layer_eval_variance_is_per_sample_broadcast_across_classes() -> None:
    layer = SNGPLayer(in_features=4, num_classes=5, num_random_features=8)
    layer.eval()
    _, variance = layer(torch.randn(3, 4))

    # variance was computed scalar-per-sample then expanded across the class axis,
    # so all values within a row are equal.
    assert torch.allclose(variance, variance[:, :1].expand_as(variance))


def test_sngp_layer_ema_normalizes_by_batch_size() -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8, momentum=0.5)
    layer.train()
    phi_t = torch.randn(6, 8)

    # Manually invoke the update so we can verify the formula.
    layer.update_precision_matrix(phi_t)

    expected = (1.0 - 0.5) * (phi_t.t() @ phi_t) / 6.0  # initial precision is zero
    assert torch.allclose(layer.precision_matrix, expected, atol=1e-5)


def test_sngp_layer_ema_combines_old_and_new_precision() -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8, momentum=0.9)
    layer.train()
    phi_a = torch.randn(4, 8)
    phi_b = torch.randn(4, 8)

    layer.update_precision_matrix(phi_a)
    after_a = layer.precision_matrix.clone()
    layer.update_precision_matrix(phi_b)

    expected = 0.9 * after_a + 0.1 * (phi_b.t() @ phi_b) / 4.0
    assert torch.allclose(layer.precision_matrix, expected, atol=1e-5)


def test_sngp_layer_reset_precision_matrix_zeros_state() -> None:
    layer = SNGPLayer(in_features=4, num_classes=3, num_random_features=8)
    layer.train()
    _ = layer(torch.randn(5, 4))
    assert not torch.equal(layer.precision_matrix, torch.zeros(8, 8))

    layer.reset_precision_matrix()

    assert torch.equal(layer.precision_matrix, torch.zeros(8, 8))
    assert bool(layer.covariance_is_stale) is True


from probly.method.sngp import reset_precision_matrix as reset_helper  # noqa: E402


def test_reset_precision_matrix_helper_walks_predictor() -> None:
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3))
    predictor = sngp(model, num_random_features=128)
    predictor.train()
    _ = predictor(torch.randn(5, 2))

    sngp_layer = next(m for m in predictor.modules() if isinstance(m, SNGPLayer))
    assert not torch.equal(sngp_layer.precision_matrix, torch.zeros(128, 128))

    reset_helper(predictor)

    assert torch.equal(sngp_layer.precision_matrix, torch.zeros(128, 128))
    assert bool(sngp_layer.covariance_is_stale) is True


def test_reset_precision_matrix_helper_no_op_for_predictor_without_sngp(capsys) -> None:
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3))

    reset_helper(model)  # not an SNGP-wrapped predictor

    assert "no SNGPLayer instances found" in capsys.readouterr().out


import inspect  # noqa: E402


def test_sngp_factory_defaults_match_imagenet_recipe() -> None:
    sig = inspect.signature(sngp)
    defaults = {name: p.default for name, p in sig.parameters.items() if p.default is not inspect.Parameter.empty}

    assert defaults["num_random_features"] == 1024
    assert defaults["norm_multiplier"] == 6.0
    assert defaults["ridge_penalty"] == 1.0
    assert defaults["momentum"] == -1.0
    assert defaults["n_power_iterations"] == 1
    assert defaults["eps"] == 1e-12
    assert defaults["name"] == "weight"


import warnings  # noqa: E402


def test_sngp_warns_on_skipped_param_bearing_layer_types() -> None:
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.LSTM(8, 8, batch_first=True),  # not handled by spectral-norm traverser
        nn.Linear(8, 3),
    )

    with pytest.warns(UserWarning, match="LSTM"):
        sngp(model)


def test_sngp_silent_for_pure_linear_conv2d_norm_models() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 4),
        nn.LayerNorm(4),
        nn.Linear(4, 3),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn any warning into an exception
        sngp(model)
