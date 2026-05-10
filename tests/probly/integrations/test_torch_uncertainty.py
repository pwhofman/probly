"""Tests for optional torch-uncertainty bindings."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_uncertainty")
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch_uncertainty.models.wrappers as tu_wrappers
from torch_uncertainty.post_processing.calibration import TemperatureScaler

from probly.calibrator import calibrate
from probly.predictor import predict
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistributionSample,
    TorchLogitCategoricalDistribution,
)
from probly.representation.sample.torch import TorchSample
from probly.representer import representer


def _linear_model() -> nn.Module:
    model = nn.Linear(2, 3)
    with torch.no_grad():
        model.weight.fill_(0.25)
        model.bias.zero_()
    return model


def test_torch_uncertainty_deep_ensemble_predict_returns_sample() -> None:
    model = tu_wrappers.deep_ensembles([_linear_model(), _linear_model()])
    x = torch.ones(4, 2)

    sample = predict(model, x)

    assert isinstance(sample, TorchSample)
    assert sample.sample_dim == 0
    assert sample.tensor.shape == (2, 4, 3)


def test_torch_uncertainty_deep_ensemble_logit_representer() -> None:
    model = tu_wrappers.deep_ensembles([_linear_model(), _linear_model()])
    x = torch.ones(4, 2)

    representation = representer(model, kind="logits")(x)

    assert isinstance(representation, TorchCategoricalDistributionSample)
    assert isinstance(representation.tensor, TorchLogitCategoricalDistribution)
    assert representation.sample_dim == 0
    assert representation.tensor.logits.shape == (2, 4, 3)


def test_torch_uncertainty_deep_ensemble_value_representer() -> None:
    model = tu_wrappers.deep_ensembles([_linear_model(), _linear_model()])
    x = torch.ones(4, 2)

    representation = representer(model)(x)

    assert isinstance(representation, TorchSample)
    assert representation.tensor.shape == (2, 4, 3)


def test_torch_uncertainty_temperature_scaler_calibrate_returns_same_object() -> None:
    scaler = TemperatureScaler(model=nn.Identity())
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1, 0, 1])
    dataloader = DataLoader(TensorDataset(logits, labels), batch_size=2)

    calibrated = calibrate(scaler, dataloader, progress=False)

    assert calibrated is scaler
    assert scaler.trained


def test_torch_uncertainty_calibration_logit_representer_returns_logit_distribution() -> None:
    """Calibration scaler representer wraps predicted logits in a logit distribution."""
    scaler = TemperatureScaler(model=nn.Identity())
    x = torch.tensor([[1.0, 0.0, 0.5]])
    representation = representer(scaler)(x)
    assert isinstance(representation, TorchLogitCategoricalDistribution)
    # The temperature scaler returns logits with the same trailing class dimension as input.
    assert representation.logits.shape[-1] == 3


def test_torch_uncertainty_mc_dropout_predict_uses_num_estimators_branch() -> None:
    """MC Dropout wrappers expose ``num_estimators`` and reshape predictions accordingly."""

    class DropoutModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2, 3)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.dropout(x))

    # Use ``on_batch=False`` so the wrapper concatenates over the batch axis,
    # producing the (num_estimators * batch_size, ...) layout probly expects.
    model = tu_wrappers.mc_dropout(DropoutModel(), num_estimators=2, on_batch=False)
    model.eval()
    x = torch.ones(4, 2)
    sample = predict(model, x)
    assert isinstance(sample, TorchSample)
    assert sample.tensor.shape == (2, 4, 3)
    assert sample.sample_dim == 0


def test_torch_uncertainty_stochastic_model_predict_uses_num_samples_branch() -> None:
    """Stochastic-model wrappers expose ``num_samples`` and route through the second branch."""

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # tile the batch ``num_samples`` times to mimic the StochasticModel forward contract.
            return self.fc(x)

    base_model = TinyModel()
    wrapper = tu_wrappers.StochasticModel(base_model, num_samples=3)
    # The probly _num_estimators helper should pick up ``num_samples``.
    from probly.integrations.torch_uncertainty import _num_estimators  # noqa: PLC0415

    assert _num_estimators(wrapper) == 3


def test_torch_uncertainty_num_estimators_missing_raises() -> None:
    """Wrappers without ``num_estimators``/``num_samples`` produce a clear error."""

    class Bare:
        pass

    from probly.integrations.torch_uncertainty import _num_estimators  # noqa: PLC0415

    with pytest.raises(ValueError, match="Cannot infer number of estimators"):
        _num_estimators(Bare())


def test_torch_uncertainty_reshape_tensor_sample_rejects_size_mismatch() -> None:
    """The flat-output reshape rejects tensors whose first dim mismatches expectations."""
    from probly.integrations.torch_uncertainty import _reshape_tensor_sample  # noqa: PLC0415

    with pytest.raises(ValueError, match="num_estimators \\* batch_size"):
        _reshape_tensor_sample(torch.zeros(7, 3), num_estimators=2, batch_size=4)


def test_torch_uncertainty_reshape_batched_output_rejects_unsupported_type() -> None:
    """``_reshape_batched_output`` rejects non-tensor / non-mapping outputs."""
    from probly.integrations.torch_uncertainty import _reshape_batched_output  # noqa: PLC0415

    with pytest.raises(TypeError, match="Unsupported torch-uncertainty output"):
        _reshape_batched_output([1, 2, 3], num_estimators=1, batch_size=3)


def test_torch_uncertainty_first_tensor_batch_size_rejects_scalar() -> None:
    """The batch-size helper rejects scalar tensors."""
    from probly.integrations.torch_uncertainty import _first_tensor_batch_size  # noqa: PLC0415

    with pytest.raises(ValueError, match="scalar"):
        _first_tensor_batch_size((torch.tensor(0.0),), {})


def test_torch_uncertainty_first_tensor_batch_size_rejects_no_tensors() -> None:
    """The batch-size helper requires at least one tensor argument."""
    from probly.integrations.torch_uncertainty import _first_tensor_batch_size  # noqa: PLC0415

    with pytest.raises(ValueError, match=r"at least one torch\.Tensor"):
        _first_tensor_batch_size((1, 2), {"name": "value"})


def test_torch_uncertainty_first_tensor_batch_size_uses_kwargs() -> None:
    """Tensors passed via kwargs satisfy the batch-size helper."""
    from probly.integrations.torch_uncertainty import _first_tensor_batch_size  # noqa: PLC0415

    out = _first_tensor_batch_size((), {"x": torch.zeros(7, 2)})
    assert out == 7


def test_torch_uncertainty_probabilities_representer_wraps_in_probability_distribution() -> None:
    """``kind='probabilities'`` produces a probability-typed categorical distribution sample."""
    from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
        TorchProbabilityCategoricalDistribution,
    )

    model = tu_wrappers.deep_ensembles([_linear_model(), _linear_model()])
    x = torch.ones(4, 2)
    representation = representer(model, kind="probabilities")(x)
    assert isinstance(representation, TorchCategoricalDistributionSample)
    assert isinstance(representation.tensor, TorchProbabilityCategoricalDistribution)


def test_torch_uncertainty_logit_representer_rejects_non_sample_output() -> None:
    """Switching to ``kind='logits'`` requires sample outputs."""
    from probly.integrations.torch_uncertainty import TorchUncertaintySampleRepresenter  # noqa: PLC0415

    class FakePredictor:
        num_estimators = 2

        def __call__(self, _x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"loc": torch.zeros(4, 3), "var": torch.ones(4, 3)}

    rep = TorchUncertaintySampleRepresenter(FakePredictor(), kind="logits")
    # The dispatched ``predict`` will return a Mapping which is not a TorchSample.
    with pytest.raises(TypeError, match="Expected tensor sample output"):
        rep.represent(torch.ones(2, 2))


def _gaussian_regression_ensemble(out_keys: tuple[str, str]) -> nn.Module:
    """Build a 2-model Gaussian regression ensemble that returns a chosen key pair."""
    mean_key, var_key = out_keys

    class GaussianMember(nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            bs = x.shape[0]
            return {mean_key: torch.zeros(bs, 1), var_key: torch.ones(bs, 1) * 0.5}

    return tu_wrappers.deep_ensembles([GaussianMember(), GaussianMember()], task="regression", probabilistic=True)


def test_torch_uncertainty_gaussian_representer_collects_loc_var_branches() -> None:
    """The Gaussian representer composes ``loc``+``var`` mappings into a Gaussian sample."""
    from probly.representation.distribution.torch_gaussian import TorchGaussianDistributionSample  # noqa: PLC0415

    ensemble = _gaussian_regression_ensemble(("loc", "var"))
    ensemble.eval()
    rep = representer(ensemble, kind="gaussian")
    out = rep(torch.ones(3, 2))
    assert isinstance(out, TorchGaussianDistributionSample)


def test_torch_uncertainty_gaussian_representer_supports_mean_alias() -> None:
    """``mean`` is accepted as an alias for ``loc``."""
    from probly.representation.distribution.torch_gaussian import TorchGaussianDistributionSample  # noqa: PLC0415

    ensemble = _gaussian_regression_ensemble(("mean", "var"))
    ensemble.eval()
    rep = representer(ensemble, kind="gaussian")
    out = rep(torch.ones(3, 2))
    assert isinstance(out, TorchGaussianDistributionSample)


def test_torch_uncertainty_gaussian_representer_supports_variance_alias() -> None:
    """``variance`` is also accepted in place of ``var``."""
    from probly.representation.distribution.torch_gaussian import TorchGaussianDistributionSample  # noqa: PLC0415

    ensemble = _gaussian_regression_ensemble(("mean", "variance"))
    ensemble.eval()
    rep = representer(ensemble, kind="gaussian")
    out = rep(torch.ones(3, 2))
    assert isinstance(out, TorchGaussianDistributionSample)


def test_torch_uncertainty_gaussian_representer_supports_scale_squared() -> None:
    """``scale`` (standard deviation) is squared into ``var`` automatically."""
    from probly.representation.distribution.torch_gaussian import TorchGaussianDistributionSample  # noqa: PLC0415

    ensemble = _gaussian_regression_ensemble(("loc", "scale"))
    ensemble.eval()
    rep = representer(ensemble, kind="gaussian")
    out = rep(torch.ones(3, 2))
    assert isinstance(out, TorchGaussianDistributionSample)


def test_torch_uncertainty_gaussian_representer_rejects_missing_mean() -> None:
    """A Gaussian mapping without ``loc``/``mean`` raises a clear error."""

    class MeanlessMember(nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"var": torch.ones(x.shape[0], 1) * 0.5}

    ensemble = tu_wrappers.deep_ensembles([MeanlessMember(), MeanlessMember()], task="regression", probabilistic=True)
    ensemble.eval()
    rep = representer(ensemble, kind="gaussian")
    with pytest.raises(ValueError, match="'loc' or 'mean'"):
        rep(torch.ones(3, 2))


def test_torch_uncertainty_gaussian_representer_rejects_missing_var() -> None:
    """A Gaussian mapping without a variance/scale tensor raises a clear error."""

    class VarlessMember(nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"mean": torch.zeros(x.shape[0], 1)}

    ensemble = tu_wrappers.deep_ensembles([VarlessMember(), VarlessMember()], task="regression", probabilistic=True)
    ensemble.eval()
    rep = representer(ensemble, kind="gaussian")
    with pytest.raises(ValueError, match="'var', 'variance', or 'scale'"):
        rep(torch.ones(3, 2))


def test_torch_uncertainty_gaussian_representer_rejects_non_mapping_output() -> None:
    """A predictor that returns a tensor cannot be coerced to ``kind='gaussian'``."""
    model = tu_wrappers.deep_ensembles([_linear_model(), _linear_model()])
    rep = representer(model, kind="gaussian")
    x = torch.ones(4, 2)
    with pytest.raises(TypeError, match="mapping output for kind='gaussian'"):
        rep(x)


def test_torch_uncertainty_unknown_representer_kind_raises() -> None:
    """Unknown ``kind`` strings produce a clear error."""
    from probly.integrations.torch_uncertainty import TorchUncertaintySampleRepresenter  # noqa: PLC0415

    model = tu_wrappers.deep_ensembles([_linear_model(), _linear_model()])
    # Construct directly to bypass the typed factory.
    rep = TorchUncertaintySampleRepresenter(model, kind="bogus")
    with pytest.raises(ValueError, match="Unknown torch-uncertainty representation kind"):
        rep.represent(torch.ones(4, 2))


def test_torch_uncertainty_conformal_predict_returns_one_hot_set() -> None:
    """Conformal classifiers route through ``_predict_torch_uncertainty_conformal``."""
    from torch_uncertainty.post_processing.conformal.thr import ConformalClsTHR  # noqa: PLC0415

    from probly.representation.conformal_set.torch import TorchOneHotConformalSet  # noqa: PLC0415

    linear = nn.Linear(2, 3)
    conformal = ConformalClsTHR(alpha=0.5, model=linear)
    x = torch.randn(64, 2)
    y = torch.randint(0, 3, (64,))
    dl = DataLoader(TensorDataset(x, y), batch_size=16)
    conformal.fit(dl)

    # predict() should bind to _predict_torch_uncertainty_conformal which checks > 0.
    result = predict(conformal, x[:8])
    assert isinstance(result, TorchOneHotConformalSet)
    assert result.tensor.dtype == torch.bool


def test_torch_uncertainty_conformal_representer_passes_through_predict() -> None:
    """The conformal representer relays the conformal-set prediction unchanged."""
    from torch_uncertainty.post_processing.conformal.thr import ConformalClsTHR  # noqa: PLC0415

    from probly.representation.conformal_set.torch import TorchOneHotConformalSet  # noqa: PLC0415

    linear = nn.Linear(2, 3)
    conformal = ConformalClsTHR(alpha=0.5, model=linear)
    x = torch.randn(48, 2)
    y = torch.randint(0, 3, (48,))
    dl = DataLoader(TensorDataset(x, y), batch_size=16)
    conformal.fit(dl)

    rep = representer(conformal)
    out = rep(x[:8])
    assert isinstance(out, TorchOneHotConformalSet)
