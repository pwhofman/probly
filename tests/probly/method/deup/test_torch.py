"""Tests for the torch DEUP implementation."""

from __future__ import annotations

import pytest


def _torch_modules():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    return torch, nn


class TestErrorPredictionHead:
    """The DEUP error head is an MLP from features to a scalar."""

    def test_layer_count_with_one_hidden_layer(self) -> None:
        torch, nn = _torch_modules()  # noqa: RUF059
        from probly.method.deup.torch import ErrorPredictionHead  # noqa: PLC0415

        head = ErrorPredictionHead(input_dim=4, hidden_size=8, n_hidden_layers=1)
        # 1 hidden Linear + 1 ReLU + 1 final Linear = 3 modules in the Sequential.
        assert len(head.net) == 3
        # First and last must be Linear; middle is ReLU.
        assert isinstance(head.net[0], nn.Linear)
        assert isinstance(head.net[1], nn.ReLU)
        assert isinstance(head.net[2], nn.Linear)
        assert head.net[0].in_features == 4
        assert head.net[0].out_features == 8
        assert head.net[2].out_features == 1

    def test_n_hidden_layers_clamped_to_minimum_one(self) -> None:
        from probly.method.deup.torch import ErrorPredictionHead  # noqa: PLC0415

        head = ErrorPredictionHead(input_dim=4, hidden_size=8, n_hidden_layers=0)
        # The constructor max(1, n_hidden_layers) ensures at least 1 hidden layer.
        # 1 hidden Linear + 1 ReLU + 1 final Linear = 3 modules.
        assert len(head.net) == 3

    def test_layer_count_with_three_hidden_layers(self) -> None:
        from probly.method.deup.torch import ErrorPredictionHead  # noqa: PLC0415

        head = ErrorPredictionHead(input_dim=4, hidden_size=8, n_hidden_layers=3)
        # 3 (Linear+ReLU) blocks + 1 final Linear = 7 modules.
        assert len(head.net) == 7

    def test_forward_returns_squeezed_output(self) -> None:
        torch, _ = _torch_modules()
        from probly.method.deup.torch import ErrorPredictionHead  # noqa: PLC0415

        head = ErrorPredictionHead(input_dim=4, hidden_size=8, n_hidden_layers=1)
        x = torch.randn(2, 4)
        out = head(x)
        # The final dimension is squeezed away, so output shape is (batch,).
        assert out.shape == (2,)


class TestStationarizingFeatureProvider:
    """Provider base class behaviour (without running fit())."""

    def test_normalize_is_identity_when_unfitted(self) -> None:
        torch, _ = _torch_modules()
        from probly.method.deup.torch import StationarizingFeatureProvider  # noqa: PLC0415

        provider = StationarizingFeatureProvider()
        x = torch.randn(3, 4)
        # Unfitted scaler -> identity normalization.
        torch.testing.assert_close(provider._normalize(x), x)  # noqa: SLF001

    def test_forward_default_raises_not_implemented(self) -> None:
        torch, _ = _torch_modules()
        from probly.method.deup.torch import StationarizingFeatureProvider  # noqa: PLC0415

        provider = StationarizingFeatureProvider()
        x = torch.randn(3, 4)
        with pytest.raises(NotImplementedError):
            provider.forward(x, x)


class TestLogMCDropoutVariance:
    """LogMCDropoutVariance computes variance of softmax outputs across MC samples."""

    def test_unfitted_forward_raises(self) -> None:
        torch, _ = _torch_modules()
        from probly.method.deup.torch import LogMCDropoutVariance  # noqa: PLC0415

        provider = LogMCDropoutVariance(n_samples=4, dropout_p=0.1)
        features = torch.randn(2, 8)
        logits = torch.randn(2, 3)
        with pytest.raises(RuntimeError, match="must be fitted"):
            provider.forward(features, logits)

    def test_after_setting_head_forward_returns_log_variance(self) -> None:
        torch, nn = _torch_modules()
        from probly.method.deup.torch import LogMCDropoutVariance  # noqa: PLC0415

        provider = LogMCDropoutVariance(n_samples=4, dropout_p=0.1)
        provider._classification_head = nn.Linear(8, 3)  # noqa: SLF001
        features = torch.randn(2, 8)
        logits = torch.randn(2, 3)
        out = provider.forward(features, logits)
        # Output shape should be (batch, 1)
        assert out.shape == (2, 1)
        # Log of variance can be any finite number.
        assert torch.isfinite(out).all()


class TestLogMAFDensity:
    """LogMAFDensity needs nflows; test that the forward path errors when unfitted."""

    def test_unfitted_forward_raises(self) -> None:
        torch, _ = _torch_modules()
        from probly.method.deup.torch import LogMAFDensity  # noqa: PLC0415

        provider = LogMAFDensity(feature_dim=4, n_transforms=1, hidden_features=4, n_blocks_per_transform=1)
        features = torch.randn(2, 4)
        logits = torch.randn(2, 3)
        with pytest.raises(RuntimeError, match="must be fitted"):
            provider.forward(features, logits)


class TestLogDUEVariance:
    """LogDUEVariance needs gpytorch + spectral norm; test unfitted error path."""

    def test_unfitted_forward_raises(self) -> None:
        torch, _ = _torch_modules()
        from probly.method.deup.torch import LogDUEVariance  # noqa: PLC0415

        provider = LogDUEVariance(num_classes=3, feature_dim=4)
        features = torch.randn(2, 4)
        logits = torch.randn(2, 3)
        with pytest.raises(RuntimeError, match="must be fitted"):
            provider.forward(features, logits)


class TestProviderRegistry:
    """The DEUP provider name registry."""

    def test_known_names_resolve(self) -> None:
        from probly.method.deup.torch import _PROVIDER_REGISTRY, _build_provider  # noqa: PLC0415

        for name in _PROVIDER_REGISTRY:
            provider = _build_provider(name, num_classes=3, feature_dim=4)
            assert isinstance(provider, _PROVIDER_REGISTRY[name])

    def test_unknown_name_raises(self) -> None:
        from probly.method.deup.torch import _build_provider  # noqa: PLC0415

        with pytest.raises(ValueError, match="Unknown DEUP"):
            _build_provider("does_not_exist", num_classes=3, feature_dim=4)

    def test_dict_spec_forwards_kwargs(self) -> None:
        from probly.method.deup.torch import LogMCDropoutVariance, _build_provider  # noqa: PLC0415

        provider = _build_provider(
            {"name": "log_mc_dropout_variance", "n_samples": 7, "dropout_p": 0.25},
            num_classes=3,
            feature_dim=4,
        )
        assert isinstance(provider, LogMCDropoutVariance)
        assert provider.n_samples == 7
        assert provider.dropout_p == 0.25


class TestTorchDEUPTraverser:
    """The traverser strips the last Linear and saves it to HEAD_MODULE."""

    def test_replaces_last_linear_with_identity(self) -> None:
        torch, nn = _torch_modules()  # noqa: RUF059
        from probly.method.deup.torch import HEAD_MODULE, torch_deup_traverser  # noqa: PLC0415
        from probly.traverse_nn import nn_compose, nn_traverser  # noqa: PLC0415
        from pytraverse import TRAVERSE_REVERSED, traverse_with_state  # noqa: PLC0415

        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 3))
        encoder, state = traverse_with_state(
            model,
            nn_compose(torch_deup_traverser, nn_traverser=nn_traverser),
            init={TRAVERSE_REVERSED: True, HEAD_MODULE: None},
        )
        # The HEAD_MODULE should be the last Linear.
        head = state[HEAD_MODULE]
        assert isinstance(head, nn.Linear)
        assert head.in_features == 4
        assert head.out_features == 3
        # The last Linear in the encoder should be Identity now.
        # The encoder is still a Sequential.
        modules = list(encoder.modules())
        # Count Linears - originally 2, now 1 (the first one).
        n_linears = sum(1 for m in modules if isinstance(m, nn.Linear))
        assert n_linears == 1


class TestTorchDEUPPredictor:
    """End-to-end torch DEUP predictor (without phase-2 training)."""

    def test_init_with_log_gmm_density(self) -> None:
        torch, nn = _torch_modules()  # noqa: RUF059
        from probly.method.deup.torch import LogGMMDensity, TorchDEUPPredictor  # noqa: PLC0415

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        deup_pred = TorchDEUPPredictor(
            model,
            hidden_size=16,
            n_hidden_layers=1,
            stationarizing_features=["log_gmm_density"],
        )
        # Components installed correctly.
        assert isinstance(deup_pred.classification_head, nn.Linear)
        assert deup_pred.classification_head.out_features == 3
        assert len(deup_pred.providers) == 1
        assert isinstance(deup_pred.providers[0], LogGMMDensity)

    def test_init_with_no_providers_raises(self) -> None:
        _, nn = _torch_modules()
        from probly.method.deup.torch import TorchDEUPPredictor  # noqa: PLC0415

        model = nn.Sequential(nn.Linear(4, 3))
        with pytest.raises(ValueError, match="at least one stationarizing"):
            TorchDEUPPredictor(model, stationarizing_features=None)

    def test_init_with_empty_providers_raises(self) -> None:
        _, nn = _torch_modules()
        from probly.method.deup.torch import TorchDEUPPredictor  # noqa: PLC0415

        model = nn.Sequential(nn.Linear(4, 3))
        with pytest.raises(ValueError, match="at least one stationarizing"):
            TorchDEUPPredictor(model, stationarizing_features=[])

    def test_init_no_linear_raises(self) -> None:
        _, nn = _torch_modules()
        from probly.method.deup.torch import TorchDEUPPredictor  # noqa: PLC0415

        # No Linear -> head detection fails.
        model = nn.Sequential(nn.ReLU(), nn.Tanh())
        with pytest.raises(ValueError, match=r"No nn\.Linear"):
            TorchDEUPPredictor(model, stationarizing_features=["log_gmm_density"])

    def test_forward_returns_logits_and_error_score(self) -> None:
        torch, nn = _torch_modules()
        from probly.method.deup.torch import TorchDEUPPredictor  # noqa: PLC0415

        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 3))
        deup_pred = TorchDEUPPredictor(
            model,
            hidden_size=8,
            n_hidden_layers=1,
            stationarizing_features=["log_gmm_density"],
        )
        x = torch.randn(2, 4)
        logits, error_score = deup_pred(x)
        assert logits.shape == (2, 3)
        # error_score = 10**log10_error has shape (batch,)
        assert error_score.shape == (2,)
        assert torch.all(error_score > 0)

    def test_predict_representation(self) -> None:
        torch, nn = _torch_modules()
        from probly.method.deup.torch import TorchDEUPPredictor, TorchDEUPRepresentation  # noqa: PLC0415

        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 3))
        deup_pred = TorchDEUPPredictor(
            model,
            hidden_size=8,
            n_hidden_layers=1,
            stationarizing_features=["log_gmm_density"],
        )
        x = torch.randn(2, 4)
        rep = deup_pred.predict_representation(x)
        assert isinstance(rep, TorchDEUPRepresentation)
        assert rep.error_score.shape == (2,)


class TestDEUPRepresentation:
    """Decomposition behaviour for a DEUP representation."""

    def test_total_equals_error_score(self) -> None:
        torch, _ = _torch_modules()
        from probly.method.deup._common import DEUPDecomposition  # noqa: PLC0415
        from probly.method.deup.torch import TorchDEUPRepresentation  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        rep = TorchDEUPRepresentation(
            softmax=TorchProbabilityCategoricalDistribution(torch.tensor([[0.5, 0.3, 0.2]])),
            error_score=torch.tensor([0.7]),
        )
        d = DEUPDecomposition(representation=rep)
        # total = error_score  # noqa: ERA001
        torch.testing.assert_close(d._total, torch.tensor([0.7]))  # noqa: SLF001
        # aleatoric = 0  # noqa: ERA001
        torch.testing.assert_close(d._aleatoric, torch.tensor([0.0]))  # noqa: SLF001

    def test_categorical_from_mean_returns_softmax(self) -> None:
        torch, _ = _torch_modules()
        from probly.decider import categorical_from_mean  # noqa: PLC0415

        # Importing the torch DEUP module triggers registration of the DEUPRepresentation handler.
        from probly.method.deup.torch import TorchDEUPRepresentation  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        softmax = TorchProbabilityCategoricalDistribution(torch.tensor([[0.5, 0.3, 0.2]]))
        rep = TorchDEUPRepresentation(softmax=softmax, error_score=torch.tensor([0.5]))
        result = categorical_from_mean(rep)
        # It should be the softmax distribution itself.
        assert result is softmax


class TestCreateDEUPRepresentation:
    """The factory for creating DEUP representations."""

    def test_torch_factory(self) -> None:
        torch, _ = _torch_modules()
        from probly.method.deup import create_deup_representation  # noqa: PLC0415
        from probly.method.deup.torch import TorchDEUPRepresentation  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        softmax = TorchProbabilityCategoricalDistribution(torch.tensor([[0.5, 0.3, 0.2]]))
        rep = create_deup_representation(softmax, torch.tensor([0.5]))
        assert isinstance(rep, TorchDEUPRepresentation)


class TestDEUPHighLevel:
    """Top-level deup() transformation."""

    def test_deup_transformation(self) -> None:
        torch, nn = _torch_modules()
        from probly.method.deup import deup  # noqa: PLC0415
        from probly.method.deup.torch import TorchDEUPPredictor  # noqa: PLC0415
        from probly.predictor import LogitClassifier  # noqa: PLC0415

        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 3))
        deup_pred = deup(
            model,
            hidden_size=16,
            n_hidden_layers=1,
            stationarizing_features=["log_gmm_density"],
            predictor_type=LogitClassifier,
        )
        assert isinstance(deup_pred, TorchDEUPPredictor)
        x = torch.randn(2, 4)
        logits, error_score = deup_pred(x)
        assert logits.shape == (2, 3)
        assert error_score.shape == (2,)


# ---------------------------------------------------------------------------
# Fitting / training-path tests for DEUP providers and predictor.
# ---------------------------------------------------------------------------


def _tiny_classification_loader(num_classes: int = 3, n_samples: int = 12, feature_dim: int = 4):
    """Build a tiny TensorDataset DataLoader for DEUP fit() coverage.

    Uses balanced labels so per-class GMM fitting (which needs >= 2 samples
    per class) succeeds.
    """
    torch, _ = _torch_modules()
    from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415

    inputs = torch.randn(n_samples, feature_dim)
    labels = torch.arange(n_samples) % num_classes
    return DataLoader(TensorDataset(inputs, labels), batch_size=4)


class TestLogGMMDensityFit:
    """LogGMMDensity._fit_internal extracts encoder features and fits the GMM."""

    def test_fit_internal_then_forward(self) -> None:
        torch, nn = _torch_modules()
        from probly.method.deup.torch import LogGMMDensity  # noqa: PLC0415

        encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        classification_head = nn.Linear(8, 3)
        loader = _tiny_classification_loader(num_classes=3, n_samples=12, feature_dim=4)

        provider = LogGMMDensity(num_classes=3, feature_dim=8)
        provider._fit_internal(encoder, classification_head, loader, torch.device("cpu"))  # noqa: SLF001

        # The GMM head should have non-default means after fitting (they were
        # zero-initialised in __init__).
        assert not torch.equal(provider.head.means, torch.zeros_like(provider.head.means))

        features = torch.randn(2, 8)
        logits = torch.randn(2, 3)
        out = provider.forward(features, logits)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()

    def test_full_fit_runs_scaler(self) -> None:
        torch, nn = _torch_modules()
        from probly.method.deup.torch import LogGMMDensity  # noqa: PLC0415

        encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        classification_head = nn.Linear(8, 3)
        loader = _tiny_classification_loader(num_classes=3, n_samples=12, feature_dim=4)

        provider = LogGMMDensity(num_classes=3, feature_dim=8)
        provider.fit(encoder, classification_head, loader, torch.device("cpu"))

        # After full fit() the MinMax scaler buffers should be populated.
        assert provider._scale_min is not None  # noqa: SLF001
        assert provider._scale_max is not None  # noqa: SLF001
        # Forward should now apply normalisation without errors.
        features = torch.randn(2, 8)
        logits = torch.randn(2, 3)
        out = provider.forward(features, logits)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()


class TestLogMCDropoutVarianceFit:
    """LogMCDropoutVariance._fit_scaler boosts n_samples then restores it."""

    def test_fit_scaler_restores_n_samples_after_run(self) -> None:
        torch, nn = _torch_modules()
        from probly.method.deup.torch import LogMCDropoutVariance  # noqa: PLC0415

        encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        classification_head = nn.Linear(8, 3)
        loader = _tiny_classification_loader(num_classes=3, n_samples=8, feature_dim=4)

        provider = LogMCDropoutVariance(n_samples=4, dropout_p=0.1)
        # _fit_internal stores the head; required for forward()/scaler.
        provider._fit_internal(encoder, classification_head, loader, torch.device("cpu"))  # noqa: SLF001
        provider._fit_scaler(encoder, classification_head, loader, torch.device("cpu"))  # noqa: SLF001

        # n_samples is restored to its original value after _fit_scaler exits.
        assert provider.n_samples == 4
        assert provider._scale_min is not None  # noqa: SLF001
        assert provider._scale_max is not None  # noqa: SLF001

    def test_full_fit_then_forward_with_classification_head(self) -> None:
        torch, nn = _torch_modules()
        from probly.method.deup.torch import LogMCDropoutVariance  # noqa: PLC0415

        encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        classification_head = nn.Linear(8, 3)
        loader = _tiny_classification_loader(num_classes=3, n_samples=8, feature_dim=4)

        provider = LogMCDropoutVariance(n_samples=4, dropout_p=0.1)
        provider.fit(encoder, classification_head, loader, torch.device("cpu"))

        # After full fit the head is stored and scaler is fitted.
        assert provider._classification_head is classification_head  # noqa: SLF001
        features = torch.randn(2, 8)
        logits = torch.randn(2, 3)
        out = provider.forward(features, logits)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()


class TestLogMAFDensityFit:
    """LogMAFDensity._fit_internal needs nflows; skip when missing."""

    def test_fit_internal_then_forward(self) -> None:
        pytest.importorskip("nflows")
        torch, nn = _torch_modules()
        from probly.method.deup.torch import LogMAFDensity  # noqa: PLC0415

        encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        classification_head = nn.Linear(8, 3)
        loader = _tiny_classification_loader(num_classes=3, n_samples=8, feature_dim=4)

        # Tiny flow + 1 epoch keeps the test fast.
        provider = LogMAFDensity(
            feature_dim=8,
            n_transforms=1,
            hidden_features=4,
            n_blocks_per_transform=1,
            flow_epochs=1,
        )
        provider._fit_internal(encoder, classification_head, loader, torch.device("cpu"))  # noqa: SLF001
        # Internal flow should be set after fitting.
        assert provider._flow is not None  # noqa: SLF001

        features = torch.randn(2, 8)
        logits = torch.randn(2, 3)
        out = provider.forward(features, logits)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()


class TestLogDUEVarianceFit:
    """LogDUEVariance._fit_internal needs gpytorch; skip when missing."""

    def test_fit_internal_then_forward(self) -> None:
        pytest.importorskip("gpytorch")
        torch, nn = _torch_modules()
        from probly.method.deup.torch import LogDUEVariance  # noqa: PLC0415

        encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        classification_head = nn.Linear(8, 3)
        loader = _tiny_classification_loader(num_classes=3, n_samples=12, feature_dim=4)

        # Use 1 epoch and a small inducing set so the test runs in well under a second.
        provider = LogDUEVariance(num_classes=3, feature_dim=8, n_inducing=4, gp_epochs=1)
        provider._fit_internal(encoder, classification_head, loader, torch.device("cpu"))  # noqa: SLF001

        # Internal GP / likelihood / scaler buffers should be populated.
        assert provider._gp_model is not None  # noqa: SLF001
        assert provider._likelihood is not None  # noqa: SLF001
        assert provider._feat_mean is not None  # noqa: SLF001
        assert provider._feat_std is not None  # noqa: SLF001
        assert provider._scale_min is not None  # noqa: SLF001
        assert provider._scale_max is not None  # noqa: SLF001

        features = torch.randn(2, 8)
        logits = torch.randn(2, 3)
        out = provider.forward(features, logits)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()

    def test_fit_scaler_is_noop(self) -> None:
        """LogDUEVariance overrides ``_fit_scaler`` to do nothing extra."""
        pytest.importorskip("gpytorch")
        torch, _ = _torch_modules()
        from probly.method.deup.torch import LogDUEVariance  # noqa: PLC0415

        provider = LogDUEVariance(num_classes=3, feature_dim=8)
        # Calling _fit_scaler before any internal state is set must not raise:
        # the override is intentionally a no-op (scaler is fitted inside
        # _fit_internal via _fit_scaler_from_features).
        provider._fit_scaler(  # noqa: SLF001
            encoder=None,  # type: ignore[arg-type]
            classification_head=None,  # type: ignore[arg-type]
            train_loader=None,  # type: ignore[arg-type]
            device=torch.device("cpu"),
        )

    def test_fit_scaler_from_features_unset_model_raises(self) -> None:
        pytest.importorskip("gpytorch")
        torch, _ = _torch_modules()
        from probly.method.deup.torch import LogDUEVariance  # noqa: PLC0415

        provider = LogDUEVariance(num_classes=3, feature_dim=8)
        with pytest.raises(RuntimeError, match="must be set"):
            provider._fit_scaler_from_features(torch.randn(4, 8), torch.device("cpu"))  # noqa: SLF001


class TestTorchDEUPPredictorSpectralNorm:
    """The predictor wires up spectral norm when a provider requires it."""

    def test_spectral_norm_auto_applied_for_due_variance(self) -> None:
        pytest.importorskip("gpytorch")
        torch, nn = _torch_modules()  # noqa: RUF059
        from torch.nn.utils import parametrize  # noqa: PLC0415

        from probly.method.deup.torch import TorchDEUPPredictor  # noqa: PLC0415

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        deup_pred = TorchDEUPPredictor(
            model,
            hidden_size=8,
            n_hidden_layers=1,
            stationarizing_features=["log_due_variance"],
        )

        # The remaining Linear in the encoder should have a parametrization
        # registered on its 'weight' attribute (i.e. spectral norm wired).
        encoder_linears = [m for m in deup_pred.encoder.modules() if isinstance(m, nn.Linear)]
        assert encoder_linears, "Encoder should contain at least one Linear after head removal."
        assert any(parametrize.is_parametrized(linear, "weight") for linear in encoder_linears)

    def test_explicit_sn_coeff_overrides_default(self) -> None:
        torch, nn = _torch_modules()  # noqa: RUF059
        from torch.nn.utils import parametrize  # noqa: PLC0415

        from probly.method.deup.torch import TorchDEUPPredictor  # noqa: PLC0415

        # log_gmm_density does NOT require spectral norm; passing sn_coeff
        # explicitly forces it on anyway.
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        deup_pred = TorchDEUPPredictor(
            model,
            hidden_size=8,
            n_hidden_layers=1,
            stationarizing_features=["log_gmm_density"],
            sn_coeff=2.0,
        )
        encoder_linears = [m for m in deup_pred.encoder.modules() if isinstance(m, nn.Linear)]
        assert any(parametrize.is_parametrized(linear, "weight") for linear in encoder_linears)


class TestTorchDEUPPredictorAfterFit:
    """End-to-end fit() then predict_representation() on a small model."""

    def test_predict_representation_after_fit(self) -> None:
        torch, nn = _torch_modules()
        from probly.method.deup.torch import TorchDEUPPredictor, TorchDEUPRepresentation  # noqa: PLC0415

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        deup_pred = TorchDEUPPredictor(
            model,
            hidden_size=8,
            n_hidden_layers=1,
            stationarizing_features=["log_gmm_density", "log_mc_dropout_variance"],
        )

        loader = _tiny_classification_loader(num_classes=3, n_samples=12, feature_dim=4)
        # Each provider has its own fit() — invoking those covers the full
        # provider.fit() path (both _fit_internal and _fit_scaler).
        for provider in deup_pred.providers:
            provider.fit(deup_pred.encoder, deup_pred.classification_head, loader, torch.device("cpu"))

        x = torch.randn(2, 4)
        rep = deup_pred.predict_representation(x)
        assert isinstance(rep, TorchDEUPRepresentation)
        assert rep.error_score.shape == (2,)
        # error_score = 10**log10_error >= 0 (untrained head can saturate to inf).
        assert torch.all(rep.error_score >= 0)
        # Softmax row sums to 1.
        torch.testing.assert_close(rep.softmax.tensor.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)
