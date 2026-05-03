"""Torch implementation of Direct Epistemic Uncertainty Prediction (DEUP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from nflows.flows.autoregressive import MaskedAutoregressiveFlow as _MaskedAutoregressiveFlow

    _NFLOWS_AVAILABLE = True
except ImportError:
    _NFLOWS_AVAILABLE = False

try:
    import gpytorch as _gpytorch

    _GPYTORCH_AVAILABLE = True
except ImportError:
    _GPYTORCH_AVAILABLE = False

from probly.layers.torch import GaussianMixtureHead, apply_spectral_norm_to_encoder
from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, State, singledispatch_traverser, traverse_with_state

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch.utils.data import DataLoader

from ._common import (
    DEUPPredictor,
    DEUPRepresentation,
    create_deup_representation,
    deup_generator,
)


@create_deup_representation.register(TorchCategoricalDistribution)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchDEUPRepresentation(DEUPRepresentation, TorchAxisProtected):
    """DEUP representation backed by torch tensors.

    Args:
        softmax: Softmax probabilities of the base classifier,
            shape ``(batch, num_classes)``.
        error_score: Predicted per-sample expected cross-entropy from the
            error head, shape ``(batch,)``.
    """

    softmax: TorchCategoricalDistribution
    error_score: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"softmax": 0}


HEAD_MODULE: GlobalVariable[nn.Module | None] = GlobalVariable("DEUP_HEAD_MODULE", default=None)


@singledispatch_traverser
def torch_deup_traverser(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Default handler: return module unchanged."""
    return obj, state


@torch_deup_traverser.register(nn.Linear)
def _(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace the last Linear (hit first under TRAVERSE_REVERSED) with Identity.

    The original layer is saved in ``HEAD_MODULE`` so its shape can be used
    to build the classification and error heads.
    """
    if state[HEAD_MODULE] is None:
        state[HEAD_MODULE] = obj
        return nn.Identity(), state
    return obj, state


class ErrorPredictionHead(nn.Module):
    r"""MLP that predicts :math:`\log_{10}` of the per-sample loss.

    Following the reference implementation of
    :cite:`lahlou2021deup`, the head takes **only**
    stationarizing features :math:`\phi(x) \in \mathbb{R}^k` (e.g. log-density
    and log-model-variance) and outputs a real-valued scalar trained with MSE
    against :math:`\log_{10}(\ell(x))` — the base-10 logarithm of the
    per-sample loss, clamped from below at ``-5``.  The output is therefore
    real-valued (no non-negativity constraint inside the network); the actual
    uncertainty score is obtained at inference by back-transforming with
    :math:`10^{\hat{e}(x)}`.

    Architecture:
    ``input_dim -> (hidden_size x n_hidden_layers) -> 1``
    with ReLU activations between layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 1024,
        n_hidden_layers: int = 5,
    ) -> None:
        r"""Initialize the error prediction head.

        Args:
            input_dim: Total dimensionality of stationarizing features fed
                to this head (sum of :attr:`StationarizingFeatureProvider.output_dim`
                over all registered providers).
            hidden_size: Width of each hidden layer.
            n_hidden_layers: Number of hidden layers (minimum 1).
        """
        super().__init__()
        n_hidden_layers = max(1, n_hidden_layers)
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, stationarizing_features: torch.Tensor) -> torch.Tensor:
        r"""Predict :math:`\log_{10}` of the per-sample loss.

        Args:
            stationarizing_features: Stationarizing feature vectors of shape
                ``(batch, input_dim)``.

        Returns:
            Predicted :math:`\log_{10}(\ell(x))` of shape ``(batch,)``.
            Apply :math:`10^{\hat{e}}` at call site to recover the
            loss-scale uncertainty score.
        """
        return self.net(stationarizing_features).squeeze(-1)


class StationarizingFeatureProvider(nn.Module):
    """Base class for DEUP stationarizing feature providers.

    A provider produces a ``(batch, output_dim)`` tensor from
    ``(features, logits)``, optionally after a one-shot ``fit`` step on
    training data.  Outputs from all providers are concatenated and fed
    **directly** to the error head — encoder features are not included.

    The base class handles MinMax normalization automatically: after fitting,
    :meth:`fit` calls :meth:`_fit_scaler` which computes per-output-dim min/max
    over training data and stores them as buffers.  Subclasses call
    ``self._normalize(raw)`` at the end of :meth:`forward` to apply the
    scaling consistently.

    Subclasses must set :attr:`output_dim` and override :meth:`forward` and
    optionally :meth:`_fit_internal`.

    If a subclass sets ``requires_spectral_norm = True``, the
    :class:`TorchDEUPPredictor` will automatically apply spectral normalization
    to the encoder when that provider is used (e.g. :class:`LogDUEVariance`).
    """

    output_dim: int
    requires_spectral_norm: ClassVar[bool] = False

    def __init__(self) -> None:
        """Initialize the provider with unset scaler buffers."""
        super().__init__()
        self.register_buffer("_scale_min", None)
        self.register_buffer("_scale_max", None)

    def _fit_internal(
        self,
        encoder: nn.Module,
        classification_head: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        amp_enabled: bool = False,
    ) -> None:
        """Fit provider-specific state from training data. Default: no-op."""

    @torch.no_grad()
    def _fit_scaler(
        self,
        encoder: nn.Module,
        classification_head: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        amp_enabled: bool = False,
    ) -> None:
        """Compute and store per-output-dim MinMax scaler from training data."""
        encoder.eval()
        classification_head.eval()
        was_training = self.training
        self.eval()
        all_out: list[torch.Tensor] = []
        for inputs_, _ in tqdm(train_loader, desc=f"Fitting {type(self).__name__} scaler"):
            inputs = inputs_.to(device, non_blocking=True)
            if device.type == "cuda" and inputs.ndim >= 4:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                feats = encoder(inputs)
                lgts = classification_head(feats)
                out = self.forward(feats, lgts)
            all_out.append(out.detach().cpu())
        self.train(was_training)
        stacked = torch.cat(all_out)  # (N, output_dim)
        self._scale_min = stacked.min(dim=0).values
        self._scale_max = stacked.max(dim=0).values

    def fit(
        self,
        encoder: nn.Module,
        classification_head: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        amp_enabled: bool = False,
    ) -> None:
        """Fit internal state then the MinMax scaler on training data.

        Args:
            encoder: Frozen encoder producing feature vectors.
            classification_head: Frozen classification head.
            train_loader: Training data loader.
            device: Device to run inference on.
            amp_enabled: Whether to use automatic mixed precision.
        """
        self._fit_internal(encoder, classification_head, train_loader, device, amp_enabled)
        self._fit_scaler(encoder, classification_head, train_loader, device, amp_enabled)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fitted MinMax normalization; identity if scaler not yet fitted."""
        if self._scale_min is None or self._scale_max is None:
            return x
        lo = self._scale_min.to(x.device)
        hi = self._scale_max.to(x.device)
        return (x - lo) / (hi - lo + 1e-8)

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Compute stationarizing features for a batch.

        Args:
            features: Encoder features of shape ``(batch, feature_dim)``.
            logits: Classification logits of shape ``(batch, num_classes)``.

        Returns:
            Normalized stationarizing features of shape ``(batch, output_dim)``.
        """
        raise NotImplementedError


class LogGMMDensity(StationarizingFeatureProvider):
    r"""Per-class GMM log-density of encoder features.

    Approximates the density-based stationarizing feature
    :math:`\log \hat{q}(x|D)` of
    :cite:`lahlou2021deup` (Sec. 2.2) by fitting a
    Gaussian Discriminant Analysis model on training-set encoder
    features (one full-covariance Gaussian per class with class-frequency
    mixing weights) and returning the marginal
    :math:`\log \hat{q}(z) = \mathrm{logsumexp}_c \log(\pi_c \,
    \mathcal{N}(z; \mu_c, \Sigma_c))`.

    The paper uses a MAF on raw inputs; we substitute a feature-space
    GMM to avoid an extra normalising-flow dependency, following the
    same justification as DDU :cite:`mukhotiDeepDeterministicUncertainty2023`.
    """

    output_dim: int = 1

    def __init__(self, num_classes: int, feature_dim: int) -> None:
        """Allocate a per-class GMM with the given dimensions; not yet fitted.

        Args:
            num_classes: Number of class components in the mixture.
            feature_dim: Dimensionality of the encoder feature space.
        """
        super().__init__()
        self.head = GaussianMixtureHead(num_classes, feature_dim)

    @torch.no_grad()
    def _fit_internal(
        self,
        encoder: nn.Module,
        classification_head: nn.Module,  # noqa: ARG002
        train_loader: DataLoader,
        device: torch.device,
        amp_enabled: bool = False,
    ) -> None:
        """Extract training features through ``encoder`` and fit the GMM on CPU."""
        encoder.eval()
        all_features: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for inputs_, targets_ in tqdm(train_loader, desc="Fitting DEUP density feature"):
            inputs = inputs_.to(device, non_blocking=True)
            if device.type == "cuda" and inputs.ndim >= 4:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            targets = targets_.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                features = encoder(inputs)
            all_features.append(features.detach().cpu())
            all_labels.append(targets.detach().cpu())
        features_cat = torch.cat(all_features)
        labels_cat = torch.cat(all_labels)
        head_device = self.head.means.device
        self.head.cpu()
        self.head.fit(features_cat, labels_cat)
        self.head.to(head_device)

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        """Return the normalized marginal log-density per sample."""
        per_class = self.head(features)  # (B, C)
        raw = torch.logsumexp(per_class, dim=-1, keepdim=True)
        return self._normalize(raw)


class LogMCDropoutVariance(StationarizingFeatureProvider):
    r"""MC-Dropout variance of the classifier as a model-uncertainty proxy.

    Approximates the model-variance stationarizing feature
    :math:`\log \hat{V}` of
    :cite:`lahlou2021deup` by applying
    feature-level dropout stochastically at inference time:

    1. The frozen classification head is stored during :meth:`_fit_internal`.
    2. At :meth:`forward` time, the incoming encoder features are perturbed with
       :func:`torch.nn.functional.dropout` (``training=True``) for
       ``n_samples`` independent passes through the stored head.
    3. The **total variance** of the resulting softmax distributions is taken as
       the variance signal and returned on a log scale.

    Unlike softmax entropy, this signal does not collapse for OOD inputs that
    the classifier handles confidently, making it a better proxy for
    epistemic uncertainty.
    """

    output_dim: int = 1

    def __init__(self, n_samples: int = 10, dropout_p: float = 0.1, eps: float = 1e-12) -> None:
        """Initialise the MC-Dropout variance provider.

        Args:
            n_samples: Number of stochastic forward passes to average over.
            dropout_p: Dropout probability applied to encoder features.
            eps: Numerical-stability floor added before the log.
        """
        super().__init__()
        self.n_samples = n_samples
        self.dropout_p = dropout_p
        self.eps = eps
        self._classification_head: nn.Module | None = None

    def _fit_internal(
        self,
        encoder: nn.Module,  # noqa: ARG002
        classification_head: nn.Module,
        train_loader: DataLoader,  # noqa: ARG002
        device: torch.device,  # noqa: ARG002
        amp_enabled: bool = False,  # noqa: ARG002
    ) -> None:
        """Store the classification head for use at forward time."""
        self._classification_head = classification_head

    @torch.no_grad()
    def _fit_scaler(
        self,
        encoder: nn.Module,
        classification_head: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        amp_enabled: bool = False,
    ) -> None:
        """Fit scaler using more MC samples for a stable min/max estimate.

        ``F.dropout(training=True)`` is inherently stochastic, so a small
        ``n_samples`` produces a noisy min/max.  Using
        ``5 * n_samples`` (at least 50) during this one-off calibration
        pass averages out variance and gives a reliable normalization range.
        """
        old_n = self.n_samples
        self.n_samples = max(self.n_samples * 5, 50)
        try:
            super()._fit_scaler(encoder, classification_head, train_loader, device, amp_enabled)
        finally:
            self.n_samples = old_n

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        r"""Return :math:`\log(\mathrm{Var}[\mathrm{softmax}] + \varepsilon)` under feature dropout."""
        if self._classification_head is None:
            msg = "LogMCDropoutVariance must be fitted before forward(). Call fit() first."
            raise RuntimeError(msg)
        probs: list[torch.Tensor] = []
        for _ in range(self.n_samples):
            dropped = F.dropout(features, p=self.dropout_p, training=True)
            probs.append(F.softmax(self._classification_head(dropped), dim=-1))
        variance = torch.stack(probs, dim=0).var(dim=0).sum(dim=-1, keepdim=True)
        return self._normalize(torch.log(variance + self.eps))


class LogMAFDensity(StationarizingFeatureProvider):
    r"""Masked Autoregressive Flow log-density of encoder features.

    Replaces :class:`LogGMMDensity` with a proper normalizing flow, giving a
    more expressive density estimate :math:`\log \hat{q}(z|D)` in the encoder
    feature space.  Uses :class:`nflows.flows.autoregressive.MaskedAutoregressiveFlow`
    trained by maximum likelihood on training-set encoder features.

    Args:
        feature_dim: Dimensionality of encoder features.
        n_transforms: Number of autoregressive transforms (flow depth).
        hidden_features: Width of each MADE sub-network.
        n_blocks_per_transform: Number of residual blocks per transform.
        flow_lr: Adam learning rate for flow training.
        flow_epochs: Number of training epochs.
    """

    output_dim: int = 1

    def __init__(
        self,
        feature_dim: int,
        n_transforms: int = 5,
        hidden_features: int = 256,
        n_blocks_per_transform: int = 2,
        flow_lr: float = 5e-4,
        flow_epochs: int = 20,
    ) -> None:
        """Initialize the MAF density provider."""
        super().__init__()
        self.feature_dim = feature_dim
        self.n_transforms = n_transforms
        self.hidden_features = hidden_features
        self.n_blocks_per_transform = n_blocks_per_transform
        self.flow_lr = flow_lr
        self.flow_epochs = flow_epochs
        self._flow: nn.Module | None = None

    def _build_flow(self) -> nn.Module:
        if not _NFLOWS_AVAILABLE:
            msg = "nflows is required for LogMAFDensity. Install with: pip install nflows"
            raise ImportError(msg)
        return _MaskedAutoregressiveFlow(
            features=self.feature_dim,
            hidden_features=self.hidden_features,
            num_layers=self.n_transforms,
            num_blocks_per_layer=self.n_blocks_per_transform,
            use_residual_blocks=True,
            use_random_masks=False,
            activation=F.relu,
            dropout_probability=0.0,
            batch_norm_within_layers=False,
            batch_norm_between_layers=False,
        )

    def _fit_internal(
        self,
        encoder: nn.Module,
        classification_head: nn.Module,  # noqa: ARG002
        train_loader: DataLoader,
        device: torch.device,
        amp_enabled: bool = False,
    ) -> None:
        """Extract training features and train the MAF by NLL minimization."""
        encoder.eval()
        all_features: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs_, _ in tqdm(train_loader, desc="Extracting features for MAF"):
                inputs = inputs_.to(device, non_blocking=True)
                if device.type == "cuda" and inputs.ndim >= 4:
                    inputs = inputs.contiguous(memory_format=torch.channels_last)
                with torch.amp.autocast(device.type, enabled=amp_enabled):
                    features = encoder(inputs)
                all_features.append(features.detach().cpu().float())
        features_cat = torch.cat(all_features)

        flow = self._build_flow().to(device)
        optimizer = torch.optim.Adam(flow.parameters(), lr=self.flow_lr)
        batch_size = 256
        for epoch in range(self.flow_epochs):
            flow.train()
            perm = torch.randperm(features_cat.size(0))
            epoch_loss = torch.zeros(1, device=device)
            n_batches = 0
            for i in range(0, features_cat.size(0), batch_size):
                # Index features_cat directly to avoid a full shuffled copy per epoch.
                batch = features_cat[perm[i : i + batch_size]].to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = -flow.log_prob(batch.float()).mean()  # ty: ignore[call-non-callable]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach()
                n_batches += 1
            if (epoch + 1) % 5 == 0 or epoch == self.flow_epochs - 1:
                tqdm.write(f"MAF epoch {epoch + 1}/{self.flow_epochs}, NLL={epoch_loss.item() / max(n_batches, 1):.4f}")
        flow.eval()
        self._flow = flow

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        """Return the normalized MAF log-density per sample."""
        if self._flow is None:
            msg = "LogMAFDensity must be fitted before forward(). Call fit() first."
            raise RuntimeError(msg)
        flow = self._flow.to(features.device)
        with torch.no_grad():
            log_prob = flow.log_prob(features.float()).unsqueeze(-1)  # ty: ignore[call-non-callable]
        return self._normalize(log_prob)


class LogDUEVariance(StationarizingFeatureProvider):
    r"""SVGP posterior variance as a distance-aware model-uncertainty proxy.

    Implements the DUE variance stationarizing feature
    :math:`\log \hat{V}` following :cite:`amersfoortSimpleScalableEpistemic2021`.
    Fits a multi-output Sparse Variational GP (SVGP, one output per class) using
    :mod:`gpytorch` on encoder features, then returns the sum of posterior
    variances across class outputs on a log scale.

    Unlike :class:`LogMCDropoutVariance`, the GP posterior variance is
    distance-aware: it is high for inputs far from the training manifold even
    when the encoder maps them near training-distribution features.

    Sets ``requires_spectral_norm = True`` so that :class:`TorchDEUPPredictor`
    automatically applies spectral normalization to the encoder, which is
    required for the feature-space distances to be meaningful (per DUE).

    Args:
        num_classes: Number of classification outputs (GP tasks).
        feature_dim: Dimensionality of encoder features.
        n_inducing: Number of SVGP inducing points.
        gp_lr: Adam learning rate for SVGP training.
        gp_epochs: Number of SVGP training epochs.
        eps: Numerical-stability floor added inside the log.
    """

    output_dim: int = 1
    requires_spectral_norm: ClassVar[bool] = True

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,  # noqa: ARG002
        n_inducing: int = 512,
        gp_lr: float = 1e-2,
        gp_epochs: int = 20,
        eps: float = 1e-12,
    ) -> None:
        """Initialize the DUE variance provider."""
        super().__init__()
        self.num_classes = num_classes
        self.n_inducing = n_inducing
        self.gp_lr = gp_lr
        self.gp_epochs = gp_epochs
        self.eps = eps
        self._gp_model: Any | None = None
        self._likelihood: Any | None = None

    def _fit_internal(
        self,
        encoder: nn.Module,
        classification_head: nn.Module,  # noqa: ARG002
        train_loader: DataLoader,
        device: torch.device,
        amp_enabled: bool = False,
    ) -> None:
        """Extract training features and fit a multi-output SVGP."""
        if not _GPYTORCH_AVAILABLE:
            msg = "gpytorch is required for LogDUEVariance. Install with: pip install gpytorch"
            raise ImportError(msg)
        gpytorch = _gpytorch

        encoder.eval()
        all_features: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs_, targets_ in tqdm(train_loader, desc="Extracting features for DUE GP"):
                inputs = inputs_.to(device, non_blocking=True)
                if device.type == "cuda" and inputs.ndim >= 4:
                    inputs = inputs.contiguous(memory_format=torch.channels_last)
                with torch.amp.autocast(device.type, enabled=amp_enabled):
                    features = encoder(inputs)
                all_features.append(features.detach().cpu().float())
                all_labels.append(targets_.cpu())
        features_cat = torch.cat(all_features)
        labels_cat = torch.cat(all_labels).long()

        perm = torch.randperm(features_cat.size(0))[: self.n_inducing]
        inducing_points = features_cat[perm].unsqueeze(0).expand(self.num_classes, -1, -1)

        num_classes = self.num_classes

        class _SVGPModel(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_pts: torch.Tensor) -> None:
                var_dist = gpytorch.variational.CholeskyVariationalDistribution(
                    inducing_pts.size(-2), batch_shape=torch.Size([num_classes])
                )
                var_strat = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                    gpytorch.variational.VariationalStrategy(
                        self, inducing_pts, var_dist, learn_inducing_locations=True
                    ),
                    num_tasks=num_classes,
                )
                super().__init__(var_strat)
                self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_classes]))
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_classes])),
                    batch_shape=torch.Size([num_classes]),
                )

            def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
                return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

        model = _SVGPModel(inducing_points)
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=self.num_classes, num_classes=self.num_classes)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=features_cat.size(0))
        optimizer = torch.optim.Adam([*model.parameters(), *likelihood.parameters()], lr=self.gp_lr)

        batch_size = 256
        for epoch in range(self.gp_epochs):
            model.train()
            likelihood.train()
            perm = torch.randperm(features_cat.size(0))
            total_loss = 0.0
            n_batches = 0
            for i in range(0, features_cat.size(0), batch_size):
                idx = perm[i : i + batch_size]
                bf, bl = features_cat[idx], labels_cat[idx]
                optimizer.zero_grad()
                output = model(bf)
                loss = -mll(output, bl)  # ty: ignore[unsupported-operator]
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 5 == 0 or epoch == self.gp_epochs - 1:
                tqdm.write(f"DUE GP epoch {epoch + 1}/{self.gp_epochs}, ELBO={-total_loss / n_batches:.4f}")

        model.eval()
        likelihood.eval()
        self._gp_model = model
        self._likelihood = likelihood
        self._fit_scaler_from_features(features_cat)

    @torch.no_grad()
    def _fit_scaler_from_features(self, features_cat: torch.Tensor) -> None:
        """Compute MinMax scaler from already-extracted CPU features.

        Avoids a second full pass through the training loader in ``_fit_scaler``.
        """
        if self._gp_model is None:
            msg = "LogDUEVariance._gp_model must be set before calling _fit_scaler_from_features."
            raise RuntimeError(msg)
        model = self._gp_model
        all_log_var: list[torch.Tensor] = []
        batch_size = 256
        for i in tqdm(range(0, features_cat.size(0), batch_size), desc="Fitting DUE variance scaler"):
            bf = features_cat[i : i + batch_size]
            pred = model(bf)
            var = pred.variance.sum(dim=-1, keepdim=True)
            all_log_var.append(torch.log(var + self.eps))
        stacked = torch.cat(all_log_var)
        self._scale_min = stacked.min(dim=0).values
        self._scale_max = stacked.max(dim=0).values

    @torch.no_grad()
    def _fit_scaler(
        self,
        encoder: nn.Module,
        classification_head: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        amp_enabled: bool = False,
    ) -> None:
        """Skip: scaler already computed at the end of ``_fit_internal``."""

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        r"""Return :math:`\log(\sum_c \hat{V}_c(x) + \varepsilon)` from GP posterior variance."""
        if self._gp_model is None:
            msg = "LogDUEVariance must be fitted before forward(). Call fit() first."
            raise RuntimeError(msg)
        model = self._gp_model
        with torch.no_grad():
            pred = model(features.cpu().float())
            variance = pred.variance.sum(dim=-1, keepdim=True).to(features.device)
        return self._normalize(torch.log(variance + self.eps))


_PROVIDER_REGISTRY: dict[str, type[StationarizingFeatureProvider]] = {
    "log_gmm_density": LogGMMDensity,
    "log_mc_dropout_variance": LogMCDropoutVariance,
    "log_maf_density": LogMAFDensity,
    "log_due_variance": LogDUEVariance,
}


def _build_provider(
    spec: str | Mapping[str, object], num_classes: int, feature_dim: int
) -> StationarizingFeatureProvider:
    """Instantiate one provider from a string name or ``{name, **kwargs}`` dict."""
    if isinstance(spec, str):
        name, kwargs = spec, {}
    else:
        spec = dict(spec)
        name = str(spec.pop("name"))
        kwargs = spec
    if name not in _PROVIDER_REGISTRY:
        known = ", ".join(sorted(_PROVIDER_REGISTRY))
        msg = f"Unknown DEUP stationarizing feature {name!r}; known: {known}."
        raise ValueError(msg)
    cls = _PROVIDER_REGISTRY[name]
    if cls in (LogGMMDensity, LogDUEVariance):
        kwargs.setdefault("num_classes", num_classes)
        kwargs.setdefault("feature_dim", feature_dim)
    elif cls is LogMAFDensity:
        kwargs.setdefault("feature_dim", feature_dim)
    return cls(**kwargs)


def _build_providers(
    specs: Iterable[str | Mapping[str, object]] | None,
    num_classes: int,
    feature_dim: int,
) -> list[StationarizingFeatureProvider]:
    if not specs:
        return []
    return [_build_provider(s, num_classes, feature_dim) for s in specs]


@deup_generator.register(nn.Module)
class TorchDEUPPredictor(nn.Module, DEUPPredictor[[torch.Tensor], TorchDEUPRepresentation]):
    r"""Torch implementation of a DEUP predictor.

    The traversal strips the last ``nn.Linear`` (the classification head) and
    replaces it with ``nn.Identity()``, turning the backbone into a pure
    feature encoder.  The original head is stored as ``classification_head``
    for phase-1 cross-entropy training.  A fresh :class:`ErrorPredictionHead`
    is attached as ``error_head`` and trained in a separate phase on held-out
    per-sample losses.

    **Phase 1** trains ``encoder`` and ``classification_head`` with standard
    cross-entropy (identical to a plain classifier).

    **Phase 2** freezes the main model; stationarizing feature providers are
    fitted on the training set, then ``error_head`` is trained on
    ``(stationarizing_features, log10_ce_target)`` pairs from the calibration
    set, minimising MSE against :math:`\log_{10}(\ell(x))` — the base-10
    log of the per-sample cross-entropy, clamped to ``[-5, 0]``.

    At inference, ``error_head`` outputs :math:`\log_{10}(\ell(x))` which is
    back-transformed to :math:`10^{\hat{e}(x)}` to recover the loss-scale
    uncertainty score.
    """

    encoder: nn.Module
    classification_head: nn.Linear
    error_head: ErrorPredictionHead

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int = 1024,
        n_hidden_layers: int = 5,
        stationarizing_features: Sequence[str | Mapping[str, object]] | None = None,
        sn_coeff: float | None = None,
    ) -> None:
        """Build the three-component DEUP predictor from a base model.

        Args:
            model: Base classification model whose last ``nn.Linear`` defines
                the feature dimension and number of classes.
            hidden_size: Width of each hidden layer in the error head.
            n_hidden_layers: Number of hidden layers in the error head.
            stationarizing_features: Sequence of stationarizing feature
                specifications.  Each entry is either a registry name
                (e.g. ``"log_maf_density"``) or a mapping
                ``{"name": ..., **kwargs}``.  Their scalar outputs are
                concatenated and fed **directly** to the error head —
                encoder features are excluded from the error head input,
                matching the reference implementation.  At least one
                provider must be given.
            sn_coeff: Lipschitz coefficient for spectral normalization of the
                encoder.  Defaults to ``None`` (no spectral norm).  When a
                provider with ``requires_spectral_norm=True`` is present (e.g.
                :class:`LogDUEVariance`), spectral norm is automatically
                enabled with coefficient 3.0 unless overridden here.

        Raises:
            ValueError: If no stationarizing feature providers are given
                (the error head has no input without them).
        """
        super().__init__()
        encoder, state = traverse_with_state(
            model,
            nn_compose(torch_deup_traverser, nn_traverser=nn_traverser),
            init={TRAVERSE_REVERSED: True, HEAD_MODULE: None},
        )
        head: nn.Linear | None = state[HEAD_MODULE]  # ty: ignore[invalid-assignment]
        if head is None:
            msg = "No nn.Linear layer found in the model; cannot identify a classification head."
            raise ValueError(msg)

        providers = _build_providers(stationarizing_features, head.out_features, head.in_features)
        if not providers:
            msg = (
                "TorchDEUPPredictor requires at least one stationarizing feature provider. "
                "Pass e.g. stationarizing_features=['log_maf_density', 'log_due_variance']."
            )
            raise ValueError(msg)

        needs_sn = any(getattr(p, "requires_spectral_norm", False) for p in providers)
        effective_sn = sn_coeff if sn_coeff is not None else (3.0 if needs_sn else None)
        if effective_sn is not None:
            apply_spectral_norm_to_encoder(encoder, effective_sn)
        self.encoder = encoder
        self.classification_head = head
        self.providers = nn.ModuleList(providers)
        input_dim = sum(p.output_dim for p in providers)
        self.error_head = ErrorPredictionHead(
            input_dim=input_dim,
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden_layers,
        )

    def _compute_stationarizing_features(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Concatenate outputs of all registered providers into one tensor."""
        return torch.cat([p(features, logits) for p in self.providers], dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Encode, classify, and predict per-sample loss on log10 scale.

        Args:
            x: Input tensor.

        Returns:
            A 2-tuple ``(logits, error_score)`` where ``logits`` has shape
            ``(batch, num_classes)`` and ``error_score = 10^{\hat{e}(x)}``
            has shape ``(batch,)`` — the predicted per-sample loss in natural
            units, back-transformed from the log10-scaled head output.
        """
        features = self.encoder(x)
        logits = self.classification_head(features)
        phi = self._compute_stationarizing_features(features, logits)
        log10_error = self.error_head(phi)
        error_score = torch.pow(torch.full_like(log10_error, 10.0), log10_error)
        return logits, error_score

    def predict_representation(self, x: torch.Tensor) -> TorchDEUPRepresentation:
        """Predict the DEUP representation (softmax + error score).

        Args:
            x: Input tensor.

        Returns:
            A :class:`TorchDEUPRepresentation` holding the softmax distribution
            and the predicted per-sample cross-entropy.
        """
        logits, error_score = self.forward(x)
        return TorchDEUPRepresentation(
            softmax=TorchCategoricalDistribution(torch.softmax(logits, dim=-1)),
            error_score=error_score,
        )
