"""Torch implementation of the DDU transformation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING, ClassVar
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, State, singledispatch_traverser, traverse_with_state

from ._common import (
    DDUPredictor,
    DDURepresentation,
    create_ddu_representation,
    ddu_generator,
    negative_log_density,
)

if TYPE_CHECKING:
    from torch import Tensor


@create_ddu_representation.register(TorchCategoricalDistribution)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchDDURepresentation(DDURepresentation, TorchAxisProtected):
    """DDU representation backed by torch tensors."""

    softmax: TorchCategoricalDistribution
    densities: Tensor
    protected_axes: ClassVar[dict[str, int]] = {"softmax": 0, "densities": 1}


@negative_log_density.register(torch.Tensor)
def torch_negative_log_density(densities: torch.Tensor) -> torch.Tensor:
    """Convert class-weighted log densities to negative GMM log density."""
    return -torch.logsumexp(densities, dim=-1)


SN_COEFF = GlobalVariable[float]("SN_COEFF", default=3.0)
HAS_RESIDUAL = GlobalVariable[bool]("HAS_RESIDUAL", default=False)
HEAD_MODULE: GlobalVariable[nn.Module | None] = GlobalVariable("HEAD_MODULE", default=None)


class _SNCoeffParametrization(nn.Module):
    """Weight parametrization that clips the spectral norm to at most coeff.

    Uses one step of power iteration to estimate the spectral norm and then
    scales the weight so that ||W||_2 <= coeff.
    """

    def __init__(self, coeff: float, weight: torch.Tensor) -> None:
        """Initialize the parametrization.

        Args:
            coeff: Maximum allowed spectral norm.
            weight: Reference weight tensor used for shape and device.
        """
        super().__init__()
        self.coeff = coeff
        self._u: torch.Tensor
        self.register_buffer("_u", F.normalize(torch.randn(weight.shape[0], device=weight.device), dim=0))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Rescale weight so that its spectral norm is at most coeff.

        Args:
            weight: The original weight tensor.

        Returns:
            Rescaled weight tensor.
        """
        w_mat = weight.reshape(weight.shape[0], -1)
        with torch.no_grad():
            v = F.normalize(w_mat.T @ self._u, dim=0)
            u = F.normalize(w_mat @ v, dim=0)
            self._u.copy_(u)
        sigma = u @ (w_mat @ v)
        return weight * (self.coeff / sigma.clamp(min=self.coeff))

    def right_inverse(self, weight: torch.Tensor) -> torch.Tensor:
        """Identity right-inverse to support state_dict loading.

        Args:
            weight: The parametrized weight.

        Returns:
            The weight unchanged.
        """
        return weight


@singledispatch_traverser
def residual_detection_traverser(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Detect residual connections by FX-tracing non-leaf modules for add operations.

    Sets HAS_RESIDUAL to True in the traversal state as soon as an addition
    node is found in any submodule's computation graph.
    """
    if state[HAS_RESIDUAL] or not list(obj.named_children()):
        return obj, state
    traced = torch.fx.symbolic_trace(obj)
    for node in traced.graph.nodes:
        if node.op == "call_function" and node.target in (torch.add, operator.add, operator.iadd):
            state[HAS_RESIDUAL] = True
            break
        if node.op == "call_method" and node.target == "add":
            state[HAS_RESIDUAL] = True
            break
    return obj, state


@singledispatch_traverser
def torch_ddu_traverser(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Default handler: return module unchanged."""
    return obj, state


@torch_ddu_traverser.register
def _(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace the classification head with Identity; apply spectral normalization to others.

    With TRAVERSE_REVERSED, the last Linear layer (the head) is encountered
    first. It is stored in HEAD_MODULE and replaced with ``nn.Identity()`` so
    the resulting model becomes a pure feature encoder.
    """
    if state[HEAD_MODULE] is None:
        state[HEAD_MODULE] = obj
        return nn.Identity(), state
    parametrize.register_parametrization(obj, "weight", _SNCoeffParametrization(state[SN_COEFF], obj.weight))
    return obj, state


@torch_ddu_traverser.register
def _(obj: nn.Conv2d, state: State) -> tuple[nn.Module, State]:
    """Apply spectral normalization to Conv2d layers.

    Stride-1x1 convolutions (typical residual-branch downsampling) are first
    replaced with an AvgPool2d followed by a stride-1 Conv2d to remove aliasing,
    then spectral normalization is applied to the new Conv2d.
    """
    stride = obj.stride[0] if isinstance(obj.stride, (tuple, list)) else obj.stride
    kernel = obj.kernel_size[0] if isinstance(obj.kernel_size, (tuple, list)) else obj.kernel_size
    if stride > 1 and kernel == 1:
        avg_pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        new_conv = nn.Conv2d(
            obj.in_channels,
            obj.out_channels,
            kernel_size=1,
            stride=1,
            padding=obj.padding if isinstance(obj.padding, int) else obj.padding[0],
            bias=obj.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(obj.weight)
            if obj.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(obj.bias)
        parametrize.register_parametrization(
            new_conv, "weight", _SNCoeffParametrization(state[SN_COEFF], new_conv.weight)
        )
        return nn.Sequential(avg_pool, new_conv), state
    parametrize.register_parametrization(obj, "weight", _SNCoeffParametrization(state[SN_COEFF], obj.weight))
    return obj, state


@torch_ddu_traverser.register
def _(obj: nn.ReLU, state: State) -> tuple[nn.Module, State]:  # noqa: ARG001
    """Replace ReLU with LeakyReLU(0.01)."""
    return nn.LeakyReLU(negative_slope=0.01, inplace=False), state


@torch_ddu_traverser.register
def _(obj: nn.ReLU6, state: State) -> tuple[nn.Module, State]:  # noqa: ARG001
    """Replace ReLU6 with LeakyReLU(0.01)."""
    return nn.LeakyReLU(negative_slope=0.01, inplace=False), state


class GaussianMixtureHead(nn.Module):
    """Per-class Gaussian density head for DDU.

    Implements the GDA density estimator from
    :cite:`mukhotiDeepDeterministicUncertainty2023`: one full-covariance
    Gaussian per class with class-frequency mixing weights.

    After fitting, ``forward`` returns the marginal log-density
    ``log q(z) = log sum_c pi_c * N(z; mu_c, Sigma_c)`` — a single scalar per
    sample used as the epistemic uncertainty score.

    Buffers:
        means: Per-class mean vectors, shape (num_classes, feature_dim).
        scale_tril: Lower-triangular Cholesky factor of each per-class covariance,
            shape (num_classes, feature_dim, feature_dim).
        log_pi: Log class-frequency priors, shape (num_classes,).
    """

    means: torch.Tensor
    scale_tril: torch.Tensor
    log_pi: torch.Tensor

    def __init__(self, num_classes: int, feature_dim: int) -> None:
        """Initialize with uniform priors, identity covariance, and zero means.

        Args:
            num_classes: Number of classes (one Gaussian component per class).
            feature_dim: Dimensionality of the feature vectors.
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.register_buffer("means", torch.zeros(num_classes, feature_dim))
        self.register_buffer(
            "scale_tril",
            torch.eye(feature_dim).unsqueeze(0).repeat(num_classes, 1, 1),
        )
        self.register_buffer(
            "log_pi",
            torch.full((num_classes,), -math.log(num_classes)),
        )
        # Jitter schedule for making sample covariances positive definite.
        # Start from 0, try increasingly large values until torch.linalg.cholesky succeeds.
        self._jitters = [0.0] + [10**exp for exp in range(-308, 0, 1)]

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Estimate per-class means, covariances, and mixing weights.

        Per-class means and unbiased sample covariances are computed from the
        provided training features.  The class-frequency prior
        ``pi_c = N_c / N`` is stored as ``log_pi``.  The minimum jitter
        ``eps * I`` that makes the Cholesky decomposition succeed is added to
        each covariance matrix before factorising, matching the robustness
        strategy of the reference implementation.

        Args:
            features: Feature vectors of shape (N, feature_dim).
            labels: Integer class labels of shape (N,).
        """
        n_total = len(labels)
        means = torch.zeros_like(self.means)
        scale_trils = torch.eye(self.feature_dim, device=features.device).unsqueeze(0).repeat(self.num_classes, 1, 1)
        log_pi = torch.full((self.num_classes,), float("-inf"), device=features.device)
        eye = torch.eye(self.feature_dim, device=features.device)
        for c in range(self.num_classes):
            mask = labels == c
            count = int(mask.sum().item())
            if count == 0:
                continue
            log_pi[c] = torch.tensor(count / n_total, device=features.device).log()
            if count < 2:
                continue
            z = features[mask]
            mu = z.mean(0)
            centered = z - mu
            cov = (centered.T @ centered) / (count - 1)
            means[c] = mu
            for jitter_eps in self._jitters:
                try:
                    scale_trils[c] = torch.linalg.cholesky(cov + jitter_eps * eye)
                    break
                except torch.linalg.LinAlgError:
                    continue
        self.means.copy_(means)
        self.scale_tril.copy_(scale_trils)
        self.log_pi.copy_(log_pi)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute per-class log-densities for each sample.

        Returns ``log(pi_c * N(z; mu_c, Sigma_c))`` for every class *c*,
        i.e. the class-prior-weighted Gaussian log-likelihood.

        Args:
            features: Feature vectors of shape (N, feature_dim).

        Returns:
            Per-class log-density scores of shape (N, num_classes).
        """
        dist = torch.distributions.MultivariateNormal(loc=self.means, scale_tril=self.scale_tril, validate_args=False)
        # log_prob broadcasts (N, 1, D) against batch shape (C,) -> (N, C)
        return self.log_pi + dist.log_prob(features.unsqueeze(1))


@ddu_generator.register(nn.Module)
class TorchDDUPredictor(nn.Module, DDUPredictor[[torch.Tensor], TorchDDURepresentation]):
    """Torch version of a DDU predictor.

    The traversal replaces the last ``nn.Linear`` (the classification head) with
    ``nn.Identity()``, producing a pure feature encoder.  The head is stored
    separately so the predictor has three clean, independent components:

    - ``encoder``: the spectrally-normalised backbone (head replaced with Identity).
    - ``classification_head``: the original last Linear layer.
    - ``density_head``: a :class:`GaussianMixtureHead` fitted post-training.

    Call ``density_head.fit(features, labels)`` after training to initialise
    the GMM parameters.
    """

    encoder: nn.Module
    classification_head: nn.Linear
    density_head: GaussianMixtureHead

    def __init__(self, model: nn.Module, sn_coeff: float = 3.0) -> None:
        """Build the three-component DDU predictor from a base model.

        Args:
            model: Base classification model to be transformed.
            sn_coeff: Lipschitz coefficient for spectral normalization.
        """
        super().__init__()
        encoder, state = traverse_with_state(
            model,
            nn_compose(residual_detection_traverser, torch_ddu_traverser, nn_traverser=nn_traverser),
            init={HAS_RESIDUAL: False, TRAVERSE_REVERSED: True, SN_COEFF: sn_coeff, HEAD_MODULE: None},
        )
        head: nn.Linear | None = state[HEAD_MODULE]  # ty: ignore[invalid-assignment]
        if head is None:
            msg = "No nn.Linear layer found in the model; cannot identify a classification head."
            raise ValueError(msg)

        if not state[HAS_RESIDUAL]:
            warnings.warn(
                "No residual connections detected in the given model. DDU is designed for models "
                "with residual connections (e.g. ResNets). Without them, features may collapse "
                "under spectral normalization and density estimates will be unreliable. "
                "Please consider to use a different method or to add residual connections to your model.",
                UserWarning,
                stacklevel=2,
            )

        self.encoder = encoder
        self.classification_head = head
        self.density_head = GaussianMixtureHead(head.out_features, head.in_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode features, classify, and score density.

        Args:
            x: Input tensor passed to the encoder.

        Returns:
            A 2-tuple of ``(logits, density_scores)`` where both have shape
            ``(N, num_classes)``.
        """
        features = self.encoder(x)
        logits = self.classification_head(features)
        densities = self.density_head(features)
        return logits, densities

    def fit_density_head(self, x: torch.Tensor, labels: torch.Tensor) -> None:
        """Fit the GMM density head to the given features and labels."""
        encoder_state = self.encoder.training  # save state
        density_state = self.density_head.training
        self.encoder.eval()
        self.density_head.train()
        features = self.encoder(x).detach()
        self.density_head.fit(features, labels)
        self.encoder.train(encoder_state)  # restore state
        self.density_head.train(density_state)

    def predict_representation(self, x: torch.Tensor) -> TorchDDURepresentation:
        """Predict the DDU representation (logits and density scores)."""
        logits, densities = self.forward(x)

        return TorchDDURepresentation(
            TorchCategoricalDistribution(torch.softmax(logits, dim=-1)),
            densities,
        )
