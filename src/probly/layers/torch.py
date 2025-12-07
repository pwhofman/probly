"""torch layer implementations."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class BayesLinear(nn.Module):
    """Implements a Bayesian linear layer.

    Attributes:
        in_features: int, number of input features
        out_features: int, number of output features
        bias: bool, whether to use a bias term
        weight_mu: torch.Tensor, mean of the posterior weights
        weight_rho: torch.Tensor, transformed standard deviation of the posterior weights
        weight_prior_mu: torch.Tensor, mean of the prior weights
        weight_prior_sigma: torch.Tensor, standard deviation of the prior weights
        bias_mu: torch.Tensor, mean of the posterior bias
        bias_rho: torch.Tensor, transformed standard deviation of the posterior bias
        bias_prior_mu: torch.Tensor, mean of the prior bias
        bias_prior_sigma: torch.Tensor, standard deviation of the prior bias
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        use_base_weights: bool = False,
        posterior_std: float = 0.05,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ) -> None:
        """Initializes the Bayesian linear layer.

        Reparameterize the standard deviation of the posterior weights using the re-parameterization trick.

        Args:
            base_layer: The original linear layer to be used.
            use_base_weights: Whether to use the weights of the base layer as prior means. Default is False.
            posterior_std: float, initial standard deviation of the posterior
            prior_mean: float, mean of the prior
            prior_std: float, standard deviation of the prior
        """
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.bias = base_layer.bias is not None

        # transform standard deviation for the re-parametrization trick
        rho = cast("float", _inverse_softplus(torch.tensor(posterior_std)))

        # posterior weights
        if not use_base_weights:
            self.weight_mu = nn.Parameter(torch.empty((self.out_features, self.in_features)))
        else:
            self.weight_mu = nn.Parameter(cast("torch.Tensor", base_layer.weight.data))
        self.weight_rho = nn.Parameter(torch.full((self.out_features, self.in_features), rho))

        # prior weights
        if not use_base_weights:
            self.register_buffer(
                "weight_prior_mu",
                torch.full((self.out_features, self.in_features), prior_mean),
            )
        else:
            self.register_buffer(
                "weight_prior_mu",
                cast("torch.Tensor", base_layer.weight.data),
            )
        self.register_buffer(
            "weight_prior_sigma",
            torch.full((self.out_features, self.in_features), prior_std),
        )

        if self.bias:
            # posterior bias
            if not use_base_weights:
                self.bias_mu = nn.Parameter(torch.empty((self.out_features,)))
            else:
                self.bias_mu = nn.Parameter(cast("torch.Tensor", base_layer.bias.data))
            self.bias_rho = nn.Parameter(
                torch.full((self.out_features,), rho),
            )

            # prior bias
            if not use_base_weights:
                self.register_buffer(
                    "bias_prior_mu",
                    torch.full((self.out_features,), prior_mean),
                )
            else:
                self.register_buffer(
                    "bias_prior_mu",
                    cast("torch.Tensor", base_layer.bias.data),
                )
            self.register_buffer(
                "bias_prior_sigma",
                torch.full((self.out_features,), prior_std),
            )

        if not use_base_weights:
            self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Bayesian linear layer.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, layer output
        """
        eps_weight = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * eps_weight
        if self.bias:
            eps_bias = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * eps_bias
            x = F.linear(x, weight, bias)
        else:
            x = F.linear(x, weight)
        return x

    def reset_parameters(self) -> None:
        """Reset the parameters of the Bayesian conv2d layer.

        Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        """
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        if self.bias is not False:
            fan_in: torch.Tensor
            fan_in, _ = init._calculate_fan_in_and_fan_out(  # noqa: SLF001
                self.weight_mu,
            )
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_mu, -bound, bound)

    @property
    def kl_divergence(self) -> torch.Tensor:
        """Computes the KL-divergence between the posterior and prior."""
        kl = torch.sum(
            _kl_divergence_gaussian(
                self.weight_mu,
                torch.log1p(torch.exp(self.weight_rho)) ** 2,
                cast("torch.Tensor", self.weight_prior_mu),
                cast("torch.Tensor", self.weight_prior_sigma) ** 2,
            ),
        )
        if self.bias:
            kl += torch.sum(
                _kl_divergence_gaussian(
                    self.bias_mu,
                    torch.log1p(torch.exp(self.bias_rho)) ** 2,
                    cast("torch.Tensor", self.bias_prior_mu),
                    cast("torch.Tensor", self.bias_prior_sigma) ** 2,
                ),
            )
        return kl


class BayesConv2d(nn.Module):
    """Implementation of a Bayesian convolutional layer.

    Attributes:
        in_channels: int, number of input channels
        out_channels: int, number of output channels
        kernel_size: int or tuple, size of the convolutional kernel
        stride: int or tuple, stride of the convolution
        padding: int or tuple, padding of the convolution
        dilation: int or tuple, dilation of the convolution
        groups: int, number of groups for grouped convolution
        bias: bool, whether to use a bias term
        weight_mu: torch.Tensor, mean of the posterior weights
        weight_rho: torch.Tensor, transformed standard deviation of the posterior weights
        weight_prior_mu: torch.Tensor, mean of the prior weights
        weight_prior_sigma: torch.Tensor, standard deviation of the prior weights
        bias_mu: torch.Tensor, mean of the posterior bias
        bias_rho: torch.Tensor, transformed standard deviation of the posterior bias
        bias_prior_mu: torch.Tensor, mean of the prior bias
        bias_prior_sigma: torch.Tensor, standard deviation of the prior bias
    """

    def __init__(
        self,
        base_layer: nn.Conv2d,
        use_base_weights: bool = False,
        posterior_std: float = 0.05,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ) -> None:
        """Initializes the Bayesian convolutional layer.

        Reparameterize the standard deviation of the posterior weights using the re-parameterization trick.

        Args:
            base_layer: The original conv2d layer to be used.
            use_base_weights: Whether to use the weights of the base layer as prior means. Default is False.
            posterior_std: float, initial standard deviation of the posterior
            prior_mean: float, mean of the prior
            prior_std: float, standard deviation of the prior
        """
        super().__init__()
        self.in_channels = base_layer.in_channels
        self.out_channels = base_layer.out_channels
        self.kernel_size = base_layer.kernel_size
        self.stride = base_layer.stride
        self.padding = base_layer.padding
        self.dilation = base_layer.dilation
        self.groups = base_layer.groups
        self.bias = base_layer.bias is not None

        # transform standard deviation for the re-parametrization trick
        rho = cast("float", _inverse_softplus(torch.tensor(posterior_std)))

        # posterior weights
        if not use_base_weights:
            self.weight_mu = nn.Parameter(
                torch.empty((self.out_channels, self.in_channels // self.groups, *self.kernel_size)),
            )
        else:
            self.weight_mu = nn.Parameter(cast("torch.Tensor", base_layer.weight.data))
        self.weight_rho = nn.Parameter(
            torch.full((self.out_channels, self.in_channels // self.groups, *self.kernel_size), rho),
        )

        # prior weights
        if not use_base_weights:
            self.register_buffer(
                "weight_prior_mu",
                torch.full(
                    (self.out_channels, self.in_channels // self.groups, *self.kernel_size),
                    prior_mean,
                ),
            )
        else:
            self.register_buffer(
                "weight_prior_mu",
                cast("torch.Tensor", base_layer.weight.data),
            )

        self.register_buffer(
            "weight_prior_sigma",
            torch.full((self.out_channels, self.in_channels // self.groups, *self.kernel_size), prior_std),
        )

        if self.bias:
            # posterior bias
            if not use_base_weights:
                self.bias_mu = nn.Parameter(torch.empty((self.out_channels,)))
            else:
                self.bias_mu = nn.Parameter(cast("torch.Tensor", base_layer.bias.data))
            self.bias_rho = nn.Parameter(torch.full((self.out_channels,), rho))

            # prior bias
            if not use_base_weights:
                self.register_buffer(
                    "bias_prior_mu",
                    torch.full((self.out_channels,), prior_mean),
                )
            else:
                self.register_buffer(
                    "bias_prior_mu",
                    cast("torch.Tensor", base_layer.bias.data),
                )
            self.register_buffer(
                "bias_prior_sigma",
                torch.full((self.out_channels,), prior_std),
            )

        if not use_base_weights:
            self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Bayesian conv2d layer.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, layer output
        """
        eps_weight = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * eps_weight
        if self.bias:
            eps_bias = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * eps_bias
            x = F.conv2d(
                x,
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            x = F.conv2d(
                x,
                weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        return x

    def reset_parameters(self) -> None:
        """Reset the parameters of the Bayesian conv2d layer.

        Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        """
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        if self.bias is not False:
            fan_in, _ = init._calculate_fan_in_and_fan_out(  # noqa: SLF001
                self.weight_mu,
            )
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias_mu, -bound, bound)

    @property
    def kl_divergence(self) -> torch.Tensor:
        """Compute the KL-divergence between the posterior and prior."""
        kl = torch.sum(
            _kl_divergence_gaussian(
                self.weight_mu,
                torch.log1p(torch.exp(self.weight_rho)) ** 2,
                cast("torch.Tensor", self.weight_prior_mu),
                cast("torch.Tensor", self.weight_prior_sigma) ** 2,
            ),
        )
        if self.bias:
            kl += torch.sum(
                _kl_divergence_gaussian(
                    self.bias_mu,
                    torch.log1p(torch.exp(self.bias_rho)) ** 2,
                    cast("torch.Tensor", self.bias_prior_mu),
                    cast("torch.Tensor", self.bias_prior_sigma) ** 2,
                ),
            )
        return kl


def _kl_divergence_gaussian(
    mu1: torch.Tensor,
    sigma21: torch.Tensor,
    mu2: torch.Tensor,
    sigma22: torch.Tensor,
) -> torch.Tensor:
    """Compute the KL-divergence between two Gaussian distributions.

    https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Examples
    Args:
        mu1: torch.Tensor, mean of the first Gaussian distribution
        sigma21: torch.Tensor, variance of the first Gaussian distribution
        mu2: torch.Tensor, mean of the second Gaussian distribution
        sigma22: torch.Tensor, variance of the second Gaussian distribution
    Returns:
        kl_div: float or numpy.ndarray shape (n_instances,), KL-divergence between the two Gaussian distributions
    """
    kl_div: torch.Tensor = 0.5 * torch.log(sigma22 / sigma21) + (sigma21 + (mu1 - mu2) ** 2) / (2 * sigma22) - 0.5
    return kl_div


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Compute the inverse softplus function.

    Args:
        x: Input tensor.

    Returns:
        Output tensor after applying the inverse softplus function.
    """
    return torch.log(torch.exp(x) - 1)


# ======================================================================================================================


class DropConnectLinear(nn.Module):
    """Custom Linear layer with DropConnect applied to weights during training.

    Attributes:
        in_features: int, number of input features.
        out_features: int, number of output features.
        p: float, probability of dropping individual weights.
        weight: torch.Tensor, weight matrix of the layer
        bias: torch.Tensor, bias of the layer

    """

    def __init__(self, base_layer: nn.Linear, p: float = 0.25) -> None:
        """Initialize a DropConnectLinear layer based on given linear base layer.

        Args:
            base_layer: nn.Linear, The original linear layer to be wrapped.
            p: float, The probability of dropping individual weights.
        """
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.p = p
        self.weight = nn.Parameter(base_layer.weight.clone().detach())
        self.bias = nn.Parameter(base_layer.bias.clone().detach()) if base_layer.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DropConnect layer.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, layer output

        """
        if self.training:
            mask = (torch.rand_like(self.weight) > self.p).float()
            weight = self.weight * mask  # Apply DropConnect
        else:
            weight = self.weight * (1 - self.p)  # Scale weights at inference time

        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        """Expose description of in- and out-features of this layer."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# ======================================================================================================================


class NormalInverseGammaLinear(nn.Module):
    """Custom Linear layer modeling the parameters of a normal-inverse-gamma-distribution.

    Attributes:
        gamma: torch.Tensor, shape (out_features, in_features), the mean of the normal distribution.
        nu: torch.Tensor, shape (out_features, in_features), parameter of the normal distribution.
        alpha: torch.Tensor, shape (out_features, in_features), parameter of the inverse-gamma distribution.
        beta: torch.Tensor, shape (out_features, in_features), parameter of the inverse-gamma distribution.
        gamma_bias: torch.Tensor, shape (out_features), the mean of the normal distribution for the bias.
        nu_bias: torch.Tensor, shape (out_features), parameter of the normal distribution for the bias.
        alpha_bias: torch.Tensor, shape (out_features), parameter of the inverse-gamma distribution for the bias.
        beta_bias: torch.Tensor, shape (out_features), parameter of the inverse-gamma distribution for the bias.
        bias: bool, whether to include bias in the layer.

    """

    def __init__(self, in_features: int, out_features: int, device: torch.device = None, *, bias: bool = True) -> None:
        """Initialize an instance of the NormalInverseGammaLinear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            device: Device to initialize the parameters on.
            bias: Whether to include bias in the layer. Defaults to True
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.empty((out_features, in_features), device=device))
        self.nu = nn.Parameter(torch.empty((out_features, in_features), device=device))
        self.alpha = nn.Parameter(torch.empty((out_features, in_features), device=device))
        self.beta = nn.Parameter(torch.empty((out_features, in_features), device=device))
        if bias:
            self.gamma_bias = nn.Parameter(torch.empty(out_features, device=device))
            self.nu_bias = nn.Parameter(torch.empty(out_features, device=device))
            self.alpha_bias = nn.Parameter(torch.empty(out_features, device=device))
            self.beta_bias = nn.Parameter(torch.empty(out_features, device=device))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of the NormalInverseGamma layer.

        Args:
            x: torch.Tensor, input data
        Returns:
            dict[str, torch.Tensor], layer output containing the parameters of the normal-inverse-gamma distribution
        """
        gamma = F.linear(x, self.gamma, self.gamma_bias)
        nu = F.softplus(F.linear(x, self.nu, self.nu_bias))
        alpha = F.softplus(F.linear(x, self.alpha, self.alpha_bias)) + 1
        beta = F.softplus(F.linear(x, self.beta, self.beta_bias))
        return {"gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta}

    def reset_parameters(self) -> None:
        """Reset the parameters of the NormalInverseGamma layer.

        Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        https://github.com/pytorch/pytorch/issues/57109.
        """
        init.kaiming_uniform_(self.gamma, a=math.sqrt(5))
        init.kaiming_uniform_(self.nu, a=math.sqrt(5))
        init.kaiming_uniform_(self.alpha, a=math.sqrt(5))
        init.kaiming_uniform_(self.beta, a=math.sqrt(5))
        if self.gamma_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.gamma)  # noqa: SLF001
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.gamma_bias, -bound, bound)
            init.uniform_(self.nu_bias, -bound, bound)
            init.uniform_(self.alpha_bias, -bound, bound)
            init.uniform_(self.beta_bias, -bound, bound)


# radial flows
class RadialFlowLayer(nn.Module):
    """Single radial flow transformation shared across all classes."""

    def __init__(self, num_classes: int, dim: int) -> None:
        """Initialize parameters for a radial flow transform."""
        super().__init__()
        self.c = num_classes
        self.dim = dim

        self.x0 = nn.Parameter(torch.zeros(self.c, self.dim))
        self.alpha_prime = nn.Parameter(torch.zeros(self.c))
        self.beta_prime = nn.Parameter(torch.zeros(self.c))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters with a small uniform init."""
        stdv = 1.0 / math.sqrt(self.dim)
        self.x0.data.uniform_(-stdv, stdv)
        self.alpha_prime.data.uniform_(-stdv, stdv)
        self.beta_prime.data.uniform_(-stdv, stdv)

    def forward(self, zc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the radial flow to latent inputs zc."""
        alpha = torch.nn.functional.softplus(self.alpha_prime)
        beta = -alpha + torch.nn.functional.softplus(self.beta_prime)

        x0 = self.x0.unsqueeze(1)
        diff = zc - x0
        r = diff.norm(dim=-1)

        h = 1.0 / (alpha.unsqueeze(1) + r)
        h_prime = -h * h
        beta_h = beta.unsqueeze(1) * h

        z_new = zc + beta_h.unsqueeze(-1) * diff

        term1 = (self.dim - 1) * torch.log1p(beta_h)
        term2 = torch.log1p(beta_h + beta.unsqueeze(1) * h_prime * r)
        log_abs_det = term1 + term2

        return z_new, log_abs_det


class BatchedRadialFlowDensity(nn.Module):
    """Radial-flow density estimator that computes P(z|c) for all classes."""

    def __init__(self, num_classes: int, dim: int, flow_length: int = 6) -> None:
        """Create a sequence of radial flow layers and base distribution."""
        super().__init__()
        self.c = num_classes
        self.dim = dim

        self.layers = nn.ModuleList(
            [RadialFlowLayer(num_classes, dim) for _ in range(flow_length)],
        )

        self.log_base_const = -0.5 * self.dim * math.log(2 * math.pi)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand input x for all classes and apply flow layers."""
        B = x.size(0)  # noqa: N806
        zc = x.unsqueeze(0).expand(self.c, B, self.dim)
        sum_log_jac = torch.zeros(self.c, B, device=x.device)

        for layer in self.layers:
            zc, log_j = layer(zc)
            sum_log_jac = sum_log_jac + log_j

        return zc, sum_log_jac

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return class-conditional log densities log P(x|c)."""
        zc, sum_log_jac = self.forward(x)  # zc: [C,B,D]

        base_logp = self.log_base_const - 0.5 * (zc**2).sum(dim=-1)
        logp = base_logp + sum_log_jac  # [C,B]

        return logp.transpose(0, 1)  # [B,C]


class ConvDPN(nn.Module):
    """Convolutional Dirichlet Prior Network producing concentration parameters (alpha)."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize the ConvDPN model with the given number of output classes."""
        super().__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Fully-connected classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Dirichlet concentration parameters (alpha)(x) > 0."""
        x = self.features(x)
        logits = self.classifier(x)
        alpha = F.softplus(logits) + 1e-3
        return alpha


class GrayscaleMNISTCNN(nn.Module):
    """Simple Evidential CNN for grayscale MNIST images.
    Returns Dirichlet parameters (alpha) as output.
    """  # noqa: D205

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize the CNN."""
        super().__init__()
        # (batch, 1, 28, 28) -> (batch, 32, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        # After 3 pooling layers: 28 -> 14 -> 7 -> 3 (with padding)
        # Actual: 28 -> 14 -> 7 -> 3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 28 -> 14

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # 14 -> 7

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # 7 -> 3

        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Turn outputs into alpha values for evidential learning
        x = self.relu(x)
        x = x + torch.ones_like(x)

        return x


class RadialFlowLayer2(nn.Module):
    """Single radial flow layer for a latent vector z ∈ R^D."""

    def __init__(self, dim: int) -> None:  # noqa: D107
        super().__init__()
        self.dim = dim

        # Learnable parameters:
        # - x0: center of the radial transformation (vector in R^D)
        # - alpha_prime, beta_prime: unconstrained scalars that we transform to valid alpha, beta
        self.x0 = nn.Parameter(torch.zeros(dim))
        self.alpha_prime = nn.Parameter(torch.zeros(1))
        self.beta_prime = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the radial flow to latent inputs z.

        Args:
            z: Tensor of shape [B, D].

        Returns:
            z_new: Transformed latent tensor, shape [B, D].
            log_abs_det: Log-absolute determinant of the Jacobian, shape [B].
        """
        # Ensure alpha > 0 and beta > -alpha for invertibility
        alpha = torch.nn.functional.softplus(self.alpha_prime)  # scalar > 0
        beta = -alpha + torch.nn.functional.softplus(self.beta_prime)  # scalar > -alpha

        # z0 is the learnable center (broadcast to [B, D])
        x0 = self.x0  # [D]

        # Difference from the center
        diff = z - x0  # [B, D]
        r = diff.norm(dim=-1)  # Distance to center, shape [B]

        # Radial flow scalar functions h(r) and h'(r)
        h = 1.0 / (alpha + r)  # [B]
        h_prime = -h * h  # [B]
        beta_h = beta * h  # [B]

        # Apply the radial flow transformation:
        z_new = z + beta_h.unsqueeze(-1) * diff  # [B, D]

        # Log determinant of the Jacobian:
        # formula derived in Rezende & Mohamed (2015)
        term1 = (self.dim - 1) * torch.log1p(beta_h)  # [B]
        term2 = torch.log1p(beta_h + beta * h_prime * r)  # [B]
        log_abs_det = term1 + term2  # [B]

        return z_new, log_abs_det


class Encoder(nn.Module):
    """Simple encoder mapping MNIST images to a low-dimensional latent vector z."""

    def __init__(self, latent_dim: int = 2) -> None:  # noqa: D107
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # [B, 1, 28, 28] -> [B, 784]
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),  # [B, latent_dim]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into latent vectors z.

        Args:
            x: Tensor of shape [B, 1, 28, 28].

        Returns:
            z: Tensor of shape [B, latent_dim].
        """
        return self.net(x)


class RadialFlowDensity(nn.Module):
    """Normalizing flow density p(z) using a stack of radial flows."""

    def __init__(self, dim: int, flow_length: int = 4) -> None:  # noqa: D107
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([RadialFlowLayer2(dim=dim) for _ in range(flow_length)])

        # Constant term for log N(z|0, I): -0.5 * D * log(2π)
        self.log_base_const = -0.5 * self.dim * math.log(2 * math.pi)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply all flow layers to x.

        Args:
            x: Tensor of shape [B, D].

        Returns:
            z: Transformed latent tensor after all flows, shape [B, D].
            sum_log_jac: Summed log-det Jacobian across flows, shape [B].
        """
        z = x
        sum_log_jac = torch.zeros(z.size(0), device=z.device)

        for layer in self.layers:
            z, log_j = layer(z)
            sum_log_jac = sum_log_jac + log_j

        return z, sum_log_jac

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log p(x) under the flow-based density.

        Args:
            x: Tensor of shape [B, D].

        Returns:
            logp: Log-density log p(x), shape [B].
        """
        # Apply flow
        z, sum_log_jac = self.forward(x)

        # Base log-prob under N(0, I): -0.5 * (D * log(2π) + ||z||^2)
        base_logp = self.log_base_const - 0.5 * (z**2).sum(dim=-1)  # [B]

        # Add the log-determinant of the Jacobian
        logp = base_logp + sum_log_jac  # [B]
        return logp


class NatPNClassifier(nn.Module):
    """Natural Posterior Network for classification with a Dirichlet posterior over class probabilities."""

    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 2,
        flow_length: int = 4,
        certainty_budget: float | None = None,
        n_prior: float | None = None,
    ) -> None:
        """Initialize the NatPN classifier and its components."""
        super().__init__()

        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # 1. Encoder: x -> z
        self.encoder = Encoder(latent_dim=latent_dim)

        # 2. Decoder: z -> logits for each class
        self.classifier = nn.Linear(latent_dim, num_classes)

        # 3. Single normalizing flow density over z
        self.flow = RadialFlowDensity(dim=latent_dim, flow_length=flow_length)

        # 4. Certainty budget N_H: scales the density into "evidence"
        #    Intuition: total evidence mass to distribute over the latent space.
        if certainty_budget is None:
            certainty_budget = float(latent_dim)
        self.certainty_budget = certainty_budget

        # 5. Prior pseudo-count n_prior and prior χ_prior
        if n_prior is None:
            n_prior = float(num_classes)

        # χ_prior: uniform over classes
        chi_prior = torch.full((num_classes,), 1.0 / num_classes)  # [C]
        alpha_prior = n_prior * chi_prior  # [C] -> Dirichlet(1,1,...,1)

        # Register as buffer so it is moved automatically with model.to(device)
        self.register_buffer("alpha_prior", alpha_prior)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input batch, shape [B, 1, 28, 28] for MNIST.

        Returns:
            alpha: Posterior Dirichlet parameters, shape [B, C].
            z: Latent representation, shape [B, latent_dim].
            log_pz: Log-density log p(z) under the flow, shape [B].
        """
        # Encode to latent space
        z = self.encoder(x)  # [B, latent_dim]

        # Class logits -> per-class χ (like normalized preferences)
        logits = self.classifier(z)  # [B, C]
        chi = torch.softmax(logits, dim=-1)  # [B, C], sums to 1

        # Flow density over z -> log p(z)
        log_pz = self.flow.log_prob(z)  # [B]

        # Convert density into scalar evidence n(x)
        # n(x) = N_H * exp(log p(z)) = N_H * p(z)
        n = self.certainty_budget * log_pz.exp()  # [B], evidence ≥ 0
        n = torch.clamp(n, min=1e-8)  # avoid exact zero for numerical stability

        # Evidence per class: n_i * χ_i  -> pseudo-counts
        evidence = n.unsqueeze(-1) * chi  # [B, C]

        # Posterior Dirichlet parameters: alpha = alpha_prior + evidence
        alpha = self.alpha_prior.unsqueeze(0) + evidence  # [B, C]

        return alpha, z, log_pz
