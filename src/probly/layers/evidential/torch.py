"""torch evidential layer implementations."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        *,
        bias: bool = True,
    ) -> None:
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


class SimpleHead(nn.Module):
    """Simple classification head outputting class evidence."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, num_classes: int = 10) -> None:  # noqa: D107
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return F.softplus(self.net(z))


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


class EvidentialHead(nn.Module):
    """Head that converts encoded features into evidential Normal-Gamma parameters."""

    def __init__(self, latent_dim: int) -> None:
        """Initialize the head.

        Args:
            latent_dim: Dimension of input features coming from the encoder.
        """
        super().__init__()
        self.linear = nn.Linear(latent_dim, 4)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert features into (mu, kappa, alpha, beta).

        Args:
            features: Feature tensor (N, feature_dim)

        Returns:
            Tuple of four tensors representing Normal-Gamma parameters.
        """
        raw = self.linear(features)

        mu = raw[:, 0:1]
        kappa = F.softplus(raw[:, 1:2])
        alpha = F.softplus(raw[:, 2:3]) + 1.0
        beta = F.softplus(raw[:, 3:4])

        return mu, kappa, alpha, beta


class DirichletHead(nn.Module):
    """Dirichlet posterior head for evidential classification.

    Takes latent representations and outputs Dirichlet parameters for uncertainty
    quantification in classification. This head should be used with an encoder to
    create a complete classification model.

    Args:
        latent_dim: Dimension of input latent vectors from an encoder.
        num_classes: Number of classification classes.
        hidden_dim: Dimension of hidden layer. Defaults to 64.
        n_prior: Prior for evidence scaling. If None, defaults to num_classes.
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        n_prior: float | None = None,
    ) -> None:
        """Initialize the DirichletHead."""
        super().__init__()

        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        if n_prior is None:
            n_prior = float(num_classes)

        chi_prior = torch.full((num_classes,), 1.0 / num_classes)
        alpha_prior = n_prior * chi_prior

        self.register_buffer("alpha_prior", alpha_prior)

    def forward(
        self,
        z: torch.Tensor,
        log_pz: torch.Tensor,
        certainty_budget: float,
    ) -> dict[str, torch.Tensor]:
        """Compute Dirichlet parameters for evidential classification.

        Args:
            z: Latent representations of shape [B, latent_dim].
            log_pz: Log probability from density estimator of shape [B].
            certainty_budget: Budget parameter for evidence scaling.

        Returns:
            Dictionary containing:
                - alpha: Dirichlet parameters [B, num_classes]
                - z: Input latent representations
                - log_pz: Log density values
                - evidence: Scaled evidence [B, num_classes]
        """
        logits = self.classifier(z)  # [B, C]
        chi = torch.softmax(logits, dim=-1)  # [B, C]

        # Total evidence n(x)
        n = certainty_budget * log_pz.exp()  # [B]
        n = torch.clamp(n, min=1e-8)

        evidence = n.unsqueeze(-1) * chi  # [B, C]
        alpha = self.alpha_prior.unsqueeze(0) + evidence

        return {
            "alpha": alpha,  # Dirichlet parameters
            "z": z,
            "log_pz": log_pz,
            "evidence": evidence,
        }


class GaussianHead(nn.Module):
    """Gaussian posterior head for evidential regression.

    Takes latent representations and outputs mean and variance for Gaussian
    uncertainty quantification in regression. This head should be used with an encoder
    to create a complete regression model.

    Args:
        latent_dim: Dimension of input latent vectors from an encoder.
        out_dim: Dimension of regression output. Defaults to 1 (univariate regression).
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int = 1,
    ) -> None:
        """Initialize the GaussianHead."""
        super().__init__()

        self.mean_net = nn.Linear(latent_dim, out_dim)
        self.log_var_net = nn.Linear(latent_dim, out_dim)

    def forward(
        self,
        z: torch.Tensor,
        log_pz: torch.Tensor,
        certainty_budget: float,
    ) -> dict[str, torch.Tensor]:
        """Compute Gaussian parameters for evidential regression.

        Args:
            z: Latent representations of shape [B, latent_dim].
            log_pz: Log probability from density estimator of shape [B].
            certainty_budget: Budget parameter for precision scaling.

        Returns:
            Dictionary containing:
                - mean: Predicted mean [B, out_dim]
                - var: Predicted variance [B, out_dim]
                - z: Input latent representations
                - log_pz: Log density values
                - precision: Scaled precision [B, out_dim]
        """
        mean = self.mean_net(z)  # [B, D]
        log_var = self.log_var_net(z)  # [B, D]

        # Epistemic uncertainty via density scaling
        precision = certainty_budget * log_pz.exp().unsqueeze(-1)
        precision = torch.clamp(precision, min=1e-8)

        var = torch.exp(log_var) / precision

        return {
            "mean": mean,
            "var": var,
            "z": z,
            "log_pz": log_pz,
            "precision": precision,
        }


class IRDHead(nn.Module):
    """Head that converts encoded features into Dirichlet concentration parameters (alpha).

    For multi-class classification, this head outputs K alpha values (one per class),
    where alpha forms a K-dimensional Dirichlet distribution.
    """

    def __init__(self, latent_dim: int, num_classes: int) -> None:
        """Initialize the Dirichlet head.

        Args:
            latent_dim: Dimension of input features from the encoder.
            num_classes: Number of output classes (K in Dirichlet(a)).
        """
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(latent_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Convert features into Dirichlet concentration parameters (alpha).

        Args:
            features: Feature tensor (batch_size, latent_dim) from encoder.

        Returns:
            Alpha parameters of shape (batch_size, num_classes), all > 0.
        """
        # Linear projection to num_classes dimensions
        logits = self.linear(features)

        # Ensure alpha > 0 by applying softplus and adding small offset
        # alpha = softplus(logits) + 1.0 ensures all values >= 1.0
        alpha = F.softplus(logits) + 1.0

        return alpha


class SimpleDirichletHead(nn.Module):
    """Head mapping latent features to Dirichlet concentration parameters."""

    def __init__(self, latent_dim: int, num_classes: int) -> None:
        """Initialize the Dirichlet classification head."""
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Produce positive Dirichlet concentration parameters."""
        return F.softplus(self.net(z)) + 1e-3
