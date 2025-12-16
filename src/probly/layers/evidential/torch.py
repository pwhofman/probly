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


class SimpleCNN(nn.Module):
    """Simple CNN."""

    def __init__(self, num_classes: int = 10) -> torch.Tensor:  # noqa: D107
        super().__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.softplus(self.fc2(x))  # use of softplus so that our output is always positive


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

        # 2. Decoder: z -> logits for each class (SMOOTHED)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

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
