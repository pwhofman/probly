"""=====================================================
GPyTorch regression with probly
=====================================================

Train an exact Gaussian process with GPyTorch, read its predictive
distribution through :func:`~probly.predictor.predict`, quantify the
predictive variance, and wrap the model with
:func:`~probly.method.conformal.conformal_absolute_error` to obtain
intervals with a coverage guarantee.
"""

from __future__ import annotations

import gpytorch
import matplotlib.pyplot as plt
import torch
from torch import nn

from probly.calibrator import calibrate
from probly.method.conformal import conformal_absolute_error
from probly.metrics import coverage, efficiency
from probly.predictor import predict
from probly.quantification.measure.variance import variance
from probly.representer import representer

torch.manual_seed(0)
ALPHA = 0.1
NUM_TRAIN = 60
NUM_CALIB = 40
NUM_TEST = 1000
NUM_GRID = 150

# %%
# Data
# ----
# A noisy sine on [0, 1]. Training, calibration, and test points share that range, which is what the
# conformal guarantee needs. A separate grid extends to 1.5 so the variance growing away from the data is visible.


def target(x: torch.Tensor) -> torch.Tensor:
    """Noise-free target function."""
    return torch.sin(2.0 * torch.pi * x)


x_train = torch.rand(NUM_TRAIN)
y_train = target(x_train) + 0.1 * torch.randn(NUM_TRAIN)
x_calib = torch.rand(NUM_CALIB)
y_calib = target(x_calib) + 0.1 * torch.randn(NUM_CALIB)
x_test = torch.rand(NUM_TEST)
y_test = target(x_test) + 0.1 * torch.randn(NUM_TEST)
x_grid = torch.linspace(0.0, 1.5, NUM_GRID)

# %%
# Define and train the exact GP
# -----------------------------
# This is the standard GPyTorch recipe: a model, a Gaussian likelihood, and the exact marginal log-likelihood.


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP with a constant mean and a scaled RBF kernel."""

    def __init__(
        self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_train, y_train, likelihood)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

model.train()
for _ in range(100):
    optimizer.zero_grad()
    loss = -mll(model(x_train), y_train)
    loss.backward()
    optimizer.step()
model.eval()

# %%
# Predict with probly
# -------------------
# ``predict`` returns a :class:`~probly.representation.distribution.torch_gaussian.TorchGaussianDistribution`
# holding the predictive mean and variance, observation noise included.

with torch.no_grad():
    prediction = predict(model, x_grid)
    predictive_variance = variance(prediction)

print(f"prediction type: {type(prediction).__name__}")
print(f"variance at x=0.5: {predictive_variance[NUM_GRID // 3]:.3f}, at x=1.5: {predictive_variance[-1]:.3f}")

# %%
# Conformal intervals
# -------------------
# probly's absolute-error conformal wrapper expects point predictions, so a thin module exposes the
# GP mean. The wrapper then calibrates the interval half-width on held-out data. Coverage is checked
# on the in-range test set; the same intervals are then drawn on the extended grid.


class GPMean(nn.Module):
    """Point predictor that returns the GP's predictive mean."""

    def __init__(self, gp: gpytorch.models.ExactGP) -> None:
        super().__init__()
        self.gp = gp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return predict(self.gp, x).mean


with torch.no_grad():
    calibrated = calibrate(conformal_absolute_error(GPMean(model)), ALPHA, y_calib, x_calib)
    test_intervals = representer(calibrated).predict(x_test)
    grid_intervals = representer(calibrated).predict(x_grid)

print(f"coverage: {coverage(test_intervals, y_test):.3f}, average interval width: {efficiency(test_intervals):.3f}")

# %%
# Plot
# ----

mean = prediction.mean.numpy()
std = prediction.std.numpy()
x_np = x_grid.numpy()
bounds = grid_intervals.tensor.numpy()

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(x_train.numpy(), y_train.numpy(), s=12, color="black", label="training data")
ax.plot(x_np, mean, label="GP mean")
ax.fill_between(x_np, mean - 2 * std, mean + 2 * std, alpha=0.25, label="GP mean +/- 2 std")
interval_label = f"{int((1 - ALPHA) * 100)}% conformal interval"
ax.plot(x_np, bounds[:, 0], linestyle="--", color="tab:red", linewidth=1, label=interval_label)
ax.plot(x_np, bounds[:, 1], linestyle="--", color="tab:red", linewidth=1)
ax.axvline(1.0, color="gray", linewidth=0.8)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()
