"""=====================================================
GPyTorch classification with probly
=====================================================

Train a sparse variational Gaussian process classifier with GPyTorch, turn it
into a categorical sample with :func:`~probly.representer.representer`, and
split its predictive entropy into aleatoric and epistemic parts with
:class:`~probly.quantification.SecondOrderEntropyDecomposition`.
"""

from __future__ import annotations

import gpytorch
import matplotlib.pyplot as plt
import torch

from probly.quantification import SecondOrderEntropyDecomposition
from probly.representer import representer

torch.manual_seed(0)
NUM_CLASSES = 3
POINTS_PER_CLASS = 60
NUM_INDUCING = 20
NUM_SAMPLES = 64
GRID_SIZE = 40

# %%
# Data
# ----
# Three Gaussian blobs in two dimensions.

centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]])
x_train = torch.cat([center + 0.6 * torch.randn(POINTS_PER_CLASS, 2) for center in centers])
y_train = torch.arange(NUM_CLASSES).repeat_interleave(POINTS_PER_CLASS)

# %%
# Define and train the SVGP classifier
# ------------------------------------
# One latent function per class through an independent multitask variational strategy,
# combined by a softmax likelihood. This is GPyTorch's standard multiclass recipe.


class SVGPClassifier(gpytorch.models.ApproximateGP):
    """Sparse variational GP with one latent function per class."""

    def __init__(self, inducing_points: torch.Tensor) -> None:
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([NUM_CLASSES])
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=NUM_CLASSES,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([NUM_CLASSES]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([NUM_CLASSES])),
            batch_shape=torch.Size([NUM_CLASSES]),
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


model = SVGPClassifier(x_train[torch.randperm(len(x_train))[:NUM_INDUCING]])
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=NUM_CLASSES, num_classes=NUM_CLASSES)
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(x_train))
optimizer = torch.optim.Adam([*model.parameters(), *likelihood.parameters()], lr=0.05)

model.train()
likelihood.train()
for _ in range(200):
    optimizer.zero_grad()
    loss = -mll(model(x_train), y_train)
    loss.backward()
    optimizer.step()
model.eval()
likelihood.eval()

# %%
# Represent and quantify
# ----------------------
# The representer draws latent functions from the GP posterior, pushes each through the softmax
# likelihood, and stacks the class distributions into a sample. The entropy decomposition then
# gives total, aleatoric, and epistemic uncertainty for every grid point.

grid_axis = torch.linspace(-5.0, 5.0, GRID_SIZE)
grid_x, grid_y = torch.meshgrid(grid_axis, grid_axis, indexing="xy")
grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

gp_representer = representer(model, num_samples=NUM_SAMPLES, likelihood=likelihood)
with torch.no_grad():
    sample = gp_representer.represent(grid)
decomposition = SecondOrderEntropyDecomposition(sample)

print(f"sample type: {type(sample).__name__}, mean epistemic entropy: {decomposition.epistemic.mean():.3f}")

# %%
# Plot
# ----
# Aleatoric uncertainty concentrates on the class boundaries; epistemic uncertainty grows away from the data.

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
panels = (
    (decomposition.total, "total entropy"),
    (decomposition.aleatoric, "aleatoric entropy"),
    (decomposition.epistemic, "epistemic entropy (mutual information)"),
)
for ax, (values, title) in zip(axes, panels, strict=True):
    image = ax.contourf(grid_x.numpy(), grid_y.numpy(), values.reshape(GRID_SIZE, GRID_SIZE).numpy(), levels=20)
    ax.scatter(x_train[:, 0].numpy(), x_train[:, 1].numpy(), c=y_train.numpy(), s=6, cmap="tab10")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
fig.tight_layout()
plt.show()
