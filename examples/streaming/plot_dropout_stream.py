"""=========================================
MC-Dropout uncertainty on a 2-D stream
=========================================

A tiny PyTorch MLP trained one sample at a time on a 2-D classification
stream. Wrapping the network with :func:`~probly.method.dropout.dropout`
makes dropout layers active during inference, so a single
:func:`~probly.representer.representer` + :func:`~probly.quantification.quantify`
call gives an MC-Dropout uncertainty decomposition on every step.

Halfway through the run we *swap the class means* of the data
distribution. Epistemic uncertainty rises immediately after the swap
because the network's stochastic forward passes start to disagree on
the unfamiliar inputs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from probly.method.dropout import dropout
from probly.quantification import quantify
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)
from probly.representer import representer

torch.manual_seed(0)
rng = np.random.default_rng(0)

# %%
# Define a tiny network that returns a categorical distribution.

class TinyNet(nn.Module):
    """Two-layer MLP for binary classification."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> TorchCategoricalDistribution:
        return TorchProbabilityCategoricalDistribution(torch.softmax(self.net(x), dim=-1))


base = TinyNet()
mc_model = dropout(base, p=0.3)
opt = torch.optim.Adam(base.parameters(), lr=5e-2)

# %%
# Build a 2-D classification stream that swaps its class means at ``t = 300``.

N_STEPS = 600
DRIFT_T = 300


def stream_step(t: int) -> tuple[np.ndarray, int]:
    """Return one sample, with class means swapped after ``DRIFT_T``."""
    cls = int(rng.integers(0, 2))
    if t < DRIFT_T:
        mu = np.array([0.0, 0.0]) if cls == 0 else np.array([2.0, 2.0])
    else:
        mu = np.array([2.0, 2.0]) if cls == 0 else np.array([0.0, 0.0])
    return rng.normal(mu, 0.5).astype(np.float32), cls


# %%
# Test-then-train loop. ``representer(mc_model, num_samples=10)`` runs
# 10 stochastic forward passes per step.

epi = np.zeros(N_STEPS)
alea = np.zeros(N_STEPS)
sampler = representer(mc_model, num_samples=10)

for t in range(N_STEPS):
    x_np, y = stream_step(t)
    x = torch.from_numpy(x_np).unsqueeze(0)

    with torch.no_grad():
        decomp = quantify(sampler.represent(x))
    epi[t] = float(decomp.epistemic)
    alea[t] = float(decomp.aleatoric)

    base.train()
    logits = base.net(x)
    loss = nn.functional.cross_entropy(logits, torch.tensor([y]))
    opt.zero_grad()
    loss.backward()
    opt.step()

# %%
# Plot epistemic and aleatoric uncertainty over time.

window = 15
kernel = np.ones(window) / window
epi_s = np.convolve(epi, kernel, mode="same")
alea_s = np.convolve(alea, kernel, mode="same")

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(alea_s, label="aleatoric", color="#1f77b4", lw=1.2)
ax.plot(epi_s, label="epistemic", color="#d62728", lw=1.4)
ax.axvline(DRIFT_T, color="black", ls="--", lw=0.8, alpha=0.5, label="class swap")
ax.set_xlabel("step t")
ax.set_ylabel("uncertainty")
ax.set_title("MC-Dropout MLP on a 2-D stream with class swap")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
plt.show()
