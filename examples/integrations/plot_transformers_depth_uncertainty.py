"""====================================================
Depth estimation uncertainty with transformers
====================================================

The transformers integration also covers dense prediction heads: depth
estimation models return their prediction under ``predicted_depth``, which
probly unwraps like classifier logits.

This example trains a small ensemble of tiny GLPN depth models on synthetic
scenes (a tilted ground plane with a near box, imaged by inverse-depth
shading) and shows the classic result that per-pixel ensemble uncertainty
concentrates at depth discontinuities, where the members disagree about the
exact object boundary.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from transformers import GLPNConfig, GLPNForDepthEstimation

from probly.method import ensemble
from probly.quantification import quantify
from probly.representer import representer

torch.manual_seed(0)

# %%
# Data
# ----
#
# Scenes are tilted planes with a raised box; the image brightness encodes
# inverse depth, so depth is recoverable except for noise.


def make_scenes(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.linspace(0, 1, 32).view(1, 32, 1)
    depth = 2.0 + 3.0 * ys * torch.rand(n, 1, 1) + 1.0
    depth = depth.expand(n, 32, 32).clone()
    for i in range(n):
        row, col = torch.randint(4, 20, (2,))
        height, width = torch.randint(6, 12, (2,))
        depth[i, row : row + height, col : col + width] = 1.0
    images = (1.0 / depth).unsqueeze(1).expand(n, 3, 32, 32).clone()
    images = images + torch.randn_like(images) * 0.02
    return images, depth


X_train, depth_train = make_scenes(200)
X_test, depth_test = make_scenes(50)

# %%
# Model
# -----
#
# An ensemble of tiny GLPN depth models, independently initialized and trained.

config = GLPNConfig(
    num_channels=3,
    num_encoder_blocks=2,
    depths=[1, 1],
    sr_ratios=[2, 1],
    hidden_sizes=[8, 16],
    patch_sizes=[7, 3],
    strides=[2, 2],
    num_attention_heads=[1, 2],
    mlp_ratios=[2, 2],
    decoder_hidden_size=16,
)

ensemble_model = ensemble(GLPNForDepthEstimation(config), num_members=3)

target = F.interpolate(depth_train.unsqueeze(1), size=(64, 64), mode="bilinear").squeeze(1)
for member in ensemble_model:
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    member.train()
    for _epoch in range(60):
        opt.zero_grad()
        loss = F.l1_loss(member(X_train).predicted_depth, target)
        loss.backward()
        opt.step()
    member.eval()

# %%
# Per-pixel uncertainty
# ---------------------
#
# The ensemble representer yields a sample of depth maps; quantification turns
# it into a per-pixel uncertainty map.

rep = representer(ensemble_model)
with torch.no_grad():
    sample = rep.represent(X_test)
    mean_depth = sample.tensor.mean(dim=sample.sample_dim)
    uncertainty = quantify(sample)["total"]

# %%
# Uncertainty concentrates at depth discontinuities
# -------------------------------------------------

index = 0
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
axes[0].imshow(X_test[index, 0], cmap="gray")
axes[0].set_title("input image")
axes[1].imshow(depth_test[index], cmap="viridis")
axes[1].set_title("true depth")
axes[2].imshow(mean_depth[index], cmap="viridis")
axes[2].set_title("mean predicted depth")
axes[3].imshow(uncertainty[index], cmap="magma")
axes[3].set_title("per-pixel uncertainty")
for ax in axes:
    ax.set_xticks(())
    ax.set_yticks(())
fig.tight_layout()
plt.show()
