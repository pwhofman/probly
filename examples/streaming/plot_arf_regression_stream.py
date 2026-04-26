"""===========================================
Streaming uncertainty with ARFRegressor
===========================================

The same recipe as the classifier example, but now with
:class:`~river.forest.ARFRegressor` on the Friedman regression stream.

For an ensemble regressor each tree returns a deterministic point
prediction, so the ensemble disagreement *is* the epistemic uncertainty
and aleatoric is zero. We plot the running prediction with a band
spanning the per-tree spread.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from river.datasets import synth
from river.forest import ARFRegressor

from probly.predictor import predict_raw

# %%
# Build a regression stream and an online ensemble.

N_STEPS = 2500

stream = iter(synth.Friedman(seed=0))
arf = ARFRegressor(n_models=10, seed=0)

# warm up the ensemble so the first per-tree predictions are non-empty
for _ in range(20):
    x, y = next(stream)
    arf.learn_one(x, y)

mean_pred = np.zeros(N_STEPS)
std_pred = np.zeros(N_STEPS)
truth = np.zeros(N_STEPS)

# %%
# On each step: read the per-tree predictions via
# :func:`~probly.predictor.predict_raw` (one short call), update the
# ensemble, and store the spread.

for t, (x, y) in enumerate(stream):
    if t >= N_STEPS:
        break
    per_tree = np.asarray([float(p[0]) for p in predict_raw(arf, x)])
    mean_pred[t] = per_tree.mean()
    std_pred[t] = per_tree.std()
    truth[t] = float(y)
    arf.learn_one(x, y)

# %%
# Plot the running prediction with the epistemic band. Early on the
# trees disagree (large band) because they have seen too few samples;
# the band tightens as the ensemble learns the function.

steps = np.arange(N_STEPS)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(steps, truth, color="#444444", lw=0.8, alpha=0.5, label="ground truth")
ax.plot(steps, mean_pred, color="#1f77b4", lw=1.2, label="ensemble mean")
ax.fill_between(
    steps,
    mean_pred - std_pred,
    mean_pred + std_pred,
    color="#d62728",
    alpha=0.25,
    label="epistemic ($\\pm$ per-tree std)",
)
ax.set_xlabel("step t")
ax.set_ylabel("y")
ax.set_title("ARFRegressor on Friedman: prediction with epistemic band")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
plt.show()
