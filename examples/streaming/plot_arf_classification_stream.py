"""=================================================
Streaming uncertainty with ARFClassifier
=================================================

``probly`` works with online learners out of the box. Here we train a
:class:`~river.forest.ARFClassifier` on a synthetic Agrawal stream with a
sudden change in the labelling rule at ``t = 750`` and read the full
total / aleatoric / epistemic decomposition on every step with a single
:func:`~probly.representer.representer` + :func:`~probly.quantification.quantify`
call.
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
from river.datasets import synth
from river.forest import ARFClassifier

from probly.quantification import quantify
from probly.representer import representer

# %%
# Build a stream with an abrupt concept change at ``t = 750``.

DRIFT_T = 750
N_STEPS = 1500

pre = synth.Agrawal(classification_function=7, seed=0)
post = synth.Agrawal(classification_function=4, seed=1)
stream = itertools.chain(
    itertools.islice(iter(pre), DRIFT_T),
    itertools.islice(iter(post), N_STEPS - DRIFT_T),
)

# %%
# Train an Adaptive Random Forest online and quantify on every step.

arf = ARFClassifier(n_models=10, seed=0)

total = np.zeros(N_STEPS)
aleatoric = np.zeros(N_STEPS)
epistemic = np.zeros(N_STEPS)

for t, (x, y) in enumerate(stream):
    decomp = quantify(representer(arf).represent(x))
    total[t] = float(decomp.total)
    aleatoric[t] = float(decomp.aleatoric)
    epistemic[t] = float(decomp.epistemic)
    arf.learn_one(x, y)

# %%
# Plot the decomposition. The epistemic component spikes right after the
# concept change, when the ensemble's trees disagree on the new rule.

window = 20


def smooth(a: np.ndarray) -> np.ndarray:
    """Centred rolling mean for visual clarity."""
    kernel = np.ones(window) / window
    return np.convolve(a, kernel, mode="same")


fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(smooth(total), label="total", color="#444444", lw=1.2)
ax.plot(smooth(aleatoric), label="aleatoric", color="#1f77b4", lw=1.2)
ax.plot(smooth(epistemic), label="epistemic", color="#d62728", lw=1.4)
ax.axvline(DRIFT_T, color="black", ls="--", lw=0.8, alpha=0.5, label="concept change")
ax.set_xlabel("step t")
ax.set_ylabel("uncertainty")
ax.set_title("ARF on Agrawal: uncertainty decomposition over time")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
plt.show()
