"""===================
Diagnosing a Method
===================

Every uncertainty method in probly can be put through a small diagnostic
suite to check that its uncertainty estimates are actually meaningful.
:func:`~probly.diagnostics.diagnose` takes a fitted representer and test
data and reports a verdict per diagnostic:

* ``pipeline``: the representation and quantification machinery works,
* ``uncertainty_variation``: the uncertainty estimates differ across instances,
* ``accuracy`` / ``ece``: predictive quality of the mean prediction (informational),
* ``decomposition_additivity``: total equals aleatoric plus epistemic,
* ``selective_prediction``: rejecting by uncertainty beats random rejection,
* ``ood_separation``: epistemic uncertainty is higher on out-of-distribution inputs,
* ``baseline_selective_prediction``: the method is not worse than the
  max-softmax confidence of a plain deterministic predictor.

Verdicts are relative to baselines where possible: an absolute threshold
cannot distinguish a broken method from a task that simply does not elicit
the diagnosed property.

This example diagnoses a deep ensemble and compares it against a regular
single network of the same architecture.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probly.diagnostics import diagnose
from probly.method import ensemble
from probly.representer import representer

from examples.utils.model import MLPClassifier

# %%
# Setup
# -----
#
# Two Gaussian blobs as in-distribution data and a cluster far away from
# both as out-of-distribution inputs.

X = torch.cat([torch.randn(1000, 2) + 2.0, torch.randn(1000, 2) - 2.0])
y = torch.cat([torch.zeros(1000, dtype=torch.long), torch.ones(1000, dtype=torch.long)])
ood = torch.randn(500, 2) * 0.5 + torch.tensor([9.0, -9.0])

loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)


# %%
# Train the method under test
# ---------------------------
#
# The method under test is a deep ensemble whose members are independently
# initialized and trained regular networks.


def train(model: nn.Module) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    model.train()
    for _epoch in range(30):
        for inputs, targets in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            opt.step()
    model.eval()


ensemble_model = ensemble(MLPClassifier(), num_members=5, predictor_type="logit_classifier")
for member in ensemble_model:
    train(member)

# %%
# Diagnose
# --------
#
# The method arrives trained; the suite only evaluates it on the given data.
# The baseline is the first ensemble member: a regular network of the same
# architecture trained the same way, using its max-softmax confidence.

rep = representer(ensemble_model)
with torch.no_grad():
    report = diagnose(rep, X, y, ood_inputs=ood, baseline=ensemble_model[0])

print(report)

# %%
# Individual results can be inspected by name, and ``report.passed`` is True
# when no diagnostic failed.

print(report["ood_separation"])
print("passed:", report.passed)

# %%
# A failing report
# ----------------
#
# An untrained ensemble of the same architecture shows how a report looks
# when the uncertainty estimates carry no signal.

untrained = ensemble(MLPClassifier(), num_members=5, predictor_type="logit_classifier")
untrained.eval()
with torch.no_grad():
    report_untrained = diagnose(representer(untrained), X, y, ood_inputs=ood, baseline=ensemble_model[0])

print(report_untrained)
