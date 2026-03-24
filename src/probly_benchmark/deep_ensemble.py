"""Benchmarking of the deep ensemble method."""

from __future__ import annotations

import torch
from torch import nn

from probly.methods import DeepEnsemble

base_model = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
ensemble_model = DeepEnsemble(base_model, num_members=3)

# create some dummy input data
input_data = torch.tensor([[1.0], [2.0], [3.0]])

# predict using the deep ensemble
predictions = ensemble_model.predict(input_data)
