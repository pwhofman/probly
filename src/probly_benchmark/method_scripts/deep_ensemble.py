"""Benchmarking of the deep ensemble method."""

from __future__ import annotations

import logging

import torch
from torch import nn

from probly.method.ensemble import ensemble
from probly.representer.sampler import IterableSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Testing the deep ensemble method.")
base_model = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
ensemble_model = ensemble(base_model, num_members=3)
sampler = IterableSampler(ensemble_model)
# create some dummy input data
input_data = torch.tensor([[1.0], [2.0], [3.0]])
# predict using the deep ensemble
predictions = sampler.predict(input_data)
logger.info(predictions)
