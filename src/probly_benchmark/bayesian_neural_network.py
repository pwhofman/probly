"""Benchmarking of the Bayesian Neural Network (BNN) model."""

from __future__ import annotations

import logging

import torch
from torch import nn

from probly.methods.bayesian_neural_network import BayesianNeuralNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# very small, simple model for testing purposes

model = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))

bnn = BayesianNeuralNetwork(
    base=model,
    num_samples=5,
    use_base_weights=True,
    posterior_std=0.1,
    prior_mean=0.0,
    prior_std=1.0,
)

x = torch.tensor([[1.0], [2.0], [3.0]])

predictions = bnn.predict(x)
logger.info(predictions)
