"""Laplace approximation benchmark code."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probly.method.laplace import laplace
from probly.quantification import quantify
from probly.representer import representer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# laplace-torch's ``subset_of_weights='last_layer'`` requires a model whose
# final module is a ``nn.Linear``. Build a small LeNet-style classifier inline
# that satisfies that constraint (the standard ``LeNet`` ends in ``Softmax``).
model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),  # (1,28,28) -> (6,24,24)
    nn.Tanh(),
    nn.AvgPool2d(2),  # -> (6,12,12)
    nn.Conv2d(6, 16, kernel_size=5),  # -> (16,8,8)
    nn.Tanh(),
    nn.AvgPool2d(2),  # -> (16,4,4)
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120),
    nn.Tanh(),
    nn.Linear(120, 84),
    nn.Tanh(),
    nn.Linear(84, 5),
)
cep = laplace(
    model,
    pred_type="glm",
    likelihood="classification",
    subset_of_weights="last_layer",
    hessian_structure="kron",
)
# Laplace is post-hoc; fit the posterior on a small loader before using it.
fit_loader = DataLoader(
    TensorDataset(torch.randn(32, 1, 28, 28), torch.randint(0, 5, (32,))),
    batch_size=8,
)
cep.fit(fit_loader)

rep = representer(cep, num_samples=10)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
logger.info(output)
logger.info(output.shape)
quantification = quantify(output)
logger.info(quantification)
logger.info(quantification.total)  # ty:ignore[unresolved-attribute]
