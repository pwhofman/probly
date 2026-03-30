"""Benchmark for credal ensembling."""

from __future__ import annotations

import torch

from probly.method.credal_ensembling import credal_ensembling
from probly.representer.representer import representer
from probly_benchmark.models import LeNet

model = LeNet(n_classes=5)
cep = credal_ensembling(model, num_members=10)
rep = representer(cep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
print(output)
