"""Benchmark for credal ensembling."""

from __future__ import annotations

import torch

from probly.method.credal_ensembling import credal_ensembling
from probly.representer.credal_ensembler._common import CredalEnsemblingRepresenter
from probly_benchmark.models import LeNet

model = LeNet(n_classes=5)
ce = credal_ensembling(model, num_members=10)
rep = CredalEnsemblingRepresenter(ce, alpha=1.0)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
