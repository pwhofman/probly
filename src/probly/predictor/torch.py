"""Predictor for torch models."""

from __future__ import annotations

from torch import nn

from .common import EnsemblePredictor

EnsemblePredictor.register(nn.ModuleList)
