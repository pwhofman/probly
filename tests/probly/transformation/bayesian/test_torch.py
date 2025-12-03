from __future__ import annotations

import pytest

from probly.layers.torch import BayesConv2d, BayesLinear
from probly.transformation import bayesian
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")
from torch import nn


def test_bayesian_torch(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Test Bayesian transformation on a PyTorch predictor."""
    original_model = torch_model_small_2d_2d  # Original PyTorch model.
    transformed_model = bayesian(original_model)  # Apply Bayesian transformation.

    # Zählt layer im Originalmodell
    lin_before = count_layers(original_model, nn.Linear)
    bayes_lin_before = count_layers(original_model, BayesLinear)
    seq_before = count_layers(original_model, nn.Sequential)

    # Zählt layer im transformierten Modell
    lin_after = count_layers(transformed_model, nn.Linear)
    bayes_lin_after = count_layers(transformed_model, BayesLinear)
    seq_after = count_layers(transformed_model, nn.Sequential)

    # Transformiertes model soll existieren und den selben Typ haben wie das Originalmodell
    assert transformed_model is not None
    assert isinstance(transformed_model, type(original_model))

    assert bayes_lin_after == lin_before + bayes_lin_before  # Anzahl der BayesLinear Layer soll gestiegen sein
    assert lin_after == 0  # Es sollten keine Linear Layer mehr vorhanden sein
    assert seq_after == seq_before  # Anzahl der Sequential Layer soll gleich bleiben


def test_conv_to_bayesian(torch_conv_linear_model: nn.Sequential) -> None:
    """Test Bayesian transformation on a PyTorch model with Conv2d layers."""
    original_model = torch_conv_linear_model  # Original PyTorch model with Conv2d layers.
    transformed_model = bayesian(original_model)  # Apply Bayesian transformation.

    # Zählt layer im Originalmodell
    lin_before = count_layers(original_model, nn.Linear)
    conv_before = count_layers(original_model, nn.Conv2d)
    bayes_lin_before = count_layers(original_model, BayesLinear)
    bayes_conv_before = count_layers(original_model, BayesConv2d)
    seq_before = count_layers(original_model, nn.Sequential)

    # Zählt layer im transformierten Modell
    lin_after = count_layers(transformed_model, nn.Linear)
    conv_after = count_layers(transformed_model, nn.Conv2d)
    bayes_lin_after = count_layers(transformed_model, BayesLinear)
    bayes_conv_after = count_layers(transformed_model, BayesConv2d)
    seq_after = count_layers(transformed_model, nn.Sequential)

    # Transformiertes model soll den selben Typ haben wie das Originalmodell
    assert isinstance(transformed_model, type(original_model))

    assert lin_after == 0  # Es sollten keine Linear Layer mehr vorhanden sein
    assert conv_after == 0  # Es sollten keine Conv2d Layer mehr vorhanden sein

    # Anzahl der Bayes Varianten soll gestiegen sein
    assert bayes_lin_after == lin_before + bayes_lin_before
    assert bayes_conv_after == conv_before + bayes_conv_before

    assert seq_after == seq_before  # Anzahl der Sequential Layer soll gleich bleiben
