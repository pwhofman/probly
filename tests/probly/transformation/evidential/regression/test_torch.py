import pytest

torch = pytest.importorskip("torch")
from torch import nn

from probly.transformation.evidential import regression as er
from probly.layers.torch import NormalInverseGammaLinear


def test_last_linear_is_replaced():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),  
    )

    new_model = er.evidential_regression(model)

    
    last = list(new_model.modules())[-1]

    
    assert isinstance(last, NormalInverseGammaLinear)


def test_model_type_stays_the_same():
    model = nn.Sequential(
        nn.Linear(3, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )

    new_model = er.evidential_regression(model)

    
    assert isinstance(new_model, type(model))
