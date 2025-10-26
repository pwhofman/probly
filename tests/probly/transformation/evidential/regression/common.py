from __future__ import annotations

import dataclasses

from probly.transformation.evidential.regression.common import (
    register,
    evidential_regression,
    REPLACED_LAST_LINEAR,
)


# ---- Dummy-Implementierungen  ----

@dataclasses.dataclass
class DummyLinear:
    """Einfache 'Linear'-Schicht als Platzhalter."""
    in_features: int
    out_features: int

    def __call__(self, batch):
        return [[0.0] * self.out_features for _ in range(len(batch))]


@dataclasses.dataclass
class NormalInverseGammaLinearStub:
    in_features: int
    out_features: int

    def __call__(self, batch):
        return [[0.0] * self.out_features for _ in range(len(batch))]


class TinyPredictor:
    def __init__(self):
        self.lin1 = DummyLinear(8, 16)
        self.lin2 = DummyLinear(16, 1)

    def predict(self, batch):
        h = self.lin1(batch)
        return self.lin2(h)


# ---- Traverser: ersetzt einmal DummyLinear ----

def replace_linear_with_nig(node, state):
    if not state[REPLACED_LAST_LINEAR]:
        state[REPLACED_LAST_LINEAR] = True
        return NormalInverseGammaLinearStub(
            in_features=node.in_features,
            out_features=node.out_features,
        )
    return node


# Registrierung 
register(cls=DummyLinear, traverser=replace_linear_with_nig)


# ------------------ Tests ----------------------

def test_replaces_only_the_last_linear():
    base = TinyPredictor()
    evidential = evidential_regression(base)

    assert isinstance(evidential.lin2, NormalInverseGammaLinearStub)
    assert isinstance(evidential.lin1, DummyLinear)


def test_returns_a_clone_not_inplace():
    base = TinyPredictor()
    evidential = evidential_regression(base)

    assert evidential is not base
    assert evidential.lin1 is not base.lin1
    assert evidential.lin2 is not base.lin2


def test_output_shape_is_unchanged():
    base = TinyPredictor()
    evidential = evidential_regression(base)

    batch = [[1.0] * 8 for _ in range(4)]
    output = evidential.predict(batch)

    assert len(output) == 4
    assert all(len(row) == 1 for row in output)


def test_idempotent_if_applied_twice():
    base = TinyPredictor()
    once = evidential_regression(base)
    twice = evidential_regression(once)

    assert isinstance(twice.lin2, NormalInverseGammaLinearStub)
    assert isinstance(twice.lin1, DummyLinear)

