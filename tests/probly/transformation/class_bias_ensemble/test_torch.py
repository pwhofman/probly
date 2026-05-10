"""Tests for the torch backend of class_bias_ensemble."""

from __future__ import annotations

import pytest


def _torch_nn():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    return torch, nn


class TestClassBiasEnsembleTraverser:
    """Traversal-side behaviour of class_bias_ensemble for torch nn.Modules.

    These tests drive the traverser via the public ``traverse_with_state`` API
    rather than calling the internal handlers directly.
    """

    def _state_init(self, *, bias_cls: int, initialized: bool, tobias_value: int = 100) -> dict:
        from probly.transformation.class_bias_ensemble import torch as _torch_register  # noqa: F401, PLC0415
        from probly.transformation.class_bias_ensemble._common import (  # noqa: PLC0415
            BIAS_CLS,
            INITIALIZED,
            TOBIAS_VALUE,
        )
        from pytraverse import TRAVERSE_REVERSED  # noqa: PLC0415

        return {
            BIAS_CLS: bias_cls,
            TOBIAS_VALUE: tobias_value,
            INITIALIZED: initialized,
            TRAVERSE_REVERSED: True,
        }

    def test_initialises_linear_bias_with_class_specific_offset(self) -> None:
        torch, nn = _torch_nn()
        from probly.transformation.class_bias_ensemble._common import (  # noqa: PLC0415
            INITIALIZED,
            class_bias_ensemble_traverser,
        )
        from probly.traverse_nn import nn_compose  # noqa: PLC0415
        from pytraverse import traverse_with_state  # noqa: PLC0415

        layer = nn.Linear(4, 3, bias=True)
        with torch.no_grad():
            layer.bias.zero_()

        result, final_state = traverse_with_state(
            layer,
            nn_compose(class_bias_ensemble_traverser),
            init=self._state_init(bias_cls=2, initialized=False, tobias_value=100),
        )
        assert final_state[INITIALIZED] is True
        # Expected: bias[(2-1) % 3] = 100  # noqa: ERA001
        assert result.bias.data[1].item() == pytest.approx(100.0)

    def test_bias_zero_class_skips_offset(self) -> None:
        torch, nn = _torch_nn()
        from probly.transformation.class_bias_ensemble._common import (  # noqa: PLC0415
            INITIALIZED,
            class_bias_ensemble_traverser,
        )
        from probly.traverse_nn import nn_compose  # noqa: PLC0415
        from pytraverse import traverse_with_state  # noqa: PLC0415

        layer = nn.Linear(4, 3, bias=True)
        with torch.no_grad():
            layer.bias.zero_()

        _, final_state = traverse_with_state(
            layer,
            nn_compose(class_bias_ensemble_traverser),
            init=self._state_init(bias_cls=0, initialized=False),
        )
        assert final_state[INITIALIZED] is True
        # No offset for member 0
        assert torch.all(layer.bias.data == 0.0)


class TestClassBiasEnsembleEnd2End:
    """class_bias_ensemble produces a list of distinct members."""

    def test_creates_list_of_members(self) -> None:
        torch, nn = _torch_nn()
        from probly.transformation.class_bias_ensemble import class_bias_ensemble  # noqa: PLC0415

        # A plain nn.Module ending in Linear is auto-registered as the
        # single permitted predictor type (LogitClassifier) by the
        # predictor_transformation decorator.
        base = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        members = class_bias_ensemble(base, num_members=3, reset_params=False, tobias_value=50)
        assert len(members) == 3
        # All three members should be distinct objects.
        assert members[0] is not members[1]
        assert members[1] is not members[2]
        # Each member's final Linear bias differs in one slot.
        biases = [m[-1].bias.data.clone() for m in members]
        # The first member (bias_cls=0) gets no offset.
        # The other members differ in different slots.
        assert not torch.equal(biases[1], biases[2])
