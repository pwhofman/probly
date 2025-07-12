"""Ensemble class implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.representation import subensemble

if TYPE_CHECKING:
    from torch import nn

reset_traverser = subensemble.reset_traverser


class Ensemble[In, KwIn](subensemble.SubEnsemble[In, KwIn]):
    """Implementation of an ensemble representation to be used for uncertainty quantification.

    Attributes:
        models: torch.nn.ModuleList, the list of models in the ensemble based on the base model.

    """

    def __init__(self, base: nn.Module, n_members: int) -> None:
        """Initialization of an instance of the Ensemble class.

        Ensemble members are constructed based on copies of the base model.

        Args:
            base: torch.nn.Module, the base model to be used.
            n_members: int, the number of members in the ensemble.
        """
        super().__init__(None, n_members, base)
