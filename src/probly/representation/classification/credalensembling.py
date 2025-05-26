"""Credal Ensembling class implementation."""

from __future__ import annotations

import torch
from torch import nn

from probly.representation import Ensemble


class CredalEnsembling(Ensemble):
    """Implementation of a credal ensembling representation to be used for uncertainty quantification.

    The Credal Ensembling representation was introduced in https://link.springer.com/article/10.1007/s10994-024-06703-y.

    Attributes:
        models: torch.nn.ModuleList, the list of models in the ensemble based on the base model.
    """

    def __init__(self, base: nn.Module, n_members: int) -> None:
        """Initialization of an instance of the Credal Ensembling class.

        Ensemble members are constructed based on copies of the base model.

        Args:
            base: torch.nn.Module, the base model to be used.
            n_members: int, the number of members in the ensemble.
        """
        super().__init__(base, n_members)

    def predict_representation(self, x: torch.Tensor, alpha: float = 0, distance: str = "euclidean") -> torch.Tensor:
        """Forward pass that gives an uncertainty representation.

        Args:
            x: torch.Tensor, input data
            alpha: float, the portion of farthest members in terms of distance to be removed
            distance: str, the distance metric to be used for the representation
                (e.g., 'euclidean', 'kl')

        Returns:
            torch.Tensor, uncertainty representation
        """
        x = super().predict_representation(x, logits=False)
        if distance == "euclidean":
            x_representative = torch.mean(x, dim=1)
            dists = torch.cdist(x, x_representative.unsqueeze(dim=1), p=2).squeeze(dim=2)
        elif distance == "kl":
            msg = "KL-divergence metric is not implemented yet."
            raise NotImplementedError(msg)
            # TODO(pwhofman): Implement KL-divergence metric https://github.com/pwhofman/probly/issues/99
        else:
            msg = f"Unknown distance metric: {distance}"
            raise ValueError(msg)
        sorted_indices = torch.argsort(dists, dim=1)
        keep_indices = sorted_indices[:, : int(round((1 - alpha) * x.shape[1]))]
        keep_indices = keep_indices.unsqueeze(dim=2).expand(-1, -1, x.shape[2])
        x = torch.gather(x, dim=1, index=keep_indices)
        return x
