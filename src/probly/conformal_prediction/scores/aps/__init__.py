"""init for aps scores."""

from probly.lazy_types import TORCH_TENSOR

from .common import aps_score_func


@aps_score_func.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch  # noqa: PLC0415


# Hier steht merh oder wengier
# if (type(probs) == TORCH_TENSOR): <-- Dies wird von dem obigen Register ersetzt
#     from . import torch
# if (type(probs) == TORCH_TENSOR): <-- Dies wird von dem Register in aps.torch.py ersetzt
#     return aps_score_torch(probs)
