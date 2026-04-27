"""Active learning pool with backend dispatch for NumPy and PyTorch."""

from probly.lazy_types import TORCH_TENSOR

from . import array as array
from ._common import ActiveLearningPool, from_dataset, query


@from_dataset.delayed_register(TORCH_TENSOR)
@query.delayed_register("probly.evaluation.active_learning.pool.torch.TorchActiveLearningPool")
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "ActiveLearningPool",
    "NumpyActiveLearningPool",
    "TorchActiveLearningPool",
    "from_dataset",
    "query",
]
