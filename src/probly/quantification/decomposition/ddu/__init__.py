"""DDU-specific decompositions of uncertainty."""

from probly.lazy_types import TORCH_TENSOR

from ._common import DDUDensityDecomposition, negative_log_density


@negative_log_density.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["DDUDensityDecomposition", "negative_log_density"]
