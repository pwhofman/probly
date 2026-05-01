"""Samplers for creating representations from predictor outputs."""

from probly.lazy_types import LAPLACE_BASE
from probly.representer._representer import representer


@representer.delayed_register(LAPLACE_BASE)
def _(_: type) -> None:
    from . import laplace as laplace  # noqa: PLC0415
