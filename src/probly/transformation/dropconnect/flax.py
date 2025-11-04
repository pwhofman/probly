from __future__ import annotations  # noqa: D100

from flax import linen as nn

from probly.layers.flax import DropConnectDense

from .common import register


def replace_flax_dropconnect_dense(obj: nn.Dense, p: float) -> DropConnectDense:
    """Function to replace flax dropconnect-layer."""
    return DropConnectDense.from_dense(obj, p=p)


register(nn.Dense, replace_flax_dropconnect_dense)
