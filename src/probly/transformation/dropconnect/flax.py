from __future__ import annotations
from flax import linen as nn
from .common import register
from probly.layers.flax import DropConnectDense

def replace_flax_dropconnect_dense(obj: nn.Dense, p: float) -> DropConnectDense:
    return DropConnectDense.from_dense(obj, p=p)

register(nn.Dense, replace_flax_dropconnect_dense)