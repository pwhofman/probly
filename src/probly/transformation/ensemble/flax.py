"""flax ensemble implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

#from flax.nnx import Linear, Sequential 

from probly.traverse_nn import nn_compose
from pytraverse import CLONE, lazydispatch_traverser, traverse 

from .common import register 

