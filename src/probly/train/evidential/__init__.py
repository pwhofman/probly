"""Train functionality for evidential models."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from .common import unified_evidential_train  # noqa: F401

# Lazy torch backend import
if TORCH_MODULE is not None:
    from . import torch  # noqa: F401
