"""Dropout ensemble implementation for uncertainty quantification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lazy_imports import LazyModule, as_package, load

if TYPE_CHECKING:
    pass
else:
    load(
        LazyModule(
            *as_package(__file__),
        ),
    )
