"""Credal set plotting (binary interval, ternary simplex, and spider/radar)."""

import contextlib

from .plot import plot_credal_set

with contextlib.suppress(ImportError):
    from . import _torch  # noqa: F401

__all__ = ["plot_credal_set"]
