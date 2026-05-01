"""Evidential regression method compatibility exports."""

from __future__ import annotations

from probly.transformation.normal_inverse_gamma_head import normal_inverse_gamma_head, register
from probly.transformation.normal_inverse_gamma_head._common import normal_inverse_gamma_head_traverser

evidential_regression = normal_inverse_gamma_head
evidential_regression_traverser = normal_inverse_gamma_head_traverser


__all__ = ["evidential_regression", "evidential_regression_traverser", "register"]
