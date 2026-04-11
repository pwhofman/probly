"""Root conftest for probly tests.

Deselects test modules whose import chain pulls in optional backends that
are not installed in the current environment. This keeps the CI matrix
slim — a jax-only job does not need torch, and vice versa — without
requiring every test module to guard its imports individually.
"""

from __future__ import annotations

import importlib.util

collect_ignore_glob: list[str] = []

if importlib.util.find_spec("torch") is None:
    # Any file that unconditionally imports torch, plus the whole
    # conformal_prediction subtree whose package __init__ eagerly imports
    # torch transitively. Conformal is slated for removal/refactor per the
    # sklearn-removal PR, so skipping it here is in line with that plan.
    collect_ignore_glob += [
        "**/test_torch.py",
        "conformal_prediction",
        # These subdir conftests import torch at module top for fixtures.
        "calibration/isotonic_regression",
        "calibration/scaling",
    ]
