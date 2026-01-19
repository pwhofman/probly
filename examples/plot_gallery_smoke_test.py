"""Sphinx-Gallery smoke test.

This page exists mainly to verify that Sphinx-Gallery is correctly configured for
the project.

These examples are executed during the documentation build. Keep them lightweight
and avoid optional heavy ML dependencies unless you explicitly want the docs build
to require them.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import probly

plt.figure(figsize=(4, 2))
plt.plot([0, 1, 2], [0, 1, 0])
plt.title(f"Sphinx-Gallery is running (probly {probly.__version__})")
plt.tight_layout()
