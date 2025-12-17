"""
=========================
Sphinx-Gallery smoke test
=========================

This page exists mainly to verify that Sphinx-Gallery is correctly configured for
the project.

The docs configuration currently sets ``plot_gallery = False``, so examples are
*not* executed during documentation builds (avoids pulling in optional heavy
dependencies). You can still use this format to write narrative tutorials with
code blocks.
"""

from __future__ import annotations

import probly

print(f"probly version: {probly.__version__}")

