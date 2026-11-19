"""Output root shared by every plot and table script.

Defaults to ``paper_artifacts/crv`` at the repository root so that camera-ready
renders never overwrite the submission figures one level up. ``paper_artifacts``
is gitignored, so those files have no other copy. Set ``$PAPER_ARTIFACTS`` to
render somewhere else, for example when running from a git worktree.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parents[1] / "paper_artifacts" / "crv"
PAPER_ARTIFACTS = Path(os.environ["PAPER_ARTIFACTS"]).expanduser() if os.environ.get("PAPER_ARTIFACTS") else _DEFAULT
