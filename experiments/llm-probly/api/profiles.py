"""Find the saved D4 profile for a model.

The profile format and ``load_profile`` / ``confidence`` come from
``selector.profile`` (D4). This module only decides which file to read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from selector.profile import PROFILES_ROOT, Profile, load_profile


def profiles_dir() -> Path:
    """Return the folder with the profile files.

    Returns:
        ``$PROFILES_DIR`` if set, otherwise ``selector.profile.PROFILES_ROOT``.
    """
    return Path(os.environ.get("PROFILES_DIR", PROFILES_ROOT))


def find_profile(model: str) -> Profile | None:
    """Find the profile of a model by short name or Hugging Face id.

    Args:
        model: Short name (``"qwen1.5b"``, the file name in ``profiles/``) or
            Hugging Face id (``"Qwen/Qwen2.5-1.5B-Instruct"``).

    Returns:
        The profile, or None if no profile matches.

    Raises:
        ValueError: If the profile file that matches by name is invalid.
    """
    directory = profiles_dir()
    if Path(model).name == model:  # plain name, no path parts
        direct = directory / f"{model}.json"
        if direct.is_file():
            return load_profile(direct)
    for path in sorted(directory.glob("*.json")):
        try:
            profile = load_profile(path)
        except (ValueError, json.JSONDecodeError):
            continue  # an invalid file must not hide the valid ones
        if profile.model_id == model:
            return profile
    return None
