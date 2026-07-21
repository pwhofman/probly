"""HuggingFace environment setup for clean experiment output."""

from __future__ import annotations

import logging
import os


def suppress_hf_noise() -> None:
    """Suppress noisy HuggingFace warnings (safetensors conversion, TP sharding)."""
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    logging.getLogger("transformers").setLevel(logging.ERROR)
