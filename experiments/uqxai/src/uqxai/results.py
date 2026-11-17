"""Reads the explanations that ``scripts/03_explain.py`` wrote, pooling its coalition samples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from uqxai import simplex

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class Explanation:
    """The attributions of one readout for one image and model.

    Attributes:
        summary: The settings and the prediction of the run.
        cells: Index of the superpixel of each pixel, of shape ``(height, width)``.
        values: Attributions of shape ``(n_features, n_classes)``, the mean over the coalition
            samples (seeds) of the run. For the class probabilities of the model, their common part
            is removed: it is zero for exact Shapley values.
        empty: The readout without any feature, of shape ``(n_classes,)``.
        full: The readout of the full image, of shape ``(n_classes,)``.
        member_probabilities: The prediction of each ensemble member, of shape ``(n_members, n_classes)``.
        noise_floor: Contrast magnitude below which a feature of ``values`` is noise. Measured from
            the first two coalition samples; a conservative bound if the run has only one.
        n_samples: Number of coalition samples that ``values`` averages.
    """

    summary: dict[str, Any]
    cells: np.ndarray
    values: np.ndarray
    empty: np.ndarray
    full: np.ndarray
    member_probabilities: np.ndarray
    noise_floor: float
    n_samples: int


def load_explanation(directory: Path, readout: str = "probability") -> Explanation:
    """Read the attributions of a readout from all coalition samples of a run.

    Args:
        directory: A result directory with ``attributions.npz`` and, for further seeds,
            ``attributions_seed<k>.npz``.
        readout: Name of the readout, e.g. ``"probability"`` or ``"mean_logit"``.

    Returns:
        The explanation, with its noise floor.
    """
    stored = np.load(directory / "attributions.npz")
    samples = [stored[f"{readout}_values"]]
    for path in sorted(directory.glob("attributions_seed*.npz")):
        with np.load(path) as further:
            if f"{readout}_values" in further:
                samples.append(further[f"{readout}_values"])
    if readout == "probability":
        samples = [simplex.contrast(sample) for sample in samples]
    values = np.mean(samples, axis=0)
    if len(samples) > 1:
        # The floor of the mean of two samples; more samples only lower the noise further.
        floor = simplex.noise_floor(samples[0], samples[1]) * np.sqrt(2 / len(samples))
    else:
        floor = simplex.robust_noise_floor(values)
    return Explanation(
        summary=json.loads((directory / "summary.json").read_text()),
        cells=stored["cells"],
        values=values,
        empty=stored[f"{readout}_empty"],
        full=stored[f"{readout}_full"],
        member_probabilities=stored["member_probabilities"],
        noise_floor=float(floor),
        n_samples=len(samples),
    )
