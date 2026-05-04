"""Uniform model wrappers for the paper experiment.

Paper §5.3 uses river ARFClassifier only; the wrapper exposes:
- learn_one(x, y) -> None
- predict_one(x) -> int (predicted class label)
- epistemic_decomposition(x) -> AleatoricEpistemicTotalDecomposition with
  .total, .aleatoric, .epistemic
"""

from __future__ import annotations

from typing import Any, Hashable, Protocol, cast

from river.forest import ARFClassifier

from probly.quantification import quantify
from probly.quantification.decomposition import AleatoricEpistemicTotalDecomposition
from probly.representer import representer

K_MEMBERS = 10


class ModelWrapper(Protocol):
    """Uniform interface for streaming UQ models."""

    def learn_one(self, x: dict[str, float], y: int) -> None: ...
    def predict_one(self, x: dict[str, float]) -> int: ...
    def epistemic_decomposition(self, x: dict[str, float]) -> AleatoricEpistemicTotalDecomposition: ...


class _ARFWrapper:
    def __init__(self, seed: int) -> None:
        self._arf = ARFClassifier(n_models=K_MEMBERS, seed=seed)
        self._seed = seed

    def learn_one(self, x: dict[str, float], y: int) -> None:
        self._arf.learn_one(x, y)

    def predict_one(self, x: dict[str, float]) -> int:
        pred = self._arf.predict_one(cast("dict[Hashable, Any]", x))
        return int(pred) if pred is not None else 0

    def epistemic_decomposition(self, x: dict[str, float]) -> AleatoricEpistemicTotalDecomposition:
        sample = representer(self._arf).represent(x)
        return cast("AleatoricEpistemicTotalDecomposition", quantify(sample))


def build_model(kind: str, seed: int) -> ModelWrapper:
    """Build a model wrapper by name.

    Args:
        kind: Only ``"arf"`` is implemented (paper uses river ARFClassifier).
        seed: Model seed (independent across (stream, seed) combos).

    Returns:
        A model exposing the ``ModelWrapper`` protocol.
    """
    if kind == "arf":
        return _ARFWrapper(seed=seed)
    msg = f"unknown model kind: {kind!r}"
    raise ValueError(msg)
