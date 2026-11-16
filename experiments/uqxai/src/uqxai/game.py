"""The cooperative game of an image: superpixels are players, model outputs are (vector) values.

One forward pass of the frozen encoder per coalition serves every output that is explained later
(each class, each head, each uncertainty measure), because all of them are cheap functions of the
encoder's features. :class:`CoalitionCache` therefore stores a *payload* per evaluated coalition, and
scalar games are read from it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torchvision.transforms.v2 import functional as TF

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

Payload = dict[str, np.ndarray]


class SuperpixelMasker:
    """A square grid of superpixels over an image; absent superpixels show a baseline image.

    Attributes:
        image: The preprocessed image of shape ``(3, height, width)``.
        patch_size: Side length of a superpixel in pixels. Superpixels at the right and bottom
            border are smaller if the image size is not a multiple of it.
        grid_shape: Number of superpixel rows and columns.
        n_players: Number of superpixels.
        cells: Index of the superpixel of each pixel, of shape ``(height, width)``.
        baseline: The image shown in place of absent superpixels, of shape ``(3, height, width)``.
    """

    def __init__(self, image: torch.Tensor, patch_size: int, baseline: str = "mean") -> None:
        """Initialize the masker.

        Args:
            image: The preprocessed (normalized) image of shape ``(3, height, width)``.
            patch_size: Side length of a superpixel in pixels.
            baseline: ``"mean"`` replaces absent superpixels with the mean color of the training
                data (zero after normalization); ``"blur"`` with a strongly blurred copy of the image.

        Raises:
            ValueError: If ``baseline`` is not known.
        """
        self.image = image
        self.patch_size = patch_size
        height, width = image.shape[-2:]
        rows, columns = -(-height // patch_size), -(-width // patch_size)
        self.grid_shape = (rows, columns)
        self.n_players = rows * columns
        row_index = torch.arange(height) // patch_size
        column_index = torch.arange(width) // patch_size
        self.cells = row_index[:, None] * columns + column_index[None, :]
        match baseline:
            case "mean":
                self.baseline = torch.zeros_like(image)
            case "blur":
                self.baseline = TF.gaussian_blur(image, kernel_size=[51, 51], sigma=[25.0, 25.0])
            case _:
                msg = f"Unknown baseline {baseline!r}; use 'mean' or 'blur'."
                raise ValueError(msg)

    def __call__(self, coalitions: np.ndarray) -> torch.Tensor:
        """Return the masked images of coalitions.

        Args:
            coalitions: Boolean matrix of shape ``(m, n_players)``; ``True`` marks present superpixels.

        Returns:
            The masked images of shape ``(m, 3, height, width)``.
        """
        present = torch.from_numpy(np.asarray(coalitions, dtype=bool))[:, self.cells]  # (m, height, width)
        return torch.where(present[:, None], self.image, self.baseline)


class CoalitionCache:
    """Evaluates coalitions at most once and stores a payload of model outputs for each.

    Attributes:
        n_players: Number of players.
        coalitions: The evaluated coalitions, a boolean matrix of shape ``(m, n_players)``.
        payload: The stored outputs; every array has ``m`` rows.
    """

    def __init__(
        self,
        n_players: int,
        evaluate: Callable[[np.ndarray], Payload] | None = None,
        batch_size: int = 64,
    ) -> None:
        """Initialize an empty cache.

        Args:
            n_players: Number of players.
            evaluate: Computes the payload of a boolean coalition matrix. ``None`` makes the cache
                read-only: asking for a coalition that is not stored raises ``KeyError``.
            batch_size: Number of coalitions per call of ``evaluate``.
        """
        self.n_players = n_players
        self._evaluate = evaluate
        self._batch_size = batch_size
        self._index: dict[bytes, int] = {}
        self._coalitions: list[np.ndarray] = []
        self._payload: dict[str, list[np.ndarray]] = {}

    def __len__(self) -> int:
        """Return the number of stored coalitions."""
        return len(self._index)

    @property
    def coalitions(self) -> np.ndarray:
        """The stored coalitions as a boolean matrix."""
        return np.concatenate(self._coalitions) if self._coalitions else np.zeros((0, self.n_players), dtype=bool)

    @property
    def payload(self) -> Payload:
        """The stored outputs, one row per stored coalition."""
        return {name: np.concatenate(chunks) for name, chunks in self._payload.items()}

    def _store(self, coalitions: np.ndarray, payload: Payload) -> None:
        for row in coalitions:
            self._index[np.packbits(row).tobytes()] = len(self._index)
        self._coalitions.append(coalitions)
        for name, values in payload.items():
            self._payload.setdefault(name, []).append(np.asarray(values))

    def __call__(self, coalitions: np.ndarray) -> Payload:
        """Return the payload of coalitions, evaluating those that are not stored yet.

        Args:
            coalitions: Boolean matrix of shape ``(m, n_players)``.

        Returns:
            The outputs of the coalitions; every array has ``m`` rows, in the order of ``coalitions``.

        Raises:
            KeyError: If a coalition is not stored and the cache is read-only.
        """
        coalitions = np.asarray(coalitions, dtype=bool)
        keys = [np.packbits(row).tobytes() for row in coalitions]
        missing = {key: row for key, row in zip(keys, coalitions, strict=True) if key not in self._index}
        if missing and self._evaluate is None:
            msg = f"{len(missing)} coalitions are not in the read-only cache."
            raise KeyError(msg)
        new = np.stack(list(missing.values())) if missing else np.zeros((0, self.n_players), dtype=bool)
        for start in range(0, len(new), self._batch_size):
            batch = new[start : start + self._batch_size]
            self._store(batch, self._evaluate(batch))  # ty: ignore[call-non-callable]
        rows = np.array([self._index[key] for key in keys], dtype=int)
        return {name: values[rows] for name, values in self.payload.items()}

    def save(self, path: Path) -> None:
        """Write the coalitions (bit-packed) and their payload to a compressed ``.npz`` file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            n_players=self.n_players,
            coalitions=np.packbits(self.coalitions, axis=1),
            **{f"payload_{name}": values for name, values in self.payload.items()},
        )

    @classmethod
    def load(cls, path: Path, evaluate: Callable[[np.ndarray], Payload] | None = None) -> CoalitionCache:
        """Read a cache written by :meth:`save`.

        Args:
            path: The ``.npz`` file.
            evaluate: Computes the payload of new coalitions; ``None`` makes the cache read-only.

        Returns:
            The cache with the stored coalitions and payload.
        """
        with np.load(path) as stored:
            cache = cls(int(stored["n_players"]), evaluate)
            coalitions = np.unpackbits(stored["coalitions"], axis=1, count=cache.n_players).astype(bool)
            payload = {name.removeprefix("payload_"): stored[name] for name in stored if name.startswith("payload_")}
        if len(coalitions):
            cache._store(coalitions, payload)
        return cache
