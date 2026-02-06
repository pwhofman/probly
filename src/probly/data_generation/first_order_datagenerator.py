"""Backend/General First-Order data generator.

General implementation using pure Python constructs (no torch dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any, Protocol, cast
import warnings

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _to_batch_array(outputs: object) -> np.ndarray:
    # Normalize various output shapes into a 2D numpy array of shape (batch, classes)
    if outputs is None:
        return np.empty((0, 0), dtype=float)
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 0:
            return np.empty((0, 0), dtype=float)
        first = outputs[0]
        if isinstance(first, (int, float)):
            arr = np.asarray(outputs, dtype=float)
            return arr[np.newaxis, :]
        return np.asarray(outputs, dtype=float)
    # Fallback: scalar or unknown object -> single-sample, one element vector
    return np.array([[float(cast("Any", outputs))]], dtype=float)


def _is_probabilities_array(arr: np.ndarray, atol: float = 1e-4) -> bool:
    if arr.size == 0:
        return False
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        return False
    within_bounds = (arr >= -atol).all() and (arr <= (1.0 + atol)).all()
    if not within_bounds:
        return False
    row_sums = arr.sum(axis=1)
    return np.abs(row_sums - 1.0).max() <= atol


def _softmax_array(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    x = arr - arr.max(axis=1, keepdims=True)
    exps = np.exp(x)
    sums = exps.sum(axis=1, keepdims=True)
    out = np.zeros_like(exps)
    np.divide(exps, sums, where=(sums > 0), out=out)
    return out


@dataclass
class FirstOrderDataGenerator:
    """General backend first-order data generator."""

    model: Callable[..., Any]
    device: str = "cpu"  # Remember: kept for API symmetry; unused here !
    batch_size: int = 64
    output_mode: str = "auto"  # 'auto' | 'logits' | 'probs'
    output_transform: Callable[[Any], Any] | None = None
    input_getter: Callable[[Any], Any] | None = None
    model_name: str | None = None
    return_numpy: bool = True

    def to_probs(self, outputs: object) -> object:
        """Convert raw model outputs to probability rows."""
        if self.output_transform is not None:
            transformed = self.output_transform(outputs)
            batch_arr = _to_batch_array(transformed)
        else:
            batch_arr = _to_batch_array(outputs)

        mode = (self.output_mode or "auto").lower()

        if mode == "probs":
            probs = batch_arr
        elif mode == "logits":
            probs = _softmax_array(batch_arr)
        elif mode == "auto":
            probs = batch_arr if _is_probabilities_array(batch_arr) else _softmax_array(batch_arr)
        else:
            msg = f"Invalid output_mode '{self.output_mode}'. Expected one of: 'auto', 'logits', 'probs'."
            raise ValueError(msg)

        return probs if self.return_numpy else probs.tolist()

    def prepares_batch_inp(self, sample: object) -> object:
        """Extract the model input from a dataset sample.

        Behavior:
        - If input_getter is provided use it to obtain the input.
        - If the sample is a tuple like (input, label, ...), return the first element.
        - Otherwise, return the sample as-is.

        Notes:
        - Lists are treated as input-only feature vectors and are NOT unpacked.
        """
        if self.input_getter is not None:
            return self.input_getter(sample)
        if isinstance(sample, tuple) and len(sample) >= 1:
            return sample[0]
        return sample

    def generate_distributions(
        self,
        dataset_or_loader: object,
        *,
        progress: bool = True,
    ) -> object:
        """Generate per-sample distributions for a dataset or loader."""
        # Prepare loader
        if isinstance(dataset_or_loader, SimpleDataLoader):
            loader = dataset_or_loader
            dataset_len = len(loader.dataset) if hasattr(loader, "dataset") else None
        else:
            dataset = cast("DatasetLike", dataset_or_loader)
            dataset_len = len(dataset)
            loader = SimpleDataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        distributions: dict[int, object] = {}
        start_idx = 0
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            # Build model input for the whole batch
            # We pass the batch as a list of inputs or a single input if batch_size=1
            inputs = [self.prepares_batch_inp(sample) for sample in batch]
            # If model expects a single object consider passing the list directly
            outputs = self.model(inputs)
            probs_batch = self.to_probs(outputs)

            if isinstance(probs_batch, list):
                batch_size_local = len(probs_batch)
                for i in range(batch_size_local):
                    idx = start_idx + i
                    distributions[idx] = probs_batch[i]
                start_idx += batch_size_local
            else:
                arr = cast("NDArray", probs_batch)
                batch_size_local = arr.shape[0]
                for i in range(batch_size_local):
                    idx = start_idx + i
                    distributions[idx] = arr[i]
                start_idx += batch_size_local
            if progress:
                logger.info("[FirstOrderDataGenerator] Batch %d/%d", batch_idx + 1, total_batches)

        if progress:
            logger.info("[FirstOrderDataGenerator] Finished %d batches", total_batches)

        if dataset_len is not None and len(distributions) != dataset_len:
            warnings.warn(
                (
                    f"[FirstOrderDataGenerator] generated {len(distributions)} distributions, "
                    f"but dataset length is {dataset_len}."
                ),
                stacklevel=2,
            )

        return distributions

    def save_distributions(
        self,
        path: str | Path,
        distributions: Mapping[int, Iterable[float]],
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Save distributions and optional metadata as JSON."""
        path = Path(path)
        serializable = {
            "meta": {
                "model_name": self.model_name,
                **(meta or {}),
            },
            "distributions": {
                str(k): np.asarray(list(v) if not isinstance(v, np.ndarray) else v, dtype=float).tolist()
                for k, v in distributions.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)

    def load_distributions(
        self, path: str | Path, *, return_numpy: bool = True
    ) -> tuple[dict[int, Any], dict[str, Any]]:
        """Load distributions and metadata from JSON.

        When return_numpy is True (default), returns numpy arrays for each
        distribution row. If False, returns plain Python lists of floats.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        meta = obj.get("meta", {}) or {}
        dists_raw = obj.get("distributions", {}) or {}
        if return_numpy:
            distributions: dict[int, Any] = {int(k): np.asarray(v, dtype=float) for k, v in dists_raw.items()}
        else:
            distributions = {int(k): [float(x) for x in v] for k, v in dists_raw.items()}
        return distributions, meta


class FirstOrderDataset:
    """Dataset wrapper pairing inputs (and labels if present) with distributions."""

    def __init__(
        self,
        base_dataset: DatasetLike,
        distributions: Mapping[int, Iterable[float]],
        input_getter: Callable[[object], object] | None = None,
        return_numpy: bool = True,
    ) -> None:
        """Initialize with base dataset and index-aligned distributions."""
        self.base_dataset = base_dataset
        if return_numpy:
            self.distributions: dict[int, object] = {
                int(k): np.asarray(list(v), dtype=float) for k, v in distributions.items()
            }
        else:
            self.distributions = {int(k): list(v) for k, v in distributions.items()}
        self.input_getter = input_getter
        self.return_numpy = return_numpy

        n = len(base_dataset)
        if len(self.distributions) != n:
            warnings.warn(
                (
                    f"[FirstOrderDataset] distributions count {len(self.distributions)} "
                    f"does not match dataset length {n}."
                ),
                stacklevel=2,
            )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.base_dataset)

    def _get_input(self, sample: object) -> object:
        if self.input_getter is not None:
            return self.input_getter(sample)
        # Only unpack tuples (input, label...) lists are feature vectors
        if isinstance(sample, tuple) and len(sample) >= 1:
            return sample[0]
        return sample

    def __getitem__(self, idx: int) -> object:
        """Return input (+ optional label) and its distribution."""
        sample = self.base_dataset[idx]
        dist = self.distributions.get(idx)
        if dist is None:
            msg = f"No distribution for index {idx}."
            raise KeyError(msg)

        # Treat as labeled sample only if it's a tuple of (input, label)
        if isinstance(sample, tuple) and len(sample) >= 2:
            inp, lbl = sample[0], sample[1]
            return inp, lbl, dist
        inp = self._get_input(sample)
        return inp, dist


class SimpleDataLoader:
    """A minimal Python data loader that batches items by index."""

    def __init__(
        self,
        dataset: DatasetLike,
        batch_size: int = 64,
        shuffle: bool = False,
    ) -> None:
        """Initialize loader with dataset, batch size, and shuffle flag."""
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)

    def __len__(self) -> int:
        """Return number of batches for current batch size."""
        n = len(self.dataset)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[object]]:
        """Yield lists of dataset items as batches."""
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield [self.dataset[i] for i in batch_idx]


def output_dataloader(
    base_dataset: DatasetLike,
    distributions: Mapping[int, Iterable[float]],
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    input_getter: Callable[[Any], Any] | None = None,
    return_numpy: bool = True,
) -> SimpleDataLoader:
    """Create a loader that yields inputs (and labels if present) with distributions.

    Note: `num_workers` and `pin_memory` are kept for API parity with Torch but ignored here.
    """
    _ = (num_workers, pin_memory)
    firstorderdataset = FirstOrderDataset(
        base_dataset, distributions, input_getter=input_getter, return_numpy=return_numpy
    )
    return SimpleDataLoader(firstorderdataset, batch_size=batch_size, shuffle=shuffle)


# Minimal dataset protocol (structural typing)
class DatasetLike(Protocol):
    """Minimal dataset protocol for typing (len and index access)."""

    def __len__(self) -> int:  # pragma: no cover - protocol stub
        """Return dataset length."""
        ...

    def __getitem__(self, idx: int) -> object:  # pragma: no cover - protocol stub
        """Return dataset item at index."""
        ...
