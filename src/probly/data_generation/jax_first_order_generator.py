"""JAX FirstOrder data generator."""

# ruff: noqa: INP001

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any, Protocol, cast
import warnings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence


import jax
import jax.nn as jnn
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def _is_probabilities(x: jnp.ndarray, atol: float = 1e-4) -> bool:
    """Check if array looks like probabilities along last dim.

    Conditions:
    - all values in [0, 1]
    - rows sum approximately to 1 (within atol)
    """
    if x.size == 0:
        return False
    min_ok = jnp.all(x >= -atol)
    max_ok = jnp.all(x <= 1 + atol)
    if not (bool(min_ok) and bool(max_ok)):
        return False
    sums = jnp.sum(x, axis=-1)
    ones = jnp.ones_like(sums)
    return bool(jnp.allclose(sums, ones, atol=atol, rtol=0))


def _ensure_2d(x: jnp.ndarray) -> jnp.ndarray:
    if x.ndim == 1:
        return x[jnp.newaxis, :]
    return x


def _to_batch_outputs(outputs: object) -> jnp.ndarray:
    """Normalize various output shapes into a 2D jnp.ndarray [batch, classes]."""
    if outputs is None:
        return jnp.zeros((0, 0), dtype=jnp.float32)
    if isinstance(outputs, jnp.ndarray):
        return _ensure_2d(outputs)
    if isinstance(outputs, (list, tuple)):
        # If first element is a number, treat as single sample vector
        if len(outputs) == 0:
            return jnp.zeros((0, 0), dtype=jnp.float32)
        first = outputs[0]
        if isinstance(first, (int, float)):
            return _ensure_2d(jnp.array(outputs, dtype=jnp.float32))
        # Else assume batch of vectors
        return _ensure_2d(jnp.array(outputs, dtype=jnp.float32))
    # Fallback: treat scalar or unknown as singlesample oneelement vector
    return _ensure_2d(jnp.array([float(cast("Any", outputs))], dtype=jnp.float32))


def _get_device(device: str | None) -> jax.Device | None:
    if not device:
        return None
    devs = jax.devices()
    # Match by platform (cpu, gpu, tpu) or exact string of repr
    for d in devs:
        if d.platform == device:
            return d
    # Try more specific forms like cuda, cuda:0, gpu:0, cpu:0
    for d in devs:
        if device in (str(d), f"{d.platform}:{d.id}"):
            return d
    return None


@dataclass
class FirstOrderDataGenerator:
    """JAX-native FirstOrder data generator.

    Parameters
    ----------
    model:
            Callable that maps a batch of inputs to logits or probs. Typically
            a JAX-transformed function that accepts jnp.ndarray inputs and
            returns jnp.ndarray outputs.
    device:
            Target device platform (e.g., cpu, gpu, tpu). Default cpu.
    batch_size:
            Batch size to use when wrapping Dataset.
    output_mode:
            One of {auto, logits, probs}. If auto, attempt to detect whether
            outputs are logits or probabilities. If logits apply softmax. If probs
            use as is.
    output_transform:
            Function to convert raw model output to probs. Rem:Overrides output_mode when provided!
    input_getter:
            Function to extract model input from a dataset item.
            Signature: input_getter(sample) -> model_input
            When None expects dataset items to be (input, target) or input only.
    model_name:
            Optional string identifier (saved with metadata).
    """

    model: Callable[..., Any]
    device: str = "cpu"
    batch_size: int = 64
    output_mode: str = "auto"
    output_transform: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    input_getter: Callable[[Any], Any] | None = None
    model_name: str | None = None

    def to_probs(self, outputs: jnp.ndarray) -> jnp.ndarray:
        """Convert model outputs to probs."""
        if self.output_transform is not None:
            return _ensure_2d(self.output_transform(outputs))

        mode = (self.output_mode or "auto").lower()
        outputs = _ensure_2d(outputs)
        if mode == "probs":
            return outputs
        if mode == "logits":
            return jnn.softmax(outputs, axis=-1)
        if mode == "auto":
            return outputs if _is_probabilities(outputs) else jnn.softmax(outputs, axis=-1)
        msg = f"Invalid output_mode '{self.output_mode}'. Expected one of: auto, logits, probs."
        raise ValueError(msg)

    def to_device(self, x: object) -> object:
        """Move arrays/nested arrays to the configured JAX device."""
        dev = _get_device(self.device)
        if dev is None:
            return x
        if isinstance(x, jnp.ndarray):
            return jax.device_put(x, device=dev)
        if isinstance(x, (list, tuple)):
            return type(x)(self.to_device(xx) for xx in x)
        if isinstance(x, dict):
            return {k: self.to_device(v) for k, v in x.items()}
        return x

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

    def _batchify_inputs(self, batch: Sequence[object]) -> jnp.ndarray:
        """Convert a batch of samples to a jnp.ndarray if possible."""
        inputs = [self.prepares_batch_inp(sample) for sample in batch]
        # Best-effort conversion to jnp.ndarray; fallback to object array
        try:
            arr = jnp.array(inputs, dtype=jnp.float32)
        except (TypeError, ValueError):
            arr = jnp.array(inputs, dtype=object)
        return arr

    def generate_distributions(
        self,
        dataset_or_loader: object,
        *,
        progress: bool = True,
    ) -> dict[int, list[float]]:
        """Generate persample probability distribs.

        Accepts Dataset-like or JaxDataLoader. Returns a dict mapping index
        to a list of probs.
        """
        if isinstance(dataset_or_loader, JaxDataLoader):
            loader = dataset_or_loader
            dataset_len = len(loader.dataset) if hasattr(loader, "dataset") else None
        else:
            dataset = cast("DatasetLike", dataset_or_loader)
            dataset_len = len(dataset)
            loader = JaxDataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        distributions: dict[int, list[float]] = {}
        start_idx = 0
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            inpt = self._batchify_inputs(batch)
            inpt = cast("jnp.ndarray", self.to_device(inpt))
            outputs = self.model(inpt)
            if not isinstance(outputs, (jnp.ndarray,)):
                msg = "Model must return a jnp.ndarray (logits or probs)."
                raise TypeError(msg)
            probs = self.to_probs(outputs)
            probs = _ensure_2d(probs)

            batch_size = probs.shape[0]
            for i in range(batch_size):
                idx = start_idx + i
                distributions[idx] = list(map(float, probs[i].tolist()))

            start_idx += batch_size
            if progress:
                logger.info("[FirstOrderDataGenerator:JAX] Batch %d/%d", batch_idx + 1, total_batches)

        if progress:
            logger.info("[FirstOrderDataGenerator:JAX] Finished %d batches", total_batches)

        if dataset_len is not None and len(distributions) != dataset_len:
            warnings.warn(
                (
                    f"[FirstOrderDataGenerator:JAX] generated {len(distributions)} distributions, "
                    f"but dataset length is {dataset_len}."
                ),
                stacklevel=2,
            )

        return distributions

    # JSON save/load methods
    def save_distributions(
        self,
        path: str | Path,
        distributions: Mapping[int, Iterable[float]],
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Save distribs and minimal metadata as JSON."""
        path = Path(path)
        serializable = {
            "meta": {
                "model_name": self.model_name,
                **(meta or {}),
            },
            "distributions": {str(k): list(v) for k, v in distributions.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)

    def load_distributions(self, path: str | Path) -> tuple[dict[int, list[float]], dict[str, Any]]:
        """Load distribs and metadata from JSON."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        meta = obj.get("meta", {}) or {}
        dists_raw = obj.get("distributions", {}) or {}
        distributions: dict[int, list[float]] = {int(k): list(v) for k, v in dists_raw.items()}
        return distributions, meta


class FirstOrderDataset:
    """Wrap an existing dataset with firstorder distribs for training/eval.

    Returns items as (input, distribution) if the base dataset yields only input
    or (input, label, distribution) if the base dataset yields (input, label).
    """

    def __init__(
        self,
        base_dataset: DatasetLike,
        distributions: Mapping[int, Iterable[float]],
        input_getter: Callable[[object], object] | None = None,
    ) -> None:
        """Initialize with base dataset and index-aligned distributions."""
        self.base_dataset = base_dataset
        self.distributions: dict[int, list[float]] = {int(k): list(v) for k, v in distributions.items()}
        self.input_getter = input_getter

        n = len(base_dataset)
        if len(self.distributions) != n:
            warnings.warn(
                (
                    f"[FirstOrderDataset:JAX] distributions count {len(self.distributions)} "
                    f"does not match dataset length {n}."
                ),
                stacklevel=2,
            )

    def __len__(self) -> int:
        """Return number of samples in the base dataset."""
        return len(self.base_dataset)

    def _get_input(self, sample: object) -> object:
        if self.input_getter is not None:
            return self.input_getter(sample)
        if isinstance(sample, tuple) and len(sample) >= 1:
            return sample[0]
        return sample

    def __getitem__(self, idx: int) -> object:
        """Return input (+ optional label) and distribution at index."""
        sample = self.base_dataset[idx]
        dist = self.distributions.get(idx)
        if dist is None:
            msg = f"No distribution for index {idx}."
            raise KeyError(msg)

        dist_arr = jnp.array(dist, dtype=jnp.float32)
        if isinstance(sample, tuple) and len(sample) >= 2:
            inp, lbl = sample[0], sample[1]
            return inp, lbl, dist_arr
        inp = self._get_input(sample)
        return inp, dist_arr


class JaxDataLoader:
    """A minimal JAX-friendly data loader that batches items by index."""

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
) -> JaxDataLoader:
    """Create a loader that yields inputs (and labels if present) with distributions.

    Note: `num_workers` and `pin_memory` are kept for API parity with Torch but ignored here.
    """
    # num_workers/pin_memory kept for signature parity; ignored in JAX
    # Emit warnings when nondefault values are provided so users know they have no effect.
    if num_workers != 0:
        warnings.warn(
            f"[JAX output_dataloader] 'num_workers'={num_workers} is ignored on JAX.",
            stacklevel=2,
        )
    if pin_memory:
        warnings.warn(
            "[JAX output_dataloader] 'pin_memory=True' is ignored on JAX.",
            stacklevel=2,
        )
    firstorderdataset = FirstOrderDataset(base_dataset, distributions, input_getter=input_getter)
    return JaxDataLoader(firstorderdataset, batch_size=batch_size, shuffle=shuffle)


class DatasetLike(Protocol):
    """Minimal dataset protocol for typing (len and index access)."""

    def __len__(self) -> int:  # pragma: no cover - protocol stub
        """Return dataset length."""
        ...

    def __getitem__(self, idx: int) -> object:  # pragma: no cover - protocol stub
        """Return dataset item at index."""
        ...
