"""JAX FirstOrder data generator."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, cast
import warnings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from pathlib import Path


import jax
import jax.nn as jnn
import jax.numpy as jnp

# Reuse the Python-first implementations for non-JAX-specific pieces
from .first_order_datagenerator import (
    DatasetLike,
    FirstOrderDataGenerator as PyFirstOrderDataGenerator,
    FirstOrderDataset as PyFirstOrderDataset,
    SimpleDataLoader,
)

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
        if len(outputs) == 0:
            return jnp.zeros((0, 0), dtype=jnp.float32)
        first = outputs[0]
        if isinstance(first, (int, float)):
            return _ensure_2d(jnp.array(outputs, dtype=jnp.float32))
        return _ensure_2d(jnp.array(outputs, dtype=jnp.float32))
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
class FirstOrderDataGenerator(PyFirstOrderDataGenerator):
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

    output_transform: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    return_jax: bool = True

    def to_probs(self, outputs: object) -> object:
        """Convert model outputs to probabilities as jnp.ndarray or lists."""
        arr = (
            _to_batch_outputs(outputs)
            if self.output_transform is None
            else _ensure_2d(cast("jnp.ndarray", self.output_transform(outputs)))
        )
        mode = (self.output_mode or "auto").lower()
        if mode == "probs":
            probs = arr
        elif mode == "logits":
            probs = jnn.softmax(arr, axis=-1)
        elif mode == "auto":
            probs = arr if _is_probabilities(arr) else jnn.softmax(arr, axis=-1)
        else:
            msg = f"Invalid output_mode '{self.output_mode}'. Expected one of: auto, logits, probs."
            raise ValueError(msg)
        # JAX-specific default: return jnp.ndarray unless explicitly disabled.
        return probs if self.return_jax else [list(map(float, row.tolist())) for row in probs]

    def to_device(self, x: object) -> object:
        """Move arrays/lists/dicts to configured JAX device if available.

        If no matching device is found, returns the input unchanged.
        """
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

    def _batchify_inputs(self, batch: Sequence[object]) -> jnp.ndarray:
        inputs = [self.prepares_batch_inp(sample) for sample in batch]
        try:
            arr = jnp.array(inputs, dtype=jnp.float32)
        except (TypeError, ValueError):
            arr = jnp.array(inputs, dtype=object)
        return arr

    def generate_distributions(self, dataset_or_loader: object, *, progress: bool = True) -> object:
        """Generate per-sample probs distribs for a dataset or loader.

        Returns a dict mapping dataset indices to either jnp.ndarray rows
        (when return_jax is True) or lists of floats.
        """
        # Use base loader for non-JAX specifics
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
            inpt = self._batchify_inputs(batch)
            inpt = cast("jnp.ndarray", self.to_device(inpt))
            outputs = self.model(inpt)
            probs_obj = self.to_probs(outputs)
            if isinstance(probs_obj, jnp.ndarray):
                probs = _ensure_2d(probs_obj)
            else:
                probs = _ensure_2d(jnp.array(probs_obj, dtype=jnp.float32))

            batch_size_local = int(probs.shape[0])
            for i in range(batch_size_local):
                idx = start_idx + i
                distributions[idx] = probs[i] if self.return_jax else list(map(float, probs[i].tolist()))
            start_idx += batch_size_local
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

    # Explicitly expose JSON save/load via pass-through methods for discoverability
    def save_distributions(
        self,
        path: str | Path,
        distributions: Mapping[int, Iterable[float]],
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Pass-through to base JSON save_distributions implementation.

        Kept here for users to know you can use this.
        """
        return super().save_distributions(path, distributions, meta=meta)

    def load_distributions(self, path: str | Path) -> tuple[dict[int, jnp.ndarray], dict[str, Any]]:
        """Load distributions and convert to JAX arrays.

        Returns:
        -------
        (distributions, meta)
            distributions: dict[int, jnp.ndarray]
            meta: dict with any metadata saved alongside distributions
        """
        dists, meta = super().load_distributions(path)
        jax_dists = {int(i): jnp.array(v, dtype=jnp.float32) for i, v in dists.items()}
        return jax_dists, meta


class FirstOrderDataset(PyFirstOrderDataset):
    """Subclass the Python-first dataset, converting distributions to jnp arrays."""

    def __getitem__(self, idx: int) -> object:
        """Return item from base dataset but convert distribution to jnp.ndarray.

        If the base dataset yields (inp, label, dist) returns
        (inp, label, jnp.array(dist)). If it yields (inp, dist) returns
        (inp, jnp.array(dist)).
        """
        item = super().__getitem__(idx)
        if isinstance(item, tuple) and len(item) == 3:
            inp, lbl, dist = item
            return inp, lbl, jnp.array(dist, dtype=jnp.float32)
        inp, dist = item
        return inp, jnp.array(dist, dtype=jnp.float32)


# Reuse SimpleDataLoader from the Python-first module


class JAXOutputDataLoader:
    """JAX-native output loader yielding batches with JAX arrays.

    Yields per-batch tuples of (inputs, distributions) or (inputs, labels, distributions)
    depending on whether the dataset has labels. Distributions are stacked
    as jnp.ndarray in shape [batch, classes]. Inputs are best-effort converted to
    jnp.ndarray, if conversion is not possible, the original
    Python sequence is returned.

    Shuffling uses jax.random.permutation. Optional device placement moves any JAX arrays
    to the selected device via jax.device_put.
    """

    def __init__(
        self,
        dataset: FirstOrderDataset,
        *,
        batch_size: int = 64,
        shuffle: bool = False,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the JAX output data loader.

        Parameters
        ----------
        dataset: FirstOrderDataset
            Dataset providing items with distributions already as `jnp.ndarray`.
        batch_size: int
            Number of samples per batch.
        shuffle: bool
            Whether to shuffle indices using `jax.random.permutation`.
        seed: int | None
            Seed for the JAX PRNG used when shuffling.
        device: str | None
            Target device for array placement (e.g., "cpu", "gpu", "tpu").
        """
        self.dataset = dataset
        self.batch_size = int(max(1, batch_size))
        self.shuffle = shuffle
        self.seed = seed
        self.device = device
        self._num_items = len(dataset)
        self._num_batches = (self._num_items + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        """Return the number of batches in the loader."""
        return self._num_batches

    def _batchify_inputs(self, values: Sequence[object]) -> object:
        try:
            return jnp.array(values)
        except (TypeError, ValueError):
            try:
                return jnp.stack([jnp.asarray(v) for v in values])
            except (TypeError, ValueError):
                return list(values)

    def _to_device_any(self, x: object) -> object:
        dev = _get_device(self.device)
        if dev is None:
            return x
        if isinstance(x, jnp.ndarray):
            return jax.device_put(x, device=dev)
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_device_any(xx) for xx in x)
        if isinstance(x, dict):
            return {k: self._to_device_any(v) for k, v in x.items()}
        return x

    def __iter__(self) -> Iterator[object]:
        """Iterate over the dataset, yielding JAX-batched outputs.

        Yields either (inputs, distributions) or (inputs, labels, distributions)
        depending on whether labels are present.
        """
        n = self._num_items
        if self.shuffle:
            key = jax.random.PRNGKey(0 if self.seed is None else int(self.seed))
            indices = jax.random.permutation(key, n)
        else:
            indices = jnp.arange(n)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_idx = indices[start:end]
            idx_list = list(map(int, list(batch_idx)))

            inputs: list[object] = []
            labels: list[object] = []
            dists: list[jnp.ndarray] = []

            for i in idx_list:
                item = self.dataset[i]
                if isinstance(item, tuple) and len(item) == 3:
                    inp, lbl, dist = item
                    inputs.append(inp)
                    labels.append(lbl)
                    dists.append(dist)
                else:
                    inp, dist = item
                    inputs.append(inp)
                    dists.append(dist)

            inpt_batch = self._batchify_inputs(inputs)
            dist_batch = jnp.stack(dists, axis=0)
            inpt_batch = self._to_device_any(inpt_batch)
            dist_batch = self._to_device_any(dist_batch)

            if labels:
                try:
                    lbl_batch = jnp.array(labels)
                except (TypeError, ValueError):
                    lbl_batch = list(labels)
                lbl_batch = self._to_device_any(lbl_batch)
                yield inpt_batch, lbl_batch, dist_batch
            else:
                yield inpt_batch, dist_batch


def output_dataloader(
    base_dataset: DatasetLike,
    distributions: Mapping[int, Iterable[float]],
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    input_getter: Callable[[Any], Any] | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> JAXOutputDataLoader:
    """Create a JAX-native loader yielding JAX arrays for distributions.

    Parameters num_workers and pin_memory are kept for API parity but ignored !
    Use seed to control shuffle permutation and device to place arrays
    (e.g., "cpu", "gpu", "tpu" or more specific like "gpu:0").
    """
    _ = (num_workers, pin_memory)
    firstorderdataset = FirstOrderDataset(base_dataset, distributions, input_getter=input_getter)
    return JAXOutputDataLoader(
        firstorderdataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        device=device,
    )


# DatasetLike is imported from the Python-first module
