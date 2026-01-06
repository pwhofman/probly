"""Torch FirstOrder data generator."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, cast
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _is_probabilities(x: torch.Tensor, atol: float = 1e-4) -> bool:
    """(For to_probs idk might delete/change later) check if tensor looks like probabilities along last dim.

    Conditions:
    - all values in [0, 1]
    - rows sum approximately to 1 (within atol)
    """
    if x.numel() == 0:
        return False
    min_ok = bool(torch.all(x >= -atol))
    max_ok = bool(torch.all(x <= 1 + atol))
    if not (min_ok and max_ok):
        return False
    sums = x.sum(dim=-1)
    return bool(torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=0))


@dataclass
class FirstOrderDataGenerator:
    """Version First-Order data generator.

    Parameters
    ----------
    model:
        A Callable that maps a batch of inputs to logits or probs.
        Normally a `torch.nn.Module`.
    device:
        Device for inference (e.g., 'cpu' or 'cuda'). Default 'cpu'.
    batch_size:
        Batch size to use when wrapping a Dataset. (Default now down 64 instead of 128.)
    output_mode:
        One of {'auto', 'logits', 'probs'}. If 'auto', attempt to detect whether
        outputs are logits or probabilities. If 'logits', apply softmax. If 'probs',
        use as is. Default of course 'auto'.
    output_transform:
        func to convert raw model output to probs. If called
        this is over output_mode.
    input_getter:
        func to extract model input from dataset item.
        Signature: input_getter(sample) -> model_input
        When None expects dataset items to be (input, target) or input only.
    model_name:
        Optional string identifier. (saved with metadata)
    """

    model: torch.nn.Module | Callable[..., Any]
    device: str = "cpu"
    batch_size: int = 64
    output_mode: str = "auto"  # your options: 'auto' | 'logits' | 'probs'
    output_transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    input_getter: Callable[[Any], Any] | None = None
    model_name: str | None = None

    def to_device(self, x: object) -> object:
        """Move tensor/nested tensors to the same device if applicable."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, (list, tuple)):
            return type(x)(self.to_device(xx) for xx in x)
        if isinstance(x, Mapping):
            return {k: self.to_device(v) for k, v in x.items()}
        return x

    def to_probs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert model outputs to probabilities."""
        if self.output_transform is not None:
            return self.output_transform(outputs)

        mode = (self.output_mode or "auto").lower()
        if mode == "probs":
            return outputs
        if mode == "logits":
            return F.softmax(outputs, dim=-1)
        if mode == "auto":
            return outputs if _is_probabilities(outputs) else F.softmax(outputs, dim=-1)
        msg = f"Invalid output_mode '{self.output_mode}'. Expected one of: 'auto', 'logits', 'probs'."
        raise ValueError(msg)

    def prepares_batch_inp(self, sample: object) -> object:
        """Extract the model input from a dataset sample or batch.

        Behavior:
        - If input_getter is provided use it.
        - If the sample/batch is a tuple or list like (inputs, labels, ...),
          return the first element (inputs).
        - Otherwise return the sample as-is.
        """
        if self.input_getter is not None:
            return self.input_getter(sample)
        if isinstance(sample, (list, tuple)) and len(sample) >= 1:
            return sample[0]
        return sample

    def extract_input(self, sample: object) -> object:
        """Deprecated name use prepares_batch_inp() instead."""
        return self.prepares_batch_inp(sample)

    @torch.no_grad()
    def generate_distributions(
        self,
        dataset_or_loader: object,
        *,
        progress: bool = True,
    ) -> dict[int, list[float]]:
        """Generate per-sample probability distributions.

        Parameters
        ----------
        dataset_or_loader:
            A torch.utils.data.Dataset or torch.utils.data.DataLoader.
            Items should be tensors or tuples/dicts that have tensors.
        progress:
            If True prints simple progress information in terminal output for user to see that progress is happening.

        Returns:
        -------
        dict[int, list[float]]
            Mapping from dataset index to list of probabilities.
        """
        # Remember Blatt3: Prepare the loader
        if isinstance(dataset_or_loader, torch.utils.data.DataLoader):
            loader = dataset_or_loader
            dataset_len = len(loader.dataset) if loader.dataset is not None else None
        else:
            dataset = cast("Dataset", dataset_or_loader)
            dataset_len = len(dataset)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if isinstance(self.model, torch.nn.Module):
            self.model = self.model.to(self.device)
            self.model.eval()
        else:
            warnings.warn(
                "[FirstOrderDataGenerator] model is not a torch.nn.Module; skipping .to()/.eval().",
                stacklevel=2,
            )

        distributions: dict[int, list[float]] = {}
        start_idx = 0
        # print in batch-loop: show progress
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            inpt = self.prepares_batch_inp(batch)
            inpt = self.to_device(inpt)
            outputs = self.model(inpt)
            if not isinstance(outputs, torch.Tensor):
                msg = "Model must return a torch.Tensor (logits or probs)."
                raise TypeError(msg)
            probs = self.to_probs(outputs)
            if probs.ndim == 1:
                probs = probs.unsqueeze(0)

            probs_np = probs.detach().cpu().numpy()

            batch_size = probs_np.shape[0]
            for i in range(batch_size):
                idx = start_idx + i
                distributions[idx] = probs_np[i].tolist()

            start_idx += batch_size
            if progress:
                logger.info("[FirstOrderDataGenerator] Batch %d/%d", batch_idx + 1, total_batches)

        # progress end marker
        if progress:
            logger.info("[FirstOrderDataGenerator] Finished %d batches", total_batches)

        # warn if generated count differs from dataset length
        if dataset_len is not None and len(distributions) != dataset_len:
            # Do not raise hard error (streaming loaders may mismatch) just warn
            warnings.warn(
                (
                    f"[FirstOrderDataGenerator] generated {len(distributions)} distributions, "
                    f"but dataset length is {dataset_len}."
                ),
                stacklevel=2,
            )

        return distributions

    def get_posterior_distributions(self) -> dict[str, dict[str, torch.Tensor]]:
        """Extracts u and p from all BayesLinear layers — issue #241.

        Returns dict compatible with future torch.save/load.
        """
        distributions: dict[str, dict[str, torch.Tensor]] = {}
        model_mod = cast("torch.nn.Module", self.model)
        for name, param in model_mod.named_parameters():
            if name.endswith("_mu"):
                base_name = name[:-3]
                distributions.setdefault(base_name, {})
                distributions[base_name]["mu"] = param.detach().clone()
            elif name.endswith("_rho"):
                base_name = name[:-4]
                distributions[base_name]["rho"] = param.detach().clone()

        return distributions

    # JSON save/load methods
    def save_distributions(
        self,
        path: str | Path,
        distributions: Mapping[int, Iterable[float]],
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Save distributions and minimal metadata as JSON."""
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
        """Load distributions and metadata from JSON.

        Returns:
        -------
        (distributions, meta)
            distributions: dict[int, list[float]]
            meta: dict with any metadata saved alongside distributions
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        meta = obj.get("meta", {}) or {}
        dists_raw = obj.get("distributions", {}) or {}
        # Convert keys back to int
        distributions: dict[int, list[float]] = {int(k): list(v) for k, v in dists_raw.items()}
        return distributions, meta


class FirstOrderDataset(Dataset):
    """Wrap an existing dataset (like base_dataset) with first-order distributions for training/eval.

    Returns items as (input, distribution) if the base dataset yields only input,
    or (input, label, distribution) if the base dataset yields (input, label).
    """

    def __init__(
        self,
        base_dataset: Dataset,
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
                    f"[FirstOrderDataset] distributions count {len(self.distributions)} "
                    f"does not match dataset length {n}."
                ),
                stacklevel=2,
            )

    def __len__(self) -> int:
        """Return number of samples in the base dataset."""
        return len(self.base_dataset)

    def _get_input(self, sample: object) -> object:
        """Extract input from a sample, using input_getter if provided."""
        if self.input_getter is not None:
            return self.input_getter(sample)
        if isinstance(sample, (list, tuple)) and len(sample) >= 1:
            return sample[0]
        return sample

    def __getitem__(self, idx: int) -> object:
        """Return input (+ optional label) and distribution at index."""
        sample = self.base_dataset[idx]
        dist = self.distributions.get(idx)
        if dist is None:
            msg = f"No distribution for index {idx}."
            raise KeyError(msg)

        dist_tensor = torch.tensor(dist, dtype=torch.float32)
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            inp, lbl = sample[0], sample[1]
            return inp, lbl, dist_tensor
        inp = self._get_input(sample)
        return inp, dist_tensor


def output_dataloader(
    base_dataset: Dataset,
    distributions: Mapping[int, Iterable[float]],
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    input_getter: Callable[[Any], Any] | None = None,
) -> DataLoader:
    """Creates DataLoader pairing inputs (labels if any available) with first-order distribs."""
    firstorderdataset = FirstOrderDataset(base_dataset, distributions, input_getter=input_getter)
    return DataLoader(
        firstorderdataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def save_distributions_pt(
    tensor_dict: dict[str, Any],
    save_path: str,
    *,
    create_dir: bool = False,
    verbose: bool = True,
) -> None:
    """Save a dictionary of tensors as a .pt/.pth file using torch.save.

    Parameters
    ----------
    tensor_dict:
        Mapping of names to tensors to serialize.
    save_path:
        Target path. If it does not end with .pt or .pth, .pt will be appended.
    create_dir:
        Whether to create the parent directory automatically.
    verbose:
        When True, prints a short summary including per-tensor shapes and total size.
    """
    path = Path(save_path)
    if not path.suffix.endswith((".pt", ".pth")):
        path = path.with_suffix(".pt")

    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(tensor_dict, path)

    if verbose:
        logger.info("Tensor dict has been saved to: %s", str(path))
        logger.info("Dictionary overview:")
        total_size = 0.0
        for key, tensor in tensor_dict.items():
            if isinstance(tensor, torch.Tensor):
                size_mb = tensor.element_size() * tensor.nelement() / (1024**2)
                total_size += size_mb
                logger.info("- %s: %s, %s, %.2f MB", key, tuple(tensor.shape), tensor.dtype, size_mb)
            else:
                logger.info("- %s: non-tensor value of type %s", key, type(tensor).__name__)
        logger.info("Total size (tensor entries): %.2f MB", total_size)


def load_distributions_pt(
    load_path: str,
    *,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Load a tensor dictionary from a .pt/.pth file using torch.load.

    Parameters
    ----------
    load_path:
        Path to the saved tensor dictionary (.pt or .pth).
    device:
        Target device for loaded tensors (e.g., 'cpu', 'cuda:0').
        When None, keeps original device information.
    verbose:
        When True, prints a short summary of the loaded contents.
    """
    path = Path(load_path)
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    tensor_dict = cast("dict[str, Any]", torch.load(path, map_location=device))

    if verbose:
        logger.info("Tensor dict has been loaded from %s", str(path))
        logger.info("Loaded contents overview:")
        for key, tensor in tensor_dict.items():
            if isinstance(tensor, torch.Tensor):
                logger.info(
                    "- %s: %s, %s, device: %s",
                    key,
                    tuple(tensor.shape),
                    tensor.dtype,
                    str(tensor.device),
                )
            else:
                logger.info("- %s: non-tensor value of type %s", key, type(tensor).__name__)

    return tensor_dict
