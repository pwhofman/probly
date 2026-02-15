"""Torch FirstOrder data generator."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reuse Basic Python general implementation for shared behaviors
from .first_order_datagenerator import (
    FirstOrderDataGenerator as PyFirstOrderDataGenerator,
)

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


def to_device(x: object, device: str) -> object:
    """Move tensor/nested tensors to the specified device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(xx, device) for xx in x)
    if isinstance(x, Mapping):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


if TYPE_CHECKING:  # typing-only import to satisfy Ruff's TC rules
    from collections.abc import Sized


@dataclass
class FirstOrderDataGenerator(PyFirstOrderDataGenerator):
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
    return_torch: bool = True

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

    # Removed deprecated alias; use prepares_batch_inp()

    @torch.no_grad()
    def generate_distributions(  # noqa: C901
        self,
        dataset_or_loader: object,
        *,
        progress: bool = True,
    ) -> object:
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
            dataset_len = len(cast("Sized", loader.dataset)) if loader.dataset is not None else None
        else:
            dataset = cast("Dataset", dataset_or_loader)
            dataset_len = len(cast("Sized", dataset))
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
            inpt = to_device(inpt, self.device)
            outputs = self.model(inpt)
            if not isinstance(outputs, torch.Tensor):
                msg = "Model must return a torch.Tensor (logits or probs)."
                raise TypeError(msg)
            probs = self.to_probs(outputs)
            if probs.ndim == 1:
                probs = probs.unsqueeze(0)

            # Ensure 2D
            if probs.ndim == 1:
                probs = probs.unsqueeze(0)

            batch_size_local = probs.shape[0]
            rows = probs.detach().cpu()
            for i in range(batch_size_local):
                idx = start_idx + i
                distributions[idx] = rows[i] if self.return_torch else rows[i].tolist()
            start_idx += batch_size_local
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

    def load_distributions(self, path: str | Path) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        """Load distributions from JSON and return Torch tensors.

        Returns:
        -------
        (distributions, meta)
            distributions: dict[int, torch.Tensor]
            meta: dict with any metadata saved alongside distributions
        """
        dists_list, meta = super().load_distributions(path)
        dists_tensor: dict[int, torch.Tensor] = {
            int(k): torch.tensor(v, dtype=torch.float32) for k, v in dists_list.items()
        }
        return dists_tensor, meta

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

        n = len(cast("Sized", base_dataset))
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
        return len(cast("Sized", self.base_dataset))

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
    """Save distributions to a torch binary file (.pt / .pth)."""
    path = Path(save_path)

    if path.suffix == "":
        path = path.with_suffix(".pt")
    elif path.suffix not in {".pt", ".pth"}:
        _msg_invalid_suffix = "File suffix must be '.pt' or '.pth'."
        raise ValueError(_msg_invalid_suffix)

    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(tensor_dict, path)

    if verbose:
        logger.info("Saved distributions to: %s", path)


def load_distributions_pt(
    load_path: str,
    *,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Load distributions from a torch binary file (.pt / .pth)."""
    path = Path(load_path)
    if not path.exists():
        _msg_not_found = f"File not found: {path}"
        raise FileNotFoundError(_msg_not_found)

    distributions = torch.load(path, map_location=device)

    if not isinstance(distributions, dict):
        _msg_not_dict = "Loaded object is not a dictionary."
        raise TypeError(_msg_not_dict)

    if verbose:
        logger.info("Loaded distributions from: %s", path)

    return distributions
