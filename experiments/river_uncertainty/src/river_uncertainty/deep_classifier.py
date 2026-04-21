"""Minimal online classifier wrapping a ``torch.nn.Module``.

This is a lightweight replacement for ``deep_river.classification.Classifier``
that avoids the Python-version conflict (deep-river requires ``<3.13``, probly
requires ``>=3.13``).  Only the functionality needed by the uncertainty
experiments is implemented: ``learn_one``, ``predict_proba_one``, and
``mc_forward_passes``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn


class OnlineClassifier:
    """Online deep classifier with MC Dropout support.

    Args:
        module_factory: ``(n_features, n_classes) -> nn.Module`` callable.
            Called lazily when the feature/class set is first known.
        optimizer_fn: PyTorch optimizer class (e.g. ``torch.optim.Adam``).
        lr: Learning rate.
        seed: Random seed for weight initialisation.
    """

    def __init__(
        self,
        module_factory: Callable[[int, int], nn.Module],
        optimizer_fn: type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 0.01,
        seed: int = 0,
    ) -> None:
        self._factory = module_factory
        self._optimizer_fn = optimizer_fn
        self._lr = lr
        self._seed = seed

        self._module: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._features: list[str] | None = None
        self._classes: list[Any] = []
        self._class_idx: dict[Any, int] = {}

    @property
    def is_initialized(self) -> bool:
        return self._module is not None

    def _init_module(self, n_features: int, n_classes: int) -> None:
        torch.manual_seed(self._seed)
        self._module = self._factory(n_features, n_classes)
        self._optimizer = self._optimizer_fn(self._module.parameters(), lr=self._lr)

    def _ensure_class(self, y: Any) -> None:
        if y not in self._class_idx:
            self._class_idx[y] = len(self._classes)
            self._classes.append(y)

    def _dict2tensor(self, x: dict[str, float]) -> torch.Tensor:
        if self._features is None:
            self._features = sorted(x.keys())
        vals = [float(x.get(f, 0.0)) for f in self._features]
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(0)

    def learn_one(self, x: dict[str, float], y: Any) -> None:
        if self._features is None:
            self._features = sorted(x.keys())

        prev_n_classes = len(self._classes)
        self._ensure_class(y)

        if not self.is_initialized:
            self._init_module(len(self._features), len(self._classes))
        elif len(self._classes) > prev_n_classes:
            # NOTE: this reinitialises the entire module from scratch, discarding
            # all learned weights.  A production implementation should expand the
            # output layer in-place.  We accept the reset here because the
            # experiment streams expose all classes within the first few samples.
            self._init_module(len(self._features), len(self._classes))

        assert self._module is not None
        assert self._optimizer is not None

        x_t = self._dict2tensor(x)
        y_t = torch.tensor([self._class_idx[y]], dtype=torch.long)

        self._module.train()
        logits = self._module(x_t)
        loss = nn.functional.cross_entropy(logits, y_t)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def predict_proba_one(self, x: dict[str, float]) -> dict[Any, float]:
        if not self.is_initialized or not self._classes:
            return {}
        assert self._module is not None

        x_t = self._dict2tensor(x)
        self._module.eval()
        with torch.inference_mode():
            logits = self._module(x_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        return {c: float(probs[i]) for i, c in enumerate(self._classes)}

    def mc_forward_passes(
        self,
        x: dict[str, float],
        n: int = 15,
    ) -> tuple[np.ndarray, tuple[Any, ...]]:
        """Run *n* stochastic forward passes with dropout active.

        Returns:
            ``(probs, classes)`` where ``probs`` has shape ``(n, n_classes)``
            and ``classes`` is the ordered class tuple.
        """
        if not self.is_initialized or not self._classes:
            k = max(len(self._classes), 1)
            classes = tuple(self._classes) if self._classes else (0,)
            return np.full((n, k), 1.0 / k), classes
        assert self._module is not None

        x_t = self._dict2tensor(x)
        self._module.train()  # keep dropout active
        samples = []
        with torch.no_grad():
            for _ in range(n):
                logits = self._module(x_t)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                samples.append(probs.numpy())

        return np.stack(samples), tuple(self._classes)
