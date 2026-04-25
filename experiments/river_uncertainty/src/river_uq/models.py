"""Uniform model wrappers for the paper experiment.

All wrappers expose:
- learn_one(x, y) -> None
- predict_one(x) -> int (predicted class label)
- epistemic_decomposition(x) -> Decomposition with .total, .aleatoric, .epistemic

Three kinds:
- "arf"            wraps river.forest.ARFClassifier; UQ via probly.representer
- "deep_ensemble"  K=10 independent online torch MLPs; UQ via direct sample build
- "mc_dropout"     single online torch MLP with K=10 stochastic forward passes
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import torch
from river.forest import ARFClassifier
from torch import nn

from probly.quantification import quantify
from probly.quantification.decomposition import Decomposition
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)
from probly.representer import representer

K_MEMBERS = 10
HIDDEN = (64, 32)
DROPOUT_P = 0.2
LR = 1e-3


class ModelWrapper(Protocol):
    """Uniform interface for streaming UQ models."""

    def learn_one(self, x: dict[str, float], y: int) -> None: ...
    def predict_one(self, x: dict[str, float]) -> int: ...
    def epistemic_decomposition(self, x: dict[str, float]) -> Decomposition: ...


# ---------- ARF ----------


class _ARFWrapper:
    def __init__(self, seed: int) -> None:
        self._arf = ARFClassifier(n_models=K_MEMBERS, seed=seed)
        self._seed = seed

    def learn_one(self, x: dict[str, float], y: int) -> None:
        self._arf.learn_one(x, y)

    def predict_one(self, x: dict[str, float]) -> int:
        pred = self._arf.predict_one(x)
        return int(pred) if pred is not None else 0

    def epistemic_decomposition(self, x: dict[str, float]) -> Decomposition:
        sample = representer(self._arf).represent(x)
        return quantify(sample)

    @property
    def n_drifts_detected(self) -> int:
        return int(self._arf.n_drifts_detected())


# ---------- shared torch MLP ----------


def _build_mlp(n_features: int, n_classes: int) -> nn.Module:
    layers: list[nn.Module] = []
    prev = n_features
    for h in HIDDEN:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(DROPOUT_P)]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


class _OnlineMLP:
    """Minimal online classifier - one SGD step per learn_one."""

    def __init__(self, seed: int, factory: Callable[[int, int], nn.Module] = _build_mlp) -> None:
        self._seed = seed
        self._factory = factory
        self._module: nn.Module | None = None
        self._opt: torch.optim.Optimizer | None = None
        self._features: list[str] | None = None
        self._classes: list[Any] = []
        self._cls_idx: dict[Any, int] = {}

    def _ensure_init(self, x: dict[str, float], y: Any | None = None) -> None:
        if self._features is None:
            self._features = sorted(x.keys())
        if y is not None and y not in self._cls_idx:
            self._cls_idx[y] = len(self._classes)
            self._classes.append(y)
        target_n_classes = max(len(self._classes), 2)
        needs_init = self._module is None
        if not needs_init:
            assert self._module is not None
            last = self._module[-1]
            assert isinstance(last, nn.Linear)
            if last.out_features < target_n_classes:
                needs_init = True
        if needs_init:
            torch.manual_seed(self._seed)
            self._module = self._factory(len(self._features), target_n_classes)
            self._opt = torch.optim.Adam(self._module.parameters(), lr=LR)

    def _to_tensor(self, x: dict[str, float]) -> torch.Tensor:
        assert self._features is not None
        vals = [float(x.get(f, 0.0)) for f in self._features]
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(0)

    def learn_one(self, x: dict[str, float], y: Any) -> None:
        self._ensure_init(x, y)
        assert self._module is not None and self._opt is not None
        x_t = self._to_tensor(x)
        y_t = torch.tensor([self._cls_idx[y]], dtype=torch.long)
        self._module.train()
        logits = self._module(x_t)
        loss = nn.functional.cross_entropy(logits, y_t)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

    def predict_proba(self, x: dict[str, float], *, dropout_active: bool = False) -> np.ndarray:
        self._ensure_init(x)
        if self._module is None or not self._classes:
            return np.full(2, 0.5)
        x_t = self._to_tensor(x)
        if dropout_active:
            self._module.train()
        else:
            self._module.eval()
        with torch.no_grad():
            logits = self._module(x_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs

    def predict_one(self, x: dict[str, float]) -> int:
        if not self._classes:
            return 0
        probs = self.predict_proba(x)
        return int(self._classes[int(np.argmax(probs))])

    @property
    def class_order(self) -> list[Any]:
        return list(self._classes)


# ---------- Deep Ensemble ----------


class _DeepEnsembleWrapper:
    def __init__(self, seed: int) -> None:
        self._members = [_OnlineMLP(seed=seed * 1000 + i) for i in range(K_MEMBERS)]
        self._seed = seed

    def learn_one(self, x: dict[str, float], y: int) -> None:
        for m in self._members:
            m.learn_one(x, y)

    def predict_one(self, x: dict[str, float]) -> int:
        sample = self._stacked_probs(x, dropout_active=False)
        mean = sample.mean(axis=0)
        order = self._members[0].class_order or [0, 1]
        return int(order[int(np.argmax(mean))])

    def _class_universe(self) -> list[Any]:
        seen: list[Any] = []
        for m in self._members:
            for c in m.class_order:
                if c not in seen:
                    seen.append(c)
        return seen or [0, 1]

    def _stacked_probs(self, x: dict[str, float], *, dropout_active: bool) -> np.ndarray:
        classes = self._class_universe()
        idx = {c: i for i, c in enumerate(classes)}
        out = np.zeros((K_MEMBERS, len(classes)), dtype=np.float64)
        for row, m in enumerate(self._members):
            probs = m.predict_proba(x, dropout_active=dropout_active)
            for ci, c in enumerate(m.class_order):
                out[row, idx[c]] = probs[ci]
            row_sum = out[row].sum()
            if row_sum > 0:
                out[row] /= row_sum
            else:
                out[row] = 1.0 / len(classes)
        return out

    def epistemic_decomposition(self, x: dict[str, float]) -> Decomposition:
        stacked = self._stacked_probs(x, dropout_active=False)
        sample = ArrayCategoricalDistributionSample(
            array=ArrayCategoricalDistribution(unnormalized_probabilities=stacked),
            sample_axis=0,
        )
        return quantify(sample)


# ---------- MC Dropout ----------


class _MCDropoutWrapper:
    def __init__(self, seed: int, n_passes: int = K_MEMBERS) -> None:
        self._mlp = _OnlineMLP(seed=seed)
        self._n_passes = n_passes

    def learn_one(self, x: dict[str, float], y: int) -> None:
        self._mlp.learn_one(x, y)

    def predict_one(self, x: dict[str, float]) -> int:
        return self._mlp.predict_one(x)

    def epistemic_decomposition(self, x: dict[str, float]) -> Decomposition:
        classes = self._mlp.class_order or [0, 1]
        rows = np.stack(
            [self._mlp.predict_proba(x, dropout_active=True) for _ in range(self._n_passes)]
        )
        if rows.shape[1] < len(classes):
            pad = np.zeros((rows.shape[0], len(classes) - rows.shape[1]))
            rows = np.concatenate([rows, pad], axis=1)
        sums = rows.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        rows = rows / sums
        sample = ArrayCategoricalDistributionSample(
            array=ArrayCategoricalDistribution(unnormalized_probabilities=rows),
            sample_axis=0,
        )
        return quantify(sample)


# ---------- Factory ----------


def build_model(kind: str, seed: int) -> ModelWrapper:
    """Build a model wrapper by name.

    Args:
        kind: One of ``"arf"``, ``"deep_ensemble"``, ``"mc_dropout"``.
        seed: Model seed (independent across (kind, stream, seed) combos).

    Returns:
        A model exposing the ``ModelWrapper`` protocol.
    """
    if kind == "arf":
        return _ARFWrapper(seed=seed)
    if kind == "deep_ensemble":
        return _DeepEnsembleWrapper(seed=seed)
    if kind == "mc_dropout":
        return _MCDropoutWrapper(seed=seed)
    msg = f"unknown model kind: {kind!r}"
    raise ValueError(msg)
