"""Baseline AL estimator for ``plain`` and ``ensemble`` methods.

These models have no UQ wrapping. They are trained with vanilla cross-entropy
and pair with the traditional AL strategies that need raw class probabilities
(``margin``) or penultimate features (``badge``). They satisfy the
:class:`probly.evaluation.active_learning.strategies.BadgeEstimator` protocol
(``predict``, ``predict_proba``, ``embed``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from probly_benchmark import models
from probly_benchmark.al_estimator._common import embed_last_linear, make_train_cfg, train_cross_entropy

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class BaselineALEstimator:
    """AL estimator for the UQ-free baselines (``plain`` and ``ensemble``).

    Construction is direct -- it does not go through
    :func:`probly_benchmark.builders.build_model` or
    :func:`probly.predict`. ``plain`` builds a single base model;
    ``ensemble`` builds a list of independently-initialised base models.
    Each ``fit`` call rebuilds the model(s) and trains with vanilla
    cross-entropy, mirroring how the AL loop expects a fresh estimator
    each iteration.
    """

    def __init__(
        self,
        *,
        method_name: str,
        cfg: DictConfig,
        base_model_name: str,
        num_classes: int,
        device: torch.device,
        in_features: int | None = None,
        num_members: int = 1,
        pred_batch_size: int = 512,
    ) -> None:
        """Initialise the baseline estimator.

        Args:
            method_name: ``"plain"`` or ``"ensemble"``. Anything else is a
                programming error and raises ``ValueError``.
            cfg: Training configuration (epochs, optimizer, scheduler, ...).
            base_model_name: Name of the base architecture (e.g.
                ``"tabular_mlp"``).
            num_classes: Number of output classes.
            device: Torch device for training and inference.
            in_features: Input feature dimension for tabular base models.
                Forwarded to :func:`probly_benchmark.models.get_base_model`
                as a kwarg. ``None`` for image base models that infer
                their input shape.
            num_members: Number of ensemble members. Ignored when
                ``method_name == "plain"``. Defaults to 1.
            pred_batch_size: Batch size used during prediction and
                embedding extraction.

        Raises:
            ValueError: If ``method_name`` is not ``"plain"`` or
                ``"ensemble"``.
        """
        if method_name not in {"plain", "ensemble"}:
            msg = f"BaselineALEstimator only supports 'plain' or 'ensemble', got {method_name!r}."
            raise ValueError(msg)
        self.method_name = method_name
        self.train_cfg = make_train_cfg(cfg)
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.device = device
        self.in_features = in_features
        self.num_members = num_members if method_name == "ensemble" else 1
        self.pred_batch_size = pred_batch_size
        self.batch_size = cfg.batch_size

        self.models: list[nn.Module] = []

    def _build_one(self) -> nn.Module:
        """Construct a single base model and move it to ``self.device``."""
        extra = {"in_features": self.in_features} if self.in_features is not None else {}
        model = models.get_base_model(self.base_model_name, self.num_classes, pretrained=False, **extra)
        model.to(self.device)
        return model

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> BaselineALEstimator:
        """Build fresh model(s) and train with vanilla cross-entropy.

        Args:
            x: Input features of shape ``(n, ...)``.
            y: Class labels of shape ``(n,)``.

        Returns:
            ``self``, with ``self.models`` populated.
        """
        x_t = x.to(dtype=torch.float32)
        y_t = y.to(dtype=torch.long)
        train_loader = DataLoader(TensorDataset(x_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.models = [self._build_one() for _ in range(self.num_members)]
        for member in self.models:
            train_cross_entropy(member, train_loader, self.train_cfg, self.device)
        return self

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities of shape ``(n, num_classes)``.

        For ``ensemble`` this is the mean of per-member softmax outputs.
        Output is on CPU.
        """
        if not self.models:
            msg = "Call fit() before predict_proba()."
            raise RuntimeError(msg)
        x_t = x.to(device=self.device, dtype=torch.float32)
        # Average over members: probs.shape = (n, num_classes); for plain, num_members=1.
        probs_sum: torch.Tensor | None = None
        for member in self.models:
            member.eval()
            parts: list[torch.Tensor] = []
            for start in range(0, len(x_t), self.pred_batch_size):
                logits = member(x_t[start : start + self.pred_batch_size])
                parts.append(F.softmax(logits, dim=-1).cpu())
            probs = torch.cat(parts)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        assert probs_sum is not None  # noqa: S101  -- num_members >= 1 by construction
        return probs_sum / float(len(self.models))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions of shape ``(n,)``."""
        return self.predict_proba(x).argmax(dim=-1)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer embeddings used by BADGE.

        For ``plain`` this hooks the last ``nn.Linear`` of the single model.
        For ``ensemble`` it averages per-member penultimate features.

        Args:
            x: Input tensor of shape ``(n, ...)``.

        Returns:
            Tensor of shape ``(n, emb_dim)`` on CPU.
        """
        if not self.models:
            msg = "Call fit() before embed()."
            raise RuntimeError(msg)
        x_t = x.to(device=self.device, dtype=torch.float32)
        if len(self.models) == 1:
            return embed_last_linear(self.models[0], x_t, self.pred_batch_size)
        embs = [embed_last_linear(m, x_t, self.pred_batch_size) for m in self.models]
        return torch.stack(embs).mean(dim=0)
