from __future__ import annotations

import json
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .base_generator import BaseDataGenerator


class PyTorchDataGenerator(BaseDataGenerator):
    """Data generator for PyTorch models."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        batch_size: int = 32,
        device: str | None = None,
        num_workers: int = 0,
    ):
        super().__init__(model, dataset, batch_size, device)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {self.device}")

        self.model.to(self.device)
        self.model.eval()

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.results: dict[str, Any] = {}

    def generate(self) -> dict[str, Any]:
        print("Generating model statistics...")

        outputs_all = []
        labels_all = []

        with torch.no_grad():
            for i, (x, y) in enumerate(self.dataloader):
                x = x.to(self.device)
                out = self.model(x)

                outputs_all.append(out.cpu())
                labels_all.append(y.cpu())

                if (i + 1) % 5 == 0:
                    print(f"Processed {(i + 1) * self.batch_size} samples")

        outputs = torch.cat(outputs_all, dim=0)
        labels = torch.cat(labels_all, dim=0)

        probs = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(probs, dim=1)
        confidences = torch.max(probs, dim=1).values

        accuracy = (pred_classes == labels).float().mean().item()

        self.results = {
            "info": {
                "framework": "pytorch",
                "dataset_size": len(self.dataset),
                "batch_size": self.batch_size,
            },
            "metrics": {
                "accuracy": accuracy,
                "correct": int((pred_classes == labels).sum().item()),
            },
            "class_distribution": {
                "predicted": self._count(pred_classes),
                "ground_truth": self._count(labels),
            },
            "confidence": {
                "mean": float(confidences.mean()),
                "std": float(confidences.std()),
            },
        }

        return self.results

    def _count(self, tensor: torch.Tensor) -> dict[int, int]:
        counts = {}
        for v in tensor.tolist():
            v = int(v)
            counts[v] = counts.get(v, 0) + 1
        return counts

    def save(self, path: str) -> None:
        if not self.results:
            print("No results to save.")
            return

        try:
            with open(path, "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"Results saved to {path}")
        except Exception as e:
            print("Saving failed:", e)

    def load(self, path: str) -> dict[str, Any]:
        try:
            with open(path) as f:
                self.results = json.load(f)
            print(f"Results loaded from {path}")
            return self.results
        except Exception as e:
            print("Loading failed:", e)
            return {}
