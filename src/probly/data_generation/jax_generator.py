from typing import Dict, Any, Optional
import json
import jax
import jax.numpy as jnp

from .base_generator import BaseDataGenerator


class JAXDataGenerator(BaseDataGenerator):
    """
    Data generator for JAX models.
    """

    def __init__(
        self,
        model,          # JAX model function
        dataset,        # (x, y) as numpy or jax arrays
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        super().__init__(model, dataset, batch_size, device)

        self.device = device or "cpu"
        self.results: Dict[str, Any] = {}

        print("JAX DataGenerator initialized.")

    def generate(self) -> Dict[str, Any]:
        print("Generating model statistics with JAX...")

        x, y = self.dataset
        x = jnp.array(x)
        y = jnp.array(y)

        # forward pass (assume classification logits)
        logits = self.model(x)
        probs = jax.nn.softmax(logits, axis=1)

        pred_classes = jnp.argmax(probs, axis=1)
        confidences = jnp.max(probs, axis=1)

        accuracy = jnp.mean(pred_classes == y)

        self.results = {
            "info": {
                "framework": "jax",
                "dataset_size": x.shape[0],
                "batch_size": self.batch_size,
            },
            "metrics": {
                "accuracy": float(accuracy),
            },
            "class_distribution": {
                "predicted": self._count(pred_classes),
                "ground_truth": self._count(y),
            },
            "confidence": {
                "mean": float(jnp.mean(confidences)),
                "std": float(jnp.std(confidences)),
            },
        }

        return self.results

    def _count(self, values) -> Dict[int, int]:
        counts = {}
        for v in values.tolist():
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

    def load(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r") as f:
                self.results = json.load(f)
            print(f"Results loaded from {path}")
            return self.results
        except Exception as e:
            print("Loading failed:", e)
            return {}
