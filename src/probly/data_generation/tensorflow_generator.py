from typing import Dict, Any, Optional
import json
import tensorflow as tf

from .base_generator import BaseDataGenerator


class TensorFlowDataGenerator(BaseDataGenerator):
    """
    Data generator for TensorFlow/Keras models.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        super().__init__(model, dataset, batch_size, device)

        self.dataset = dataset.batch(batch_size)
        self.device = device or "cpu"
        self.results: Dict[str, Any] = {}

        print("TensorFlow DataGenerator initialized.")

    def generate(self) -> Dict[str, Any]:
        print("Generating model statistics with TensorFlow...")

        preds_all = []
        labels_all = []

        for i, (x, y) in enumerate(self.dataset):
            preds = self.model(x, training=False)
            preds_all.append(preds)
            labels_all.append(y)

            if (i + 1) % 5 == 0:
                print(f"Processed {(i + 1) * self.batch_size} samples")

        preds_all = tf.concat(preds_all, axis=0)
        labels_all = tf.concat(labels_all, axis=0)

        probs = tf.nn.softmax(preds_all, axis=1)
        pred_classes = tf.argmax(probs, axis=1)
        confidences = tf.reduce_max(probs, axis=1)

        accuracy = tf.reduce_mean(
            tf.cast(pred_classes == labels_all, tf.float32)
        ).numpy()

        self.results = {
            "info": {
                "framework": "tensorflow",
                "dataset_size": int(preds_all.shape[0]),
                "batch_size": self.batch_size,
            },
            "metrics": {
                "accuracy": float(accuracy),
            },
            "class_distribution": {
                "predicted": self._count(pred_classes.numpy()),
                "ground_truth": self._count(labels_all.numpy()),
            },
            "confidence": {
                "mean": float(tf.reduce_mean(confidences).numpy()),
                "std": float(tf.math.reduce_std(confidences).numpy()),
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
