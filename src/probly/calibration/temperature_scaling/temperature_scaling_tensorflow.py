"""Temperature scaling calibration model using TensorFlow."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


class TemperatureScaling(tf.keras.Model):
    """Applies temperature scaling to a trained tensorflow (Keras) model.

    This model wraps an existing classifier and calibrates its
    logits using a learned temperature parameter.
    """

    def __init__(self, model: tf.keras.Model) -> None:
        """Initialize the temperature scaling wrapper.

        Args:
            model (tf.keras.Model): Trained TensorFlow model to calibrate.
        """
        super().__init__()
        self.model = model
        self.temperature = tf.Variable(1.0, dtype=tf.float32, trainable=True)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass with temperature scaling applied.

        Args:
            inputs: Input tensor to the wrapped model.
            training: Whether the model is in training mode.

        Returns:
            Scaled logits tensor.
        """
        logits = self.model(inputs, training=training)
        return logits / self.temperature

    def set_temperature(
        self,
        val_x: tf.Tensor,
        val_labels: tf.Tensor,
        epochs: int = 1000,
        learning_rate: float = 0.01,
    ) -> TemperatureScaling:
        """Optimize the temperature parameter using validation data.

        Args:
            val_x: Validation input features.
            val_labels: Validation labels.
            epochs: retrain numbers.
            learning_rate: how fast it learns.
        """
        logits = self.model(val_x, training=False)
        labels = val_labels

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for _epoch in range(epochs):
            with tf.GradientTape() as tape:
                scaled_logits = logits / self.temperature
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        labels,
                        scaled_logits,
                        from_logits=True,
                    ),
                )
            grads = tape.gradient(loss, [self.temperature])
            optimizer.apply_gradients(zip(grads, [self.temperature], strict=False))
            if tf.abs(grads[0]) < 1e-6:
                break
        return self

    def predict(self, x: tf.Tensor, softed: bool = True) -> tf.Tensor | np.ndarray:
        """Make predictions using the calibrated model with optional temperature scaling.

        Args:
            x (tf.Tensor): Input data for which predictions are to be made.
            softed (bool, optional): If True, return probabilities using softmax;
                                    if False, return scaled logits. Defaults to True.

        Returns:
            tf.Tensor or np.ndarray: Softmax probabilities if `softed` is True,
                                    otherwise the logits scaled by temperature.
        """
        logits = self.model(x, training=False)
        scaled_logits = logits / self.temperature
        if softed:
            probs = tf.nn.softmax(scaled_logits, axis=-1)
            return probs
        return scaled_logits


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Compute the expected calibration error (ECE)."""
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    bins = np.linspace(0, 1, num_bins + 1, endpoint=True)
    bin_indices = np.digitize(confs, bins, right=True) - 1
    num_instances = probs.shape[0]
    ece = 0.0

    for i in range(num_bins):
        _bin = np.where(bin_indices == i)[0]
        if _bin.shape[0] == 0:
            continue
        acc_bin = np.mean(preds[_bin] == labels[_bin])
        conf_bin = np.mean(confs[_bin])
        weight = _bin.shape[0] / num_instances
        ece += weight * abs(acc_bin - conf_bin)

    return float(ece)
