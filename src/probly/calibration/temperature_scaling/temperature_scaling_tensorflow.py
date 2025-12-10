import tensorflow as tf
from typing import Tuple
import numpy as np

class TemperatureScaling(tf.keras.Model): 
    # Initialize the temperature scaling wrapper
    def __init__(self, model): 
        super().__init__()
        self.model = model
        self.temperature = tf.Variable(1.0, dtype=tf.float32, trainable=True)

    # Runs a forward pass
    def call(self, inputs, training=False):
        logits = self.model(inputs, training=training)
        return logits/self.temperature

    # Find the optimal temperature using gradient descent on validation set
    def set_temperature(self, val_x, val_labels, epochs=1000, learning_rate=0.01): 
        
        
        logits = self.model(val_x, training=False)
        labels = val_labels


        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for epoch in range(epochs): 
            with tf.GradientTape() as tape: 
                scaled_logits = logits / self.temperature
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(labels, scaled_logits, from_logits=True)
                )
            grads = tape.gradient(loss, [self.temperature])
            optimizer.apply_gradients(zip(grads, [self.temperature]))
            if tf.abs(grads[0]) < 1e-6: 
                break

        print(f"Optimal Temperature: {self.temperature.numpy():.3f}")
        
        return self
        
    def predict(self, x, softed = True): 
        logits = self.model(x, training=False)
        scaled_logits = logits / self.temperature
        if softed: 
            probs = tf.nn.softmax(scaled_logits, axis=-1)
            return probs
        return scaled_logits
        
# Calculate ECE
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Compute the expected calibration error (ECE) of the predicted probabilities :cite:`guoOnCalibration2017`.

    Args:
        probs: The predicted probabilities as an array of shape (n_instances, n_classes).
        labels: The true labels as an array of shape (n_instances,).
        num_bins: The number of bins to use for the calibration error calculation.

    Returns:
        ece: The expected calibration error.
    """
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    bins = np.linspace(0, 1, num_bins + 1, endpoint=True)
    bin_indices = np.digitize(confs, bins, right=True) - 1
    num_instances = probs.shape[0]
    ece = 0
    for i in range(num_bins):
        _bin = np.where(bin_indices == i)[0]
        # check if bin is empty
        if _bin.shape[0] == 0:
            continue
        acc_bin = np.mean(preds[_bin] == labels[_bin])
        conf_bin = np.mean(confs[_bin])
        weight = _bin.shape[0] / num_instances
        ece += weight * np.abs(acc_bin - conf_bin)
    return float(ece)