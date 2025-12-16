from __future__ import annotations

import tensorflow as tf

from probly.calibration.temperature_scaling.temperature_scaling_tensorflow import TemperatureScaling


class DummyLogitsModel(tf.keras.Model):
    """Simple deterministic model returning fixed logits."""

    def call(self, inputs: tf.Tensor, _training: bool = False) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        return tf.tile(tf.constant([[1.0, 2.0, 3.0]]), [batch_size, 1])


def test_call_applies_temperature_scaling() -> None:
    base_model = DummyLogitsModel()
    temp_model = TemperatureScaling(base_model)

    temp_model.temperature.assign(2.0)

    x = tf.zeros((2, 4))
    scaled_logits = temp_model(x)

    expected = tf.constant([[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]])
    tf.debugging.assert_near(scaled_logits, expected, atol=1e-6)


def test_predict_softmax_output() -> None:
    base_model = DummyLogitsModel()
    temp_model = TemperatureScaling(base_model)

    x = tf.zeros((3, 4))
    probs = temp_model.predict(x, softed=True)

    # Probabilities should sum to 1
    sums = tf.reduce_sum(probs, axis=1)
    tf.debugging.assert_near(sums, tf.ones_like(sums), atol=1e-6)

    # All probabilities must be in [0, 1]
    assert tf.reduce_all(probs >= 0.0)
    assert tf.reduce_all(probs <= 1.0)


def test_predict_logits_output() -> None:
    base_model = DummyLogitsModel()
    temp_model = TemperatureScaling(base_model)

    temp_model.temperature.assign(0.5)
    x = tf.zeros((1, 4))
    logits = temp_model.predict(x, softed=False)

    expected = tf.constant([[2.0, 4.0, 6.0]])
    tf.debugging.assert_near(logits, expected, atol=1e-6)


def test_set_temperature_updates_value() -> None:
    base_model = DummyLogitsModel()
    temp_model = TemperatureScaling(base_model)

    x_val = tf.zeros((10, 4))
    y_val = tf.zeros((10,), dtype=tf.int32)

    initial_temp = temp_model.temperature.numpy()

    temp_model.set_temperature(
        x_val,
        y_val,
        epochs=50,
        learning_rate=0.1,
    )

    updated_temp = temp_model.temperature.numpy()
    assert initial_temp != updated_temp
