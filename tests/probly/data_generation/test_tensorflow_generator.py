from __future__ import annotations

import numpy as np
import tensorflow as tf

from probly.data_generation.tensorflow_generator import TensorFlowDataGenerator


def create_simple_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(4,))])
    return model


def create_simple_dataset():
    rng = np.random.default_rng(0)
    x = rng.random((10, 4), dtype=np.float32)
    y = np.array([0, 1] * 5, dtype=np.int64)
    return tf.data.Dataset.from_tensor_slices((x, y))


def test_generate_basic():
    model = create_simple_model()
    dataset = create_simple_dataset()

    generator = TensorFlowDataGenerator(
        model=model,
        dataset=dataset,
        batch_size=2,
    )

    results = generator.generate()

    assert results is not None
    assert "metrics" in results
    assert "accuracy" in results["metrics"]


def test_save_and_load(tmp_path):
    model = create_simple_model()
    dataset = create_simple_dataset()

    generator = TensorFlowDataGenerator(model, dataset, batch_size=2)
    generator.generate()

    file_path = tmp_path / "results.json"
    generator.save(str(file_path))

    assert file_path.exists()

    new_generator = TensorFlowDataGenerator(model, dataset, batch_size=2)
    loaded_results = new_generator.load(str(file_path))

    assert loaded_results is not None
    assert "metrics" in loaded_results
