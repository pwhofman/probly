from typing import Any, Optional

from .pytorch_generator import PyTorchDataGenerator
from .tensorflow_generator import TensorFlowDataGenerator
from .jax_generator import JAXDataGenerator


def create_data_generator(
    framework: str,
    model: Any,
    dataset: Any,
    batch_size: int = 32,
    device: Optional[str] = None,
):
    """
    Create a data generator based on the selected framework.
    """

    framework = framework.lower()

    if framework == "pytorch":
        return PyTorchDataGenerator(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
        )

    if framework == "tensorflow":
        return TensorFlowDataGenerator(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
        )

    if framework == "jax":
        return JAXDataGenerator(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
        )

    raise ValueError(f"Unknown framework: {framework}")
