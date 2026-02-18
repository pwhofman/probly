First-Order Data Generator - API Reference
=========================================



Table of Contents
------------------

- `Factory Pattern <#factory-pattern>`_
- `Framework-Specific Implementations <#framework-specific-implementations>`_
 - `PyTorch <#pytorch-firstorderdatagenerator>`_
 - `JAX <#jax-firstorderdatagenerator>`_
 - `Framework-Agnostic <#framework-agnostic-firstorderdatagenerator>`_
- `Base Generator <#base-generator>`_
- `Utility Functions <#utility-functions>`_
- `Data Types & Workflows <#data-types--workflows>`_

---


Factory Pattern
---------------



create_data_generator()
~~~~~~~~~~~~~~~~~~~~~~~


````python
def create_data_generator(
 framework: str,
 model: object,
 dataset: object,
 batch_size: int = 32,
 device: str | None = None,
) -> BaseDataGenerator
`````

Creates a framework-specific Data Generator.

**Parameters:**
- ``framework``: String - ``"pytorch"``, ``"tensorflow"`` or ``"jax"``
- ``model``: Model object (framework-specific)
- ``dataset``: Dataset object (framework-specific)
- ``batch_size``: Batch size for processing
- ``device``: Optional - Device string for inference

**Returns:**
- ``BaseDataGenerator``: Framework-specific generator instance

**Raises:**
- ``ValueError``: If framework is unknown

**Example:**
`````python
from probly.data_generator.factory import create_data_generator

Automatic Framework Selection
================================

generator = create_data_generator(
 framework='pytorch',
 model=my_torch_model,
 dataset=my_dataset,
 batch_size=32,
 device='cuda'
)
`````

---


Framework-Specific Implementations
---------------------------------------



PyTorch FirstOrderDataGenerator
-------------------------------



Class Description
~~~~~~~~~~~~~~~~~~~


`````python
@dataclass
class FirstOrderDataGenerator:
 """PyTorch-specific First-Order Data Generator."""
`````

Main class for PyTorch-based generation of First-Order distributions.


Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~


| Parameter | Type | Default | Description |
|-----------|-----|----------|--------------|
| ``model`` | ``torch.nn.Module \| Callable`` | **Required** | PyTorch Model or Callable |
| ``device`` | ``str`` | ``"cpu"`` | Device: ``"cpu"`` or ``"cuda"`` |
| ``batch_size`` | ``int`` | ``64`` | Batch size for inference |
| ``output_mode`` | ``str`` | ``"auto"`` | ``"auto"``, ``"logits"`` or ``"probs"`` |
| ``output_transform`` | ``Callable[[torch.Tensor], torch.Tensor] \| None`` | ``None`` | Custom Transform |
| ``input_getter`` | ``Callable[[Any], Any] \| None`` | ``None`` | Custom Input extraction |
| ``model_name`` | ``str \| None`` | ``None`` | Identifier for metadata |


Methods
~~~~~~~~



``generate_distributions()``
~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
@torch.no_grad()
def generate_distributions(
 self,
 dataset_or_loader: Dataset | DataLoader,
 *,
 progress: bool = True,
) -> dict[int, list[float]]
`````

Generates probability distributions for all samples.

**Parameters:**
- ``dataset_or_loader``: PyTorch ``Dataset`` or ``DataLoader``
- ``progress``: Bool - Enable progress bar

**Returns:**
- ``dict[int, list[float]]``: Index → Probability list

**Raises:**
- ``TypeError``: If model doesn't return a ``torch.Tensor``
- ``warnings.warn``: If number of distributions ≠ dataset length

**Example:**
`````python
generator = FirstOrderDataGenerator(model=model, device='cuda')
distributions = generator.generate_distributions(dataset, progress=True)
`````


``save_distributions()``
~~~~~~~~~~~~~~~~~~~~~~


`````python
def save_distributions(
 self,
 path: str | Path,
 distributions: Mapping[int, Iterable[float]],
 *,
 meta: dict[str, Any] | None = None,
) -> None
`````

Saves distributions as JSON.

**Parameters:**
- ``path``: Target path for JSON file
- ``distributions``: Distributions to save
- ``meta``: Optional metadata

**Example:**
`````python
generator.save_distributions(
 'output/dists.json',
 distributions,
 meta={'dataset': 'MNIST', 'accuracy': 0.95}
)
`````


``load_distributions()``
~~~~~~~~~~~~~~~~~~~~~~


`````python
def load_distributions(
 self,
 path: str | Path
) -> tuple[dict[int, list[float]], dict[str, Any]]
`````

Loads distributions from JSON.

**Returns:**
- ``tuple``: ``(distributions, metadata)``

**Example:**
`````python
dists, meta = generator.load_distributions('output/dists.json')
print(f"Model: {meta['model_name']}, Samples: {len(dists)}")
`````


``to_device()``
~~~~~~~~~~~~~


`````python
def to_device(self, x: object) -> object
`````

Moves tensor(s) to configured device. Supports nested structures (lists, tuples, dicts).


``to_probs()``
~~~~~~~~~~~~


`````python
def to_probs(self, outputs: torch.Tensor) -> torch.Tensor
`````

Converts model outputs to probabilities.
- Applies ``output_transform`` if present
- Otherwise: based on ``output_mode``
 - ``"auto"``: Detects automatically
 - ``"logits"``: Applies Softmax
 - ``"probs"``: Use directly


``prepares_batch_inp()`` / ``extract_input()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
def prepares_batch_inp(self, sample: object) -> object
`````

Extracts model input from dataset sample.
- Uses ``input_getter`` if present
- Otherwise: Unpacks Tuple/List ``(input, label, ...)``

**Note:** ``extract_input()`` is deprecated, use ``prepares_batch_inp()``


``get_posterior_distributions()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
def get_posterior_distributions(self) -> dict[str, dict[str, torch.Tensor]]
`````

Extracts μ and ρ from BayesLinear layers (for Bayesian Neural Networks).

**Returns:**
- ``dict``: Parameters with ``"mu"`` and ``"rho"`` tensors

---


FirstOrderDataset (PyTorch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
class FirstOrderDataset(Dataset):
 """PyTorch Dataset wrapper for First-Order distributions."""
`````

**Parameters:**
- ``base_dataset``: Original PyTorch Dataset
- ``distributions``: Index-aligned distributions
- ``input_getter``: Optional - Custom input extraction

**Methods:**
- ``__len__()``: Returns number of samples
- ``__getitem__(idx)``: Returns ``(input, label, distribution)`` or ``(input, distribution)``

**Example:**
`````python
fo_dataset = FirstOrderDataset(base_dataset, distributions)
input, label, dist = fo_dataset[0] # With labels
or
====

input, dist = fo_dataset[0] # Without labels
`````


output_dataloader() (PyTorch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
def output_dataloader(
 base_dataset: Dataset,
 distributions: Mapping[int, Iterable[float]],
 *,
 batch_size: int = 64,
 shuffle: bool = False,
 num_workers: int = 0,
 pin_memory: bool = False,
 input_getter: Callable[[Any], Any] | None = None,
) -> DataLoader
`````

Creates PyTorch DataLoader with First-Order distributions.

**Example:**
`````python
fo_loader = output_dataloader(
 base_dataset=dataset,
 distributions=distributions,
 batch_size=32,
 shuffle=True,
 num_workers=4,
 pin_memory=True
)

for inputs, labels, dists in fo_loader:
 # Training...
 pass
`````

---


JAX FirstOrderDataGenerator
---------------------------



Class Description
~~~~~~~~~~~~~~~~~~~


`````python
@dataclass
class FirstOrderDataGenerator:
 """JAX-native First-Order Data Generator."""
`````

JAX-specific implementation with jnp.ndarray support.


Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~


| Parameter | Type | Default | Description |
|-----------|-----|----------|--------------|
| ``model`` | ``Callable[..., Any]`` | **Required** | JAX-transformed function |
| ``device`` | ``str`` | ``"cpu"`` | Device: ``"cpu"``, ``"gpu"``, ``"tpu"`` |
| ``batch_size`` | ``int`` | ``64`` | Batch size |
| ``output_mode`` | ``str`` | ``"auto"`` | Output mode |
| ``output_transform`` | ``Callable[[jnp.ndarray], jnp.ndarray] \| None`` | ``None`` | Custom transform |
| ``input_getter`` | ``Callable \| None`` | ``None`` | Input extraction |
| ``model_name`` | ``str \| None`` | ``None`` | Identifier |


JAX-Specific Methods
~~~~~~~~~~~~~~~~~~~~~~~~



``to_device()``
~~~~~~~~~~~~~


`````python
def to_device(self, x: object) -> object
`````

Moves arrays to JAX Device using ``jax.device_put()``.

**Supported Devices:**
- Platform names: ``"cpu"``, ``"gpu"``, ``"tpu"``
- Specific IDs: ``"gpu:0"``, ``"tpu:1"``


``to_probs()``
~~~~~~~~~~~~


`````python
def to_probs(self, outputs: jnp.ndarray) -> jnp.ndarray
`````

Converts to probabilities using ``jax.nn.softmax()``.

**JAX-specific Features:**
- Uses ``jnp.ndarray`` instead of ``torch.Tensor``
- Automatic device placement
- Compatible with jax.jit and jax.vmap


``_batchify_inputs()``
~~~~~~~~~~~~~~~~~~~~


`````python
def _batchify_inputs(self, batch: Sequence[object]) -> jnp.ndarray
`````

Converts Python list to jnp.ndarray for batch processing.


JaxDataLoader
~~~~~~~~~~~~~


`````python
class JaxDataLoader:
 """Minimal JAX-friendly DataLoader."""
`````

**Parameters:**
- ``dataset``: DatasetLike with ``__len__`` and ``__getitem__``
- ``batch_size``: Batch size
- ``shuffle``: Bool - Random order

**Methods:**
- ``__len__()``: Number of batches
- ``__iter__()``: Iterator over batches (as lists)

**Example:**
`````python
from probly.data_generator.jax_first_order_generator import JaxDataLoader

loader = JaxDataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
 # batch is list of samples
 pass
`````


output_dataloader() (JAX)
~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
def output_dataloader(
 base_dataset: DatasetLike,
 distributions: Mapping[int, Iterable[float]],
 *,
 batch_size: int = 64,
 shuffle: bool = False,
 num_workers: int = 0, # Ignored - only for API compatibility
 pin_memory: bool = False, # Ignored
 input_getter: Callable[[Any], Any] | None = None,
) -> JaxDataLoader
`````

**Note:** ``num_workers`` and ``pin_memory`` are ignored (warns on non-default values).

---


Framework-Agnostic FirstOrderDataGenerator
------------------------------------------



Class Description
~~~~~~~~~~~~~~~~~~~


`````python
@dataclass
class FirstOrderDataGenerator:
 """Pure Python First-Order Generator (no framework required)."""
`````

Framework-independent implementation for maximum compatibility.


Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~


| Parameter | Type | Default | Description |
|-----------|-----|----------|--------------|
| ``model`` | ``Callable[..., Any]`` | **Required** | Callable model |
| ``device`` | ``str`` | ``"cpu"`` | Only for API symmetry (unused) |
| ``batch_size`` | ``int`` | ``64`` | Batch size |
| ``output_mode`` | ``str`` | ``"auto"`` | Output mode |
| ``output_transform`` | ``Callable \| None`` | ``None`` | Custom transform |
| ``input_getter`` | ``Callable \| None`` | ``None`` | Input extraction |
| ``model_name`` | ``str \| None`` | ``None`` | Identifier |


Pure Python Features
~~~~~~~~~~~~~~~~~~~~



``to_probs()``
~~~~~~~~~~~~


`````python
def to_probs(self, outputs: object) -> list[list[float]]
`````

Converts arbitrary outputs to Python lists of probabilities.

**Supports:**
- Numpy arrays
- Python Lists/Tuples
- Scalar values
- Framework-specific arrays (via conversion)

**Output:** ``list[list[float]]`` - 2D list of probabilities


``_to_batch_outputs()``
~~~~~~~~~~~~~~~~~~~~~


`````python
def _to_batch_outputs(outputs: object) -> list[list[float]]
`````

Normalizes various output shapes to ``list[list[float]]``.


``_softmax_row()``
~~~~~~~~~~~~~~~~


`````python
def _softmax_row(row: Sequence[float]) -> list[float]
`````

Pure Python Softmax implementation (no dependencies).


SimpleDataLoader
~~~~~~~~~~~~~~~~


`````python
class SimpleDataLoader:
 """Minimal Python DataLoader (no framework dependencies)."""
`````

**Parameters:**
- ``dataset``: DatasetLike with ``__len__`` and ``__getitem__``
- ``batch_size``: Batch size
- ``shuffle``: Bool - Random order

**Example:**
`````python
from probly.data_generator.first_order_datagenerator import SimpleDataLoader

loader = SimpleDataLoader(dataset, batch_size=32, shuffle=False)
for batch in loader:
 # batch is list of samples
 pass
`````

---


Base Generator
--------------



BaseDataGenerator
~~~~~~~~~~~~~~~~~


`````python
class BaseDataGenerator`M, D, Dev <ABC>`_:
 """Abstract base class for all Data Generators."""
`````

**Generics:**
- ``M``: Model type
- ``D``: Dataset type
- ``Dev``: Device type

**Abstract Methods:**
`````python
@abstractmethod
def generate(self) -> dict[str, Any]:
 """Run model and collect stats."""

@abstractmethod
def save(self, path: str) -> None:
 """Save results to file."""

@abstractmethod
def load(self, path: str) -> dict[str, Any]:
 """Load results from file."""
`````

**Implemented in:**
- ``PyTorchDataGenerator``: For PyTorch metrics
- ``TensorFlowDataGenerator``: For TensorFlow metrics
- ``JAXDataGenerator``: For JAX metrics

---


Utility Functions
---------------



_is_probabilities() (PyTorch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
def _is_probabilities(x: torch.Tensor, atol: float = 1e-4) -> bool
`````

Checks if tensor represents probabilities:
- All values in [0, 1]
- Rows sum to ~1.0 (within tolerance)


_is_probabilities() (JAX)
~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
def _is_probabilities(x: jnp.ndarray, atol: float = 1e-4) -> bool
`````

JAX version with same logic.


_is_prob_vector() (Framework-Agnostic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
def _is_prob_vector(v: Sequence[float], atol: float = 1e-4) -> bool
`````

Pure Python version for individual vectors.


_ensure_2d() (JAX)
~~~~~~~~~~~~~~~~~~


`````python
def _ensure_2d(x: jnp.ndarray) -> jnp.ndarray
`````

Ensures array is 2D (adds batch dimension if needed).


_get_device() (JAX)
~~~~~~~~~~~~~~~~~~~


`````python
def _get_device(device: str | None) -> jax.Device | None
`````

Converts device string to JAX Device object.

---


Data Types
----------



Distribution Dict
~~~~~~~~~~~~~~~~~


`````python
dict[int, list[float]]
`````

Mapping from dataset index to probability list.

**Example:**
`````python
{
 0: [0.1, 0.3, 0.6], # Sample 0
 1: [0.2, 0.5, 0.3], # Sample 1
 ...
}
`````

**Invariants:**
- Keys: Non-negative Integers
- Values: Lists of Floats in [0, 1]
- Sum(values) ≈ 1.0 (within tolerance)


Metadata Dict
~~~~~~~~~~~~~


`````python
dict[str, Any]
`````

Arbitrary metadata for saved distributions.

**Standard Fields:**
`````python
{
 'model_name': str, # Model name
 'dataset': str, # Dataset name
 'num_classes': int, # Number of classes
 'framework': str, # 'pytorch', 'jax', 'tensorflow'
 'accuracy': float, # Model accuracy
 'timestamp': str, # ISO format timestamp
}
`````


DatasetLike Protocol
~~~~~~~~~~~~~~~~~~~~


`````python
class DatasetLike(Protocol):
 def __len__(self) -> int: ...
 def __getitem__(self, idx: int) -> object: ...
`````

Minimal dataset interface for framework compatibility.

---


JSON File Format
----------------



Structure
~~~~~~~~


`````json
{
 "meta": {
 "model_name": "resnet50_v1",
 "dataset": "MNIST",
 "num_classes": 10,
 "framework": "pytorch",
 "accuracy": 0.95,
 "timestamp": "2025-01-11T10:30:00Z"
 },
 "distributions": {
 "0": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.17, 0.13, 0.10, 0.07],
 "1": [0.03, 0.05, 0.08, 0.12, 0.18, 0.21, 0.15, 0.10, 0.06, 0.02],
 "...": "..."
 }
}
`````


Properties
~~~~~~~~~~~~~


- **Encoding**: UTF-8
- **Format**: JSON with ``ensure_ascii=False``
- **Keys in distributions**: Strings (converted from int)
- **Values**: Lists of Floats
- **Cross-Framework**: All frameworks can read same JSON files

---


Typical Workflows
------------------



Workflow 1: PyTorch with GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
from probly.data_generator.torch_first_order_generator import (
 FirstOrderDataGenerator,
 output_dataloader
)

1. Create generator
======================

generator = FirstOrderDataGenerator(
 model=pretrained_model,
 device='cuda',
 output_mode='logits',
 batch_size=128
)

2. Generate distributions
==========================

distributions = generator.generate_distributions(dataset, progress=True)

3. Save
============

generator.save_distributions(
 'output/dists.json',
 distributions,
 meta={'dataset': 'CIFAR-10', 'accuracy': 0.92}
)

4. Load and Training
=====================

dists, meta = generator.load_distributions('output/dists.json')
fo_loader = output_dataloader(dataset, dists, batch_size=64, shuffle=True)

for inputs, labels, target_dists in fo_loader:
 # Training...
 pass
`````


Workflow 2: JAX on TPU
~~~~~~~~~~~~~~~~~~~~~~~


`````python
from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator
import jax

1. Generator with TPU
====================

generator = FirstOrderDataGenerator(
 model=jax_model_fn,
 device='tpu',
 output_mode='logits',
 batch_size=256
)

2. Generate
=============

distributions = generator.generate_distributions(jax_dataset)

3. Save (compatible with other frameworks)
================================================

generator.save_distributions('output/jax_dists.json', distributions)
`````


Workflow 3: Framework Switching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
Generate with PyTorch
======================

torch_gen = FirstOrderDataGenerator(model=torch_model, device='cuda')
dists = torch_gen.generate_distributions(torch_dataset)
torch_gen.save_distributions('dists.json', dists)

Load with JAX
=============

from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator
jax_gen = FirstOrderDataGenerator(model=jax_model)
loaded_dists, meta = jax_gen.load_distributions('dists.json')
Distributions are framework-independent!
=======================================

`````


Workflow 4: Custom Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
Custom Input Getter
===================

def get_image_and_metadata(sample):
 return sample['image'], sample['metadata']

Custom Output Transform (Label Smoothing)
=========================================

def label_smoothing(logits):
 probs = torch.softmax(logits, dim=-1)
 alpha = 0.1
 return (1 - alpha) * probs + alpha / logits.shape[-1]

generator = FirstOrderDataGenerator(
 model=model,
 input_getter=get_image_and_metadata,
 output_transform=label_smoothing,
 device='cuda'
)
`````

---


Error Handling
----------------



Typical Exceptions
~~~~~~~~~~~~~~~~~~~


| Exception | Framework | Reason | Solution |
|-----------|-----------|-------|--------|
| ``TypeError: Model must return a torch.Tensor`` | PyTorch | Wrong return type | Use ``output_transform`` |
| ``TypeError: Model must return a jnp.ndarray`` | JAX | Wrong return type | Ensure model returns jnp.ndarray |
| ``KeyError: No distribution for index X`` | All | Missing distribution | Check index alignment |
| ``ValueError: Invalid output_mode '...'`` | All | Invalid mode | Use 'auto', 'logits' or 'probs' |
| ``FileNotFoundError`` | All | JSON not found | Check path |
| ``json.JSONDecodeError`` | All | Invalid JSON | Check file format |


Warnings
~~~~~~~~


`````python
warnings.warn(
 "[FirstOrderDataGenerator] generated X distributions, but dataset length is Y."
)
`````

Occurs when number of distributions ≠ dataset length. Usually harmless (e.g. with ``drop_last=True``), but check index alignment.

`````python
warnings.warn(
 "[JAX output_dataloader] 'num_workers'=4 is ignored on JAX."
)
`````

JAX-specific: ``num_workers`` and ``pin_memory`` have no effect.

---


Compatibility
--------------



Framework Compatibility Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


| Feature | PyTorch | TensorFlow | JAX | Agnostic |
|---------|---------|------------|-----|----------|
| Distribution Generation | | | | |
| JSON Save/Load | | | | |
| Custom Transforms | | | | |
| DataLoader Integration | | | | |
| GPU Support | | | | N/A |
| TPU Support | | | | N/A |
| Auto-detection | | | | |


Python & Dependencies
~~~~~~~~~~~~~~~~~~~~~


- **Python**: >= 3.8 (Type hints required)
- **PyTorch**: >= 1.8.0 (for torch module)
- **TensorFlow**: >= 2.4.0 (for tensorflow module)
- **JAX**: >= 0.3.0 (for jax module)
- **Optional**: numpy (for agnostic module)

---


Performance Notes
--------------------



Batch Size
~~~~~~~~~~~


`````python
Small (32-64): Less GPU memory, slower
==============================================

generator = FirstOrderDataGenerator(model=model, batch_size=32)

Medium (64-128): Good balance
=============================

generator = FirstOrderDataGenerator(model=model, batch_size=128)

Large (256+): Faster, requires lots of GPU memory
=================================================

generator = FirstOrderDataGenerator(model=model, batch_size=256)
`````


DataLoader Optimization
~~~~~~~~~~~~~~~~~~~~~~


`````python
PyTorch
=======

fo_loader = output_dataloader(
 dataset, distributions,
 batch_size=64,
 num_workers=4, # CPU parallelization (not on Windows)
 pin_memory=True, # Faster GPU transfer
 shuffle=True
)

JAX (num_workers ignored)
===========================

fo_loader = output_dataloader(
 dataset, distributions,
 batch_size=64,
 shuffle=True # In-memory shuffle
)
`````


Memory Management
~~~~~~~~~~~~~~~~~


`````python
For large datasets: Batch-wise generation
============================================

distributions = {}
for batch_idx, batch in enumerate(dataloader):
 batch_dists = generator.generate_distributions(batch)
 distributions.update(batch_dists)

 if batch_idx % 10 == 0:
 # Periodic saving
 generator.save_distributions(f'checkpoint_{batch_idx}.json', distributions)
`````

---


Further Resources
------------------


- **Main Documentation**: ``docs/data_generation_guide.md``
- **Multi-Framework Guide**: ``docs/multi_framework_guide.md``
- **Examples**: ``examples/simple_usage.py``
- **Tests**: ``tests/test_*_first_order.py``
- **Source Code**: ``probly/data_generator/`
