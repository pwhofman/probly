=====================
First-Order Data Generator
=====================

A framework-agnostic Python tool for generating first-order data for evaluating credal sets and uncertainty quantification in machine learning.

Overview
=========

What is First-Order Data?
--------------------------

In uncertainty quantification with credal sets, we need ground-truth conditional distributions ``p(Y|X)`` to compute coverage metrics. These ground-truth distributions, also called first-order data, are typically not directly available.

This generator approximates ``p(Y|X)`` by:

1. Using a well pre-trained model (e.g., from Huggingface)
2. Transforming samples ``x`` to distributions ``ĥ(x) ≈ p(·|x)``
3. Storing and managing these distributions for training and evaluation

Problem and Solution
-------------------

**Problem:** Measuring coverage of credal sets

- Required: Ground-truth ``p(Y|X)``
- Available: Only data points ``(x, y)``

**Solution:** First-Order Data Generator

- Input: Pretrained Model + Dataset
- Output: Approximated conditional distributions
- Usage: Coverage evaluation, model training

Features
========

Multi-Framework Support
-----------------------

- **PyTorch**: Full integration with torch.nn.Module and DataLoader
- **TensorFlow/Keras**: Native tf.data.Dataset and tf.keras.Model support
- **JAX**: JAX-native implementation with jnp.ndarray
- **Framework-Agnostic**: Pure Python fallback for other frameworks

Core Features
--------------

- **Distribution Generation**: Creates ``ĥ(x) ≈ p(Y|x)`` for all samples
- **Flexible Output Handling**: Auto-detection of logits vs. probabilities
- **Efficient Storage**: JSON-based persistence with metadata
- **DataLoader Integration**: Seamless integration into training workflows
- **Customizable Processing**: User-defined input getters and output transforms

Installation
============

.. code-block:: bash

   # Base installation (framework-agnostic)
   pip install probly

   # With PyTorch support
   pip install probly[torch]

   # With TensorFlow support
   pip install probly[tensorflow]

   # With JAX support
   pip install probly[jax]

Quick Start
===========

Simplest Example (PyTorch)
-------------------------------

.. code-block:: python

   from probly.data_generator.torch_first_order_generator import FirstOrderDataGenerator
   import torch

   # Pretrained model
   model = torch.load('my_pretrained_model.pt')
   model.eval()

   # Dataset
   dataset = MyDataset()

   # Create generator and generate distributions
   generator = FirstOrderDataGenerator(
       model=model,
       device='cuda',
       output_mode='logits'
   )

   distributions = generator.generate_distributions(dataset, progress=True)

   # Save
   generator.save_distributions(
       'output/distributions.json',
       distributions,
       meta={'dataset': 'MNIST', 'accuracy': 0.95}
   )

Multi-Framework Example
-------------------------

.. code-block:: python

   from probly.data_generator.factory import create_data_generator

   # Automatic framework detection
   generator = create_data_generator(
       framework='pytorch',  # or 'tensorflow', 'jax'
       model=model,
       dataset=dataset,
       batch_size=32,
       device='cuda'
   )

   distributions = generator.generate_distributions(dataset)

Documentation
=============

Main Documentation
------------------

- **User Guide** - ``docs/data_generation_guide.rst`` - Comprehensive guide with examples
- **API Reference** - ``docs/api_reference.rst`` - Complete API documentation
- **Multi-Framework Guide** - ``docs/multi_framework_guide.rst`` - Framework-specific details

Examples
---------

- **Simple Usage** - ``examples/simple_usage.py`` - Basic PyTorch example
- **Tutorial Notebook** - ``examples/first_order_tutorial.ipynb`` - Interactive tutorial
- **Advanced Examples** - ``examples/`` - Additional use cases

Architecture
===========

Project Structure
---------------

.. code-block:: text

   probly/data_generator/
       base_generator.py         # Abstract base class
       factory.py                # Framework factory
       torch_first_order_generator.py    # PyTorch implementation
       first_order_datagenerator.py      # Framework-agnostic implementation
       jax_first_order_generator.py      # JAX implementation
       pytorch_generator.py      # PyTorch metrics generator
       tensorflow_generator.py   # TensorFlow metrics generator
       jax_generator.py          # JAX metrics generator

Design Patterns
---------------

.. code-block:: python

   # Abstract Base Class
   class BaseDataGenerator[M, D, Dev](ABC):
       """Framework-agnostic interface"""
       @abstractmethod
       def generate(self) -> dict[str, Any]: ...
       @abstractmethod
       def save(self, path: str) -> None: ...
       @abstractmethod
       def load(self, path: str) -> dict[str, Any]: ...

   # Framework-specific implementations
   class PyTorchDataGenerator(BaseDataGenerator[torch.nn.Module, Dataset, str]):
       """PyTorch-specific implementation"""

   class TensorFlowDataGenerator(BaseDataGenerator[tf.keras.Model, tf.data.Dataset, str]):
       """TensorFlow-specific implementation"""

   class JAXDataGenerator(BaseDataGenerator[object, tuple, str]):
       """JAX-specific implementation"""

Usage Examples
====================

1. PyTorch Classification
--------------------------

.. code-block:: python

   from probly.data_generator.torch_first_order_generator import FirstOrderDataGenerator
   from torch.utils.data import DataLoader

   # Initialize generator
   generator = FirstOrderDataGenerator(
       model=pretrained_model,
       device='cuda',
       batch_size=64,
       output_mode='logits',
       model_name='resnet50'
   )

   # Generate distributions
   distributions = generator.generate_distributions(dataset, progress=True)

   # Save with metadata
   generator.save_distributions(
       'output/dists.json',
       distributions,
       meta={
           'dataset': 'CIFAR-10',
           'num_classes': 10,
           'accuracy': 0.92
       }
   )

2. DataLoader and Training
---------------------------

.. code-block:: python

   from probly.data_generator.torch_first_order_generator import (
       FirstOrderDataset,
       output_dataloader
   )

   # Load
   dists, meta = generator.load_distributions('output/dists.json')

   # Create DataLoader with first-order distributions
   fo_loader = output_dataloader(
       base_dataset=dataset,
       distributions=dists,
       batch_size=32,
       shuffle=True,
       num_workers=4,
       pin_memory=True
   )

   # Training with soft targets (Knowledge Distillation)
   for inputs, labels, target_dists in fo_loader:
       logits = student_model(inputs)
       loss = F.kl_div(
           F.log_softmax(logits, dim=-1),
           target_dists,
           reduction='batchmean'
       )
       loss.backward()
       optimizer.step()

3. JAX Implementation
----------------------

.. code-block:: python

   from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator
   import jax.numpy as jnp

   # JAX-native generator
   generator = FirstOrderDataGenerator(
       model=jax_model_fn,
       device='gpu',
       batch_size=64,
       output_mode='logits'
   )

   # Generate distributions
   distributions = generator.generate_distributions(jax_dataset, progress=True)

   # Save (compatible with other frameworks)
   generator.save_distributions('output/jax_dists.json', distributions)

4. Framework-Agnostic (Pure Python)
------------------------------------

.. code-block:: python

   from probly.data_generator.first_order_datagenerator import FirstOrderDataGenerator

   # Works with any callable models
   def my_model(inputs):
       # Your model implementation
       return predictions

   generator = FirstOrderDataGenerator(
       model=my_model,
       batch_size=64,
       output_mode='logits'
   )

   distributions = generator.generate_distributions(dataset)

5. Custom Processing
-----------------------------------

.. code-block:: python

   # Custom input getter for complex data structures
   def custom_input_getter(sample):
       return sample['image']

   # Custom output transform (e.g., label smoothing)
   def custom_transform(logits):
       probs = torch.softmax(logits, dim=-1)
       alpha = 0.1
       n_classes = logits.shape[-1]
       return (1 - alpha) * probs + alpha / n_classes

   generator = FirstOrderDataGenerator(
       model=model,
       input_getter=custom_input_getter,
       output_transform=custom_transform,
       device='cuda'
   )

JSON File Format
================

Structure
--------

.. code-block:: json

   {
       "meta": {
           "model_name": "resnet50_v1",
           "dataset": "MNIST",
           "num_classes": 10,
           "accuracy": 0.95,
           "framework": "pytorch",
           "timestamp": "2025-01-11"
       },
       "distributions": {
           "0": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.17, 0.13, 0.10, 0.07],
           "1": [0.03, 0.05, 0.08, 0.12, 0.18, 0.21, 0.15, 0.10, 0.06, 0.02],
           "...": "..."
       }
   }

Properties
-------------

- **Encoding**: UTF-8
- **Format**: JSON with ``ensure_ascii=False``
- **Keys in distributions**: Strings (converted from int)
- **Values**: Lists of floats
- **Cross-Framework**: All frameworks can read the same JSON files

Configuration Options
======================

FirstOrderDataGenerator Parameters
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``model``
     - Callable
     - **Required**
     - Model function or nn.Module
   * - ``device``
     - str
     - ``"cpu"``
     - Device: ``"cpu"``, ``"cuda"``, ``"gpu"``, ``"tpu"``
   * - ``batch_size``
     - int
     - ``64``
     - Batch size for inference
   * - ``output_mode``
     - str
     - ``"auto"``
     - ``"auto"``, ``"logits"`` or ``"probs"``
   * - ``output_transform``
     - Callable|None
     - ``None``
     - Custom transform function
   * - ``input_getter``
     - Callable|None
     - ``None``
     - Custom input extraction
   * - ``model_name``
     - str|None
     - ``None``
     - Identifier for metadata

output_mode Explanation
---------------------

- **auto**: Automatically detects logits vs. probabilities

  - Checks if values are in [0,1] and sum to 1
  - Applies softmax if necessary

- **logits**: Always applies softmax
- **probs**: Uses outputs directly without transformation

Use Cases
===============

1. Credal Set Evaluation
------------------------

.. code-block:: python

   # Generate teacher distributions
   teacher_gen = FirstOrderDataGenerator(model=teacher_model)
   teacher_dists = teacher_gen.generate_distributions(test_set)

   # Evaluate student credal sets
   coverage = evaluate_credal_coverage(
       student_credal_sets=student_model.predict(test_set),
       ground_truth_dists=teacher_dists
   )
   print(f"Coverage: {coverage:.2%}")

2. Knowledge Distillation
--------------------------

.. code-block:: python

   # Generate teacher distributions
   teacher_dists = teacher_gen.generate_distributions(train_set)

   # Train student with soft targets
   student_loader = output_dataloader(train_set, teacher_dists, batch_size=64)
   train_with_soft_targets(student_model, student_loader, epochs=10)

3. Uncertainty Quantification
------------------------------

.. code-block:: python

   # Ensemble of models
   ensemble_dists = []
   for model in ensemble_models:
       gen = FirstOrderDataGenerator(model=model)
       dists = gen.generate_distributions(dataset)
       ensemble_dists.append(dists)

   # Analyze prediction variance
   uncertainty_scores = compute_prediction_entropy(ensemble_dists)

Best Practices
==============

Recommended Practices
--------------------

1. **Enable evaluation mode**

   .. code-block:: python

      model.eval()  # PyTorch
      # or model.trainable = False  # TensorFlow

2. **Consistent indexing**

   .. code-block:: python

      loader = DataLoader(dataset, batch_size=32, shuffle=False)
      # shuffle=False important for correct index alignment

3. **Document metadata**

   .. code-block:: python

      meta = {
          'model_architecture': 'ResNet-50',
          'dataset': 'CIFAR-10',
          'training_accuracy': 0.95,
          'timestamp': datetime.now().isoformat(),
          'notes': 'Pre-trained on ImageNet'
      }

4. **Plan storage space**

   - Approx. 1 MB per 10,000 samples with 10 classes
   - Approx. 10 MB per 10,000 samples with 100 classes

Performance Optimization
-----------------------

.. code-block:: python

   # For maximum performance
   generator = FirstOrderDataGenerator(
       model=model,
       device='cuda',
       batch_size=128  # Larger = faster (if GPU memory allows)
   )

   fo_loader = output_dataloader(
       dataset,
       distributions,
       batch_size=64,
       num_workers=4,  # Parallel loading (not on Windows)
       pin_memory=True,  # Faster GPU transfer
       shuffle=True
   )

Common Issues and Solutions
------------------------------

**Issue**: ``"Model must return a torch.Tensor"``

.. code-block:: python

   # Solution: Use output_transform
   def extract_logits(output):
       return output['logits'] if isinstance(output, dict) else output

   generator = FirstOrderDataGenerator(
       model=model,
       output_transform=extract_logits
   )

**Issue**: Distributions don't sum to 1.0

.. code-block:: python

   # Solution: Set output_mode correctly
   generator = FirstOrderDataGenerator(
       model=model,
       output_mode='logits'  # If your model outputs logits
   )

**Issue**: Warning about different lengths

.. code-block:: python

   # Usually harmless - happens with drop_last=True
   # If problematic: Check DataLoader configuration
   loader = DataLoader(dataset, batch_size=32, drop_last=False)

Tests
=====

.. code-block:: bash

   # Run all tests
   pytest tests/test_first_order_generator.py -v

   # PyTorch tests
   pytest tests/test_torch_first_order.py -v

   # JAX tests
   pytest tests/test_jax_first_order.py -v

   # TensorFlow tests
   pytest tests/test_tf_first_order.py -v

   # With coverage
   pytest tests/ --cov=probly.data_generator --cov-report=html

Additional Resources
==================

Documentation
-------------

- **Data Generation Guide** - ``docs/data_generation_guide.rst`` - Comprehensive tutorial
- **API Reference** - ``docs/api_reference.rst`` - Complete API documentation
- **Multi-Framework Guide** - ``docs/multi_framework_guide.rst`` - Framework-specific details
- **Architecture Overview** - ``docs/architecture.rst`` - System design

Examples and Tutorials
------------------------

- **Simple Usage** - ``examples/simple_usage.py`` - Basic example
- **Advanced Tutorial** - ``examples/advanced_usage.py`` - Advanced features
- **Jupyter Notebook** - ``examples/first_order_tutorial.ipynb`` - Interactive tutorial

Code
----

- **GitHub Repository** - https://github.com/your-org/probly - Source code
- **Issue Tracker** - https://github.com/your-org/probly/issues - Bugs and features

Compatibility
==============

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Framework
     - Version
     - Status
     - Notes
   * - PyTorch
     - >= 1.8.0
     - Full Support
     - Recommended
   * - TensorFlow
     - >= 2.4.0
     - Full Support
     - tf.data.Dataset
   * - JAX
     - >= 0.3.0
     - Full Support
     - jnp.ndarray
   * - Python
     - >= 3.8
     - Required
     - Type hints

**Devices:**

- CPU (all frameworks)
- CUDA/GPU (PyTorch, JAX, TensorFlow)
- TPU (JAX, TensorFlow)

License
======

Part of the ``probly`` project - see main repository for license information.
