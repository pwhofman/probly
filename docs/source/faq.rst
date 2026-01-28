.. _faq:

FAQ and Troubleshooting
=======================

This page provides answers to frequently asked questions about using ``probly`` and
solutions to common problems that users may encounter. It is organized into sections
covering installation issues, basic usage questions, uncertainty methods, integration
with different frameworks, and debugging tips.

If you cannot find an answer to your question here, please refer to the
:ref:`core_concepts` section for conceptual background, the :ref:`main_components`
section for detailed component descriptions, or the :ref:`examples_and_tutorials`
section for practical demonstrations.

1. Installation and Setup
--------------------------

1.1 Which Python versions does ``probly`` support?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``probly`` is designed to work with **Python 3.12 and above**.
If you are using an older Python version, you may encounter compatibility issues
with dependencies or type hints. We recommend upgrading to Python 3.12 or later.

For installation instructions, see :ref:`installation`.

1.2 How do I install ``probly``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can install ``probly`` using either ``pip`` or ``uv``:

.. code-block:: sh

   pip install probly

or

.. code-block:: sh

   uv add probly

For more details, refer to the :ref:`installation` section.

1.3 Installation fails with dependency conflicts. What should I do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dependency conflicts can occur when ``probly`` requires specific versions of libraries
that conflict with other packages in your environment. Here are some solutions:

**Create a clean virtual environment:**

.. code-block:: sh

   python -m venv probly_env
   source probly_env/bin/activate  # On Windows: probly_env\Scripts\activate
   pip install probly

**Use uv for dependency resolution:**

The ``uv`` package manager often handles dependency conflicts more effectively:

.. code-block:: sh

   uv venv
   uv pip install probly

**Check for conflicting packages:**

If you have existing PyTorch, JAX, or Flax installations, make sure they are compatible
with the versions required by ``probly``. You may need to upgrade or downgrade these packages.

1.4 Do I need to install PyTorch or JAX separately?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. ``probly`` integrates with PyTorch and Flax/JAX but does not install them automatically
as dependencies. This allows you to choose the appropriate version and configuration
(CPU or GPU) for your system.

Install PyTorch following the instructions at https://pytorch.org/, or install JAX
following https://github.com/google/jax#installation.

2. Basic Usage Questions
------------------------

2.1 How do I make my model uncertainty-aware?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``probly`` provides high-level transformation functions that wrap your existing model
to make it uncertainty-aware. The most common approach is to use one of the transformations
from the ``probly.transformation`` module.

Example using Monte Carlo Dropout :cite:`galDropoutBayesian2016`:

.. code-block:: python

   import probly
   from probly.representation.sampling import sampler_factory
   import numpy as np

   # Your trained model
   trained_model = ...

   # Step 1: Apply dropout transformation
   dropout_model = probly.transformation.dropout(trained_model, p=0.5)

   # Step 2: Create a sampler that generates multiple predictions
   sampler = sampler_factory(dropout_model, num_samples=10)

   # Step 3: Generate predictions (returns a list of tensors)
   predictions = sampler(input_data)

   # Step 4: Stack predictions into array for quantification
   # Shape will be (num_samples, batch_size, num_outputs)
   stacked_preds = np.stack([p.detach().numpy() for p in predictions])

For more details, see :ref:`introduction` section 3.2.

2.2 Do I need to retrain my model to use ``probly``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**No.** One of the key design principles of ``probly`` is that it works with models
you have already trained. You train your model exactly as usual, then apply a
``probly`` transformation to add uncertainty awareness during inference.

Some uncertainty methods, such as evidential networks or Bayesian neural networks,
do require specific training procedures, but even these can often be retrofitted
to existing architectures with minimal changes.

See :ref:`introduction` section 3 for the complete workflow.

2.3 What is an uncertainty representation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An uncertainty representation is a structured object that contains information about
how confident or uncertain a model is about its predictions. Instead of returning
a single prediction, the model returns additional information such as:

* Multiple stochastic samples (from dropout or ensembles)
* Distribution parameters (from evidential models)
* Probability intervals or credal sets

``probly`` unifies these different formats into a consistent interface so they can
be quantified and used in downstream tasks.

For a detailed explanation, see :ref:`core_concepts` section 2.

2.4 How do I quantify uncertainty from a representation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have predictions from a sampling-based uncertainty method, you can use the
quantification functions in ``probly.quantification`` to compute numerical uncertainty scores:

.. code-block:: python

   import numpy as np
   from probly.quantification import classification
   from probly.representation.sampling import sampler_factory

   # Create sampler from your model
   sampler = sampler_factory(dropout_model, num_samples=10)

   # Generate predictions (returns list of tensors)
   predictions = sampler(input_data)

   # Stack into numpy array: shape (num_samples, batch_size, num_classes)
   stacked_preds = np.stack([p.detach().numpy() for p in predictions])

   # Compute epistemic uncertainty using mutual information
   eu_scores = classification.mutual_information(stacked_preds)

   # Compute total entropy
   entropy_scores = classification.entropy(stacked_preds)

These scores can then be used for tasks such as out-of-distribution detection
or selective prediction.

See :ref:`core_concepts` section 3.1 for more quantification methods.

3. Uncertainty Methods
----------------------

3.1 Which uncertainty method should I use?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The choice of uncertainty method depends on your specific use case, computational
constraints, and the type of uncertainty you want to capture:

**Monte Carlo Dropout** :cite:`galDropoutBayesian2016`

* **Pros:** Easy to implement, works with any model that has dropout layers, computationally efficient
* **Cons:** May underestimate uncertainty, requires multiple forward passes
* **Use when:** You want a quick and simple way to add uncertainty to existing models

**Ensembles** :cite:`lakshminarayananSimpleScalable2017`

* **Pros:** Robust, well-calibrated, captures epistemic uncertainty effectively
* **Cons:** Requires training multiple models, higher memory and computation costs
* **Use when:** You have computational resources and need reliable uncertainty estimates

**Evidential Neural Networks** :cite:`sensoyEvidentialDeep2018,aminiDeepEvidential2020`

* **Pros:** Single forward pass, explicitly models higher-order uncertainty
* **Cons:** Requires specific training procedures and loss functions
* **Use when:** You need fast inference and can modify your training pipeline

**Bayesian Neural Networks** :cite:`blundellWeightUncertainty2015`

* **Pros:** Principled probabilistic framework, captures full posterior distribution
* **Cons:** Computationally expensive, requires specialized training
* **Use when:** You need theoretically grounded uncertainty and have computational resources

For conceptual background, see :ref:`core_concepts` and :ref:`introduction`.

3.2 What is the difference between epistemic and aleatoric uncertainty?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Epistemic uncertainty** (also called model uncertainty) reflects what the model does not know
because it has not seen similar data during training. This uncertainty can be reduced
by collecting more training data or improving the model.

For a theoretical foundation of this decomposition, see :cite:`depewegDecompositionUncertainty2018`.

**Aleatoric uncertainty** (also called data uncertainty) reflects inherent noise or ambiguity
in the data itself, such as sensor noise, label disagreements, or inherently ambiguous cases.
This uncertainty cannot be reduced by simply collecting more data.

``probly`` provides tools to quantify both types of uncertainty. For example,
mutual information captures epistemic uncertainty, while total entropy includes both types.

See :ref:`core_concepts` section 1.1 for detailed explanations with visualizations.

3.3 Can I combine different uncertainty methods?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. ``probly`` is designed so that different uncertainty methods share a common interface.
This means you can:

* Compare different methods on the same dataset
* Use ensemble-based and dropout-based uncertainty in parallel
* Switch between methods without changing your downstream analysis code

The unified representation format makes it easy to experiment with different approaches
and choose the one that works best for your application.

4. Framework Integration
------------------------

4.1 Does ``probly`` work with PyTorch?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. ``probly`` has full support for PyTorch models. You can wrap any ``torch.nn.Module``
with a ``probly`` transformation to make it uncertainty-aware.

Example:

.. code-block:: python

   import torch
   import probly

   # Your PyTorch model
   model = torch.nn.Sequential(
       torch.nn.Linear(784, 256),
       torch.nn.ReLU(),
       torch.nn.Linear(256, 10)
   )

   # Apply transformation
   mc_dropout_model = probly.transformation.dropout(model, p=0.5)

4.2 Does ``probly`` work with Flax/JAX?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. ``probly`` supports Flax/JAX models through the same transformation interface.
You can apply uncertainty transformations to ``flax.nnx.Module`` objects.

See :ref:`introduction` section 5 for supported frameworks.

4.3 Can I use ``probly`` with scikit-learn models?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``probly`` is primarily designed for neural network frameworks like PyTorch and Flax/JAX.
However, some uncertainty quantification functions can work with probability outputs
from scikit-learn models if they are formatted correctly.

For full integration, we recommend using neural network-based models.

4.4 How do I use ``probly`` with pre-trained models?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can apply ``probly`` transformations to any pre-trained model, whether you trained it yourself
or loaded it from a model zoo. Simply pass the pre-trained model to a transformation function:

.. code-block:: python

   import torchvision.models as models
   import probly

   # Load pre-trained ResNet
   pretrained_model = models.resnet18(pretrained=True)

   # Apply dropout transformation
   uncertain_model = probly.transformation.dropout(pretrained_model, p=0.3)

5. Common Errors and Solutions
-------------------------------

5.1 Error: "Cannot find module 'probly'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This error means ``probly`` is not installed in your current Python environment.

**Solution:**

Make sure you have activated the correct virtual environment and installed ``probly``:

.. code-block:: sh

   pip install probly

If you are using Jupyter notebooks, ensure the notebook kernel matches your virtual environment.

5.2 Error: "Shape mismatch in uncertainty quantification"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This error occurs when the array passed to quantification has an unexpected shape.

**Solution:**

The quantification functions expect a numpy array with shape ``(num_samples, batch_size, num_classes)``
for classification tasks. Make sure you are stacking predictions correctly:

.. code-block:: python

   from probly.quantification import classification
   import numpy as np

   # If you have a list of predictions from sampler_factory:
   predictions = sampler(input_data)  # Returns list of tensors

   # Stack into correct shape
   stacked = np.stack([p.detach().numpy() for p in predictions])
   print(stacked.shape)  # Should be (num_samples, batch_size, num_classes)

   # Now quantification will work
   mi_scores = classification.mutual_information(stacked)

Make sure the first dimension is the number of samples, not the batch size.

5.3 Warning: "Model returns deterministic output"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This warning appears when a transformation that requires stochastic behavior
(like dropout) produces identical outputs across multiple forward passes.

**Solution:**

* Make sure dropout is enabled during inference
* Check that your model actually contains dropout layers
* Verify that the transformation was applied correctly

Example:

.. code-block:: python

   # Incorrect: dropout disabled during eval
   model.eval()
   representation = mc_dropout_model(input_data)  # All outputs identical!

   # Correct: keep model in training mode for MC Dropout
   model.train()
   representation = mc_dropout_model(input_data)

5.4 Error: "Out of memory during ensemble prediction"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensemble methods require running multiple models simultaneously, which can consume
significant GPU memory.

**Solution:**

* Reduce batch size
* Process ensemble members sequentially instead of in parallel
* Use gradient checkpointing if available
* Consider using a smaller number of ensemble members

6. Performance and Optimization
--------------------------------

6.1 How many forward passes should I use for Monte Carlo Dropout?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of forward passes (samples) is a trade-off between accuracy and computational cost:

* **10-30 samples:** Good starting point for most applications
* **50-100 samples:** Better uncertainty estimates, higher computational cost
* **100+ samples:** Diminishing returns, use only if very precise estimates are needed

You can experiment with different numbers of samples and evaluate using uncertainty
calibration metrics.

6.2 Is ``probly`` slow compared to standard inference?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uncertainty-aware inference is inherently more expensive than standard inference because:

* Monte Carlo Dropout requires multiple forward passes
* Ensembles require multiple models
* Bayesian methods involve sampling procedures

However, ``probly`` is designed to be as efficient as possible within these constraints.
If speed is critical, consider:

* Using fewer samples for Monte Carlo methods
* Using evidential networks (single forward pass)
* Batching uncertainty computations
* Using GPU acceleration

6.3 How can I speed up uncertainty quantification?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Use vectorized operations:**

``probly`` quantification functions are implemented with vectorized operations
and work efficiently on batched data.

**Reduce the number of samples:**

If using Monte Carlo methods, try using fewer samples during development and
increase only for final evaluation.

**Use appropriate hardware:**

Move computations to GPU if available:

.. code-block:: python

   import torch

   model = model.to('cuda')
   input_data = input_data.to('cuda')

7. Troubleshooting Advanced Features
------------------------------------

7.1 Frequent Issues and Error Messages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Custom Transformations**

When implementing custom uncertainty transformations, users may encounter:

* **Type errors:** Ensure your custom transformation returns the expected representation format
* **Shape mismatches:** Verify that output dimensions match the expected uncertainty representation
* **Integration issues:** Check that the transformation is compatible with the base model framework

**Large Models**

Working with large models introduces specific challenges:

* **Memory errors during ensemble creation:** Large models multiplied across ensemble members can exceed GPU memory

  **Solution:** Use gradient checkpointing, reduce batch size, or process ensemble members sequentially

* **Slow inference with MC Dropout:** Multiple forward passes on large models can be time-consuming

  **Solution:** Reduce the number of samples, use mixed precision, or consider single-pass methods like evidential networks

**Integration with Flax/TensorFlow/scikit-learn**

* **Flax/JAX compatibility:** Ensure you're using compatible JAX and Flax versions (JAX ≥0.8.0, Flax ≥0.12.0)

  **Solution:** Check version compatibility in your environment and update if needed

* **TensorFlow models:** ``probly`` primarily supports PyTorch and Flax/JAX. For TensorFlow, you may need to convert models or use probability outputs directly

* **scikit-learn integration:** While ``probly`` is designed for neural networks, some quantification functions can work with probability outputs from scikit-learn classifiers if properly formatted

**Performance Problems**

* **Slow uncertainty quantification:** Vectorized operations are optimized, but large batch sizes or many samples can still be slow

  **Solution:** Profile your code to identify bottlenecks, reduce sample counts during development, use GPU acceleration

* **High memory usage:** Storing multiple samples from ensemble or MC Dropout methods requires significant memory

  **Solution:** Process in smaller batches, use streaming quantification where possible, or reduce the number of samples

7.2 Systematic Debugging Approach
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When encountering issues with ``probly``, follow this systematic approach to isolate problems:

**Step 1: Reduce Model Complexity**

Start with a minimal model to verify the transformation works:

.. code-block:: python

   import torch
   import probly

   # Create a simple model
   simple_model = torch.nn.Sequential(
       torch.nn.Linear(10, 5),
       torch.nn.ReLU(),
       torch.nn.Linear(5, 2)
   )

   # Test transformation
   dropout_model = probly.transformation.dropout(simple_model, p=0.3)

   # Verify it works
   test_input = torch.randn(4, 10)
   output = dropout_model(test_input)
   print("Simple model works:", output.shape)

**Step 2: Use Smaller Data**

Test with a small synthetic dataset before using your full data:

.. code-block:: python

   import numpy as np
   from probly.representation.sampling import sampler_factory

   # Small synthetic data
   small_data = torch.randn(10, 10)

   # Test sampler
   sampler = sampler_factory(dropout_model, num_samples=5)
   predictions = sampler(small_data)

   # Verify output format
   stacked = np.stack([p.detach().numpy() for p in predictions])
   print("Predictions shape:", stacked.shape)  # Should be (5, 10, 2)

**Step 3: Disable Features Incrementally**

If using multiple features, disable them one by one to identify the problematic component:

* Remove custom transformations
* Use fewer samples
* Simplify quantification metrics
* Test on CPU before GPU

**Step 4: Distinguish Transformation vs. Integration Issues**

**Transformation issues** typically manifest as:

* Incorrect output shapes
* Deterministic outputs when stochastic behavior is expected
* Type errors when calling transformation functions

**Integration issues** typically manifest as:

* Framework-specific errors (PyTorch vs. Flax)
* Incompatibility with model architectures
* Device placement errors (CPU vs. GPU)

**Debugging example:**

.. code-block:: python

   # Test if issue is with transformation or integration

   # 1. Test transformation directly
   transformed = probly.transformation.dropout(model, p=0.5)
   out1 = transformed(test_input)
   out2 = transformed(test_input)

   # Should be different if dropout is working
   print("Outputs differ:", not torch.allclose(out1, out2))

   # 2. Test integration with sampler
   from probly.representation.sampling import sampler_factory
   sampler = sampler_factory(transformed, num_samples=3)
   samples = sampler(test_input)

   # Should get list of 3 different outputs
   print("Got", len(samples), "samples")

7.3 Getting Help
^^^^^^^^^^^^^^^^

**What Information to Include in Bug Reports**

When reporting bugs or asking for help, include:

1. **Environment details:**

   * Python version
   * ``probly`` version
   * Framework versions (PyTorch/JAX/Flax)
   * Operating system

2. **Minimal reproducible example:**

   * Simplest code that demonstrates the issue
   * Sample data or synthetic data that triggers the problem
   * Expected vs. actual behavior

3. **Error messages:**

   * Complete stack trace
   * Any warning messages
   * Console output

**Example bug report:**

.. code-block:: text

   **Environment:**
   - Python 3.12.1
   - probly 0.1.0
   - PyTorch 2.1.0
   - Ubuntu 22.04

   **Issue:**
   Getting shape mismatch when using mutual_information with dropout predictions


   **Code:**

.. code-block:: python

   import probly
   import torch

   model = torch.nn.Linear(10, 3)
   dropout_model = probly.transformation.dropout(model, p=0.5)
   # ... rest of minimal example


   **Error:**

.. code-block:: text

   ValueError: Shape mismatch in mutual_information...


**Where to Get Help**

* **GitHub Issues:** Report bugs and request features at https://github.com/pwhofman/probly/issues
* **FAQ & Troubleshooting:** Check this document for common solutions
* **Documentation:** Refer to :ref:`core_concepts`, :ref:`introduction`, and :ref:`examples_and_tutorials`
* **Community:** Discuss with other users through the GitHub issue tracker

8. Advanced Topics
------------------

8.1 Can I use custom uncertainty quantification metrics?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. If you have a custom metric, you can implement it as a function that takes
a stacked numpy array of predictions and returns numerical scores.

Example:

.. code-block:: python

   import numpy as np

   def custom_uncertainty_metric(stacked_predictions):
       # Your custom metric implementation
       # stacked_predictions shape: (num_samples, batch_size, num_classes)
       # Example: compute variance across samples
       return np.var(stacked_predictions, axis=0).mean(axis=1)

   # Use it
   from probly.representation.sampling import sampler_factory

   sampler = sampler_factory(model, num_samples=10)
   predictions = sampler(input_data)
   stacked = np.stack([p.detach().numpy() for p in predictions])

   custom_scores = custom_uncertainty_metric(stacked)

8.2 How do I integrate ``probly`` into a production system?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For production deployment:

* **Optimize inference:** Reduce the number of samples or use single-pass methods like evidential networks
* **Batch processing:** Process multiple inputs together for efficiency
* **Uncertainty thresholds:** Define application-specific thresholds for rejection or alerts
* **Monitoring:** Log uncertainty scores alongside predictions for analysis
* **Fallback strategies:** Define what happens when uncertainty is too high

8.3 Where can I find more examples?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For detailed usage examples, refer to:

* The :ref:`introduction` section for workflow examples
* The :ref:`installation` section for quickstart code
* The notebooks in the ``notebooks/examples/`` directory of the repository

9. Getting Help
---------------

9.1 Where can I report bugs or request features?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please report bugs and feature requests on the ``probly`` GitHub repository:

https://github.com/pwhofman/probly/issues

Include:

* Python and ``probly`` versions
* Minimal code to reproduce the issue
* Expected vs. actual behavior
* Any error messages or stack traces

9.2 How can I contribute to ``probly``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We welcome contributions! Please see the Contributing Guide for details on:

* Setting up a development environment
* Code style and conventions
* Submitting pull requests
* Adding new uncertainty methods or quantification functions

9.3 Where can I discuss ``probly`` with other users?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Join the ``probly`` community:

* Issue tracker for questions: https://github.com/pwhofman/probly/issues

For questions about uncertainty quantification in general, the broader machine learning
community resources may also be helpful.
