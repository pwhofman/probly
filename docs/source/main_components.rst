.. _main_components:

Main Components
==================
``probly`` is designed to bridge the gap between standard deterministic deep
learning and probabilistic modeling. Instead of requiring you to rewrite your
models from scratch to account for uncertainty, ``probly`` provides a modular
set of tools that integrate directly into your existing PyTorch or Flax workflows.

The library is built around three core pillars that mirror the uncertainty
estimation pipeline: **Transformation**, **Representation**,
and **Quantification**.

1. Transformations
----------------------


The core of ``probly`` lies in its **Transformations**. These functions take a standard, deterministic PyTorch
(or Flax) model and automatically modify its architecture to make it uncertainty-aware. Transformations work by
strategically modifying model layers, adding stochastic components, or duplicating architectures to create
ensembles—all without requiring manual code rewriting.

Each transformation method offers different trade-offs between computational cost, uncertainty quality, and
implementation complexity. Whether you need fast approximate uncertainty or more principled Bayesian inference,
``probly`` provides a transformation suited to your needs.

Below are the links to the Python notebooks describing the primary transformations available in the library:

1.1 Dropout
^^^^^^^^^^^^^^^^^^^^

Dropout-based uncertainty leverages the interpretation of dropout as approximate Bayesian inference.
By enabling dropout at test time (Monte Carlo Dropout), you can generate multiple stochastic forward passes
to estimate model uncertainty. For further information and a detailed explanation, refer to the notebook:
:doc:`Dropout Transformation <notebooks/examples/dropout_transformation>`

1.2 DropConnect
^^^^^^^^^^^^^^^^^^^^

DropConnect extends dropout by randomly dropping weights instead of activations, often providing improved
uncertainty estimates while maintaining computational efficiency. For further information and a detailed explanation, refer to the notebook:

:doc:`DropConnect Transformation <notebooks/examples/dropconnect_transformation>`

1.3 Bayesian
^^^^^^^^^^^^^^^^^^^^

The Bayesian transformation converts deterministic weights to probabilistic distributions, enabling fully
principled Bayesian neural networks. This approach provides theoretically grounded uncertainty estimates but
at increased computational cost. For further information and a detailed explanation, refer to the notebook:

:doc:`Bayesian Transformation <notebooks/examples/bayesian_transformation>`

1.4 Ensemble
^^^^^^^^^^^^^^^^^^^^

Ensemble transformations create multiple independent models trained on different data subsets or with
different initializations. By aggregating predictions across ensemble members, you obtain diverse uncertainty
estimates that often perform well in practice. For further information and a detailed explanation, refer to the notebook:

:doc:`Ensemble Transformation <notebooks/examples/ensemble_transformation>`

:doc:`Ensemble Transformation Comparison <notebooks/examples/transformations_comparison>`

:doc:`Ensemble SciKitLearn Transformation <notebooks/examples/sklearn_ensemble_tutorial>`

:doc:`Ensemble SciKitLearn Random Forests <notebooks/examples/Ensembling_RandomForests>`

1.4.1 BatchEnsemble
----------------------------

BatchEnsembles utilize the same key idea of Ensembles, training multiple networks to estimate uncertainty, but
with a different approach. While Ensembles in a classical sense are trained sequentially, BatchEnsembles are all trained
within a device at the same time. This is a achieved by modifying the input and output tensors by rank-one-vectors in each foward.
One can understand this as adding noise. Further and more detailed information can be either found in the paper
by :cite:`YemingWen2020`, from which this transformation is inspired, or by working through the provided Jupyter notebook.

Probly provides the user with a torch implementation to transfrom architectures with Linear and Conv2D layers in their
BatchEnsebmle counterparts. A Flax support is also available.


:doc:`BatchEnsemble Transformation <notebooks/examples/batch_ensemble_notebook>`

1.4.2 SubEnsemble
----------------------------

SubEnsemble are a computationally efficient alternative to standard ensembles. By sharing most of the model parameters across ensemble members
and only varying a small subset (e.g., the final layer), SubEnsembles achieve diverse predictions with significantly reduced memory and compute overhead.
:cite:`MatiasValdenegro-Toro2019`

:doc:`SubEnsemble Transformation <notebooks/examples/subensembles_fast_uncertainty>`

:doc:`Ensemble Subensemble Comparison <notebooks/examples/ensemble_subensemble_comparison>`

1.5 Evidential
^^^^^^^^^^^^^^^^^^^^

Evidential methods model uncertainty by learning the parameters of a higher-order distribution
over predictions. Unlike dropout or ensemble methods that require multiple forward passes,
evidential models provide uncertainty estimates in a **single forward pass** by outputting
the parameters of a Dirichlet distribution (classification) or Normal-Inverse-Gamma distribution
(regression).

This approach provides a principled framework for capturing both **aleatoric uncertainty**
(inherent data noise) and **epistemic uncertainty** (model uncertainty due to limited data)
simultaneously :cite:`sensoyEvidentialDeep2018`.

``probly`` supports seven evidential deep learning methods through a unified training interface,
covering both classification and regression tasks.

:doc:`Evidential Methods Documentation <evidential_methods>`

:doc:`Evidential Classification Transformation <notebooks/examples/evidential_classification_transformation>`

:doc:`Evidential Regression Transformation <notebooks/examples/evidential_regression_transformation>`


2. Utilities and Layers
--------------------------


Beyond transformations, ``probly`` offers a comprehensive suite of **Utilities and Layers** that facilitate
building and training uncertainty-aware models. These components are specifically designed to work seamlessly
with probabilistic outputs and enable end-to-end uncertainty-aware workflows.

**Key utilities include:**

- :doc:`Custom Loss Functions <notebooks/examples/utilities_and_layers/custom_loss_functions>`:

Tailored loss functions that properly account for
uncertainty in predictions. These include negative log-likelihood variants, evidential loss functions,and calibration-aware losses that ensure your model
learns meaningful uncertainty estimates.

- :doc:`Metrics <notebooks/examples/utilities_and_layers/metrics>`:

Specialized metrics to evaluate not only prediction accuracy but also the
quality of uncertainty estimates, including calibration error, sharpness, and proper scoring rules.

- :doc:`Probabilistic Layers <notebooks/examples/utilities_and_layers/probabilistic_layers>`:

Drop-in replacements for standard layers (Linear, Conv2D, etc.)
that incorporate stochasticity,enabling Bayesian inference within your models.

-  :doc:`Utility Functions <notebooks/examples/utilities_and_layers/utility_functions>`:

Helper functions for extracting mean and variance from model outputs,
computing prediction intervals, and formatting probabilistic predictions for downstream tasks.

These components ensure that users can not only modify their models to be uncertainty-aware but also
effectively train, evaluate, and deploy them within the same unified framework.


3. Evaluation and Quantification
-----------------------------------

Finally, ``probly`` provides comprehensive tools for **Evaluation and Quantification** of uncertainty estimates.
This includes rigorous methods to assess the calibration of uncertainty estimates, techniques to visualize
uncertainty in predictions, and approaches to interpret where uncertainty comes from.

These tools are essential for understanding how well a model's uncertainty estimates align with real-world outcomes,
validating that uncertainty is meaningful, and making informed decisions based on those estimates.

**Key evaluation tools include:**

- :doc:`Calibration Metrics <notebooks/examples/evaluation_and_quantification/calibration_metrics>`:

Tools to assess how well uncertainty estimates correspond to actual prediction errors. This includes expected calibration error (ECE), maximum calibration error (MCE),
and negative log-likelihood metrics.

- :doc:`Visualization Tools <notebooks/examples/evaluation_and_quantification/visualization_tools>`:

Methods to visualize uncertainty in predictions
through confidence bands, prediction interval plots, and uncertainty heatmaps, aiding in qualitative interpretation and model debugging.

- :doc:`Interpretation Techniques <notebooks/examples/evaluation_and_quantification/interpretation_techniques>`:

Approaches to decompose uncertaintyinto aleatoric (data) and epistemic (model) components,identify which inputs drive uncertainty, and understand the sources and implications of uncertainty in
model outputs.



**Conclusion**
-----------------
By providing a comprehensive suite of tools across these three pillars—Transformations, Utilities & Layers,
and Evaluation & Quantification—``probly`` aims to make uncertainty estimation accessible and practical
for a wide range of machine learning applications, from research prototyping to production deployment.
