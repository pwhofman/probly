.. _main_components:

****************
Main Components
****************
``probly`` is designed to bridge the gap between standard deterministic deep
learning and probabilistic modeling. Instead of requiring you to rewrite your
models from scratch to account for uncertainty, ``probly`` provides a modular
set of tools that integrate directly into your existing PyTorch or Flax workflows.

The library is built around three core pillars that mirror the uncertainty
estimation pipeline: **Transformation**, **Representation**,
and **Quantification**.

1. Transformations
========================


The core of ``probly`` lies in its **Transformations**. These functions take a standard, deterministic PyTorch
(or Flax) model and automatically modify its architecture to make it uncertainty-aware. Transformations work by
strategically modifying model layers, adding stochastic components, or duplicating architectures to create
ensembles—all without requiring manual code rewriting.

Each transformation method offers different trade-offs between computational cost, uncertainty quality, and
implementation complexity. Whether you need fast approximate uncertainty or more principled Bayesian inference,
``probly`` provides a transformation suited to your needs.

Below are the links to the Python notebooks describing the primary transformations available in the library:

1.1 Dropout
~~~~~~~~~~~~~~~~~~~~~~~

Dropout-based uncertainty leverages the interpretation of dropout as approximate Bayesian inference.
By enabling dropout at test time (Monte Carlo Dropout), you can generate multiple stochastic forward passes
to estimate model uncertainty.

:doc:`Dropout Transformation <notebooks/examples/dropout_transformation>`

1.2 DropConnect
~~~~~~~~~~~~~~~~~~~~~~~

DropConnect extends dropout by randomly dropping weights instead of activations, often providing improved
uncertainty estimates while maintaining computational efficiency.

:doc:`DropConnect Transformation <notebooks/examples/dropconnect_transformation>`

1.3 Bayesian
~~~~~~~~~~~~~~~~~~~~~~~

The Bayesian transformation converts deterministic weights to probabilistic distributions, enabling fully
principled Bayesian neural networks. This approach provides theoretically grounded uncertainty estimates but
at increased computational cost.

:doc:`Bayesian Transformation <notebooks/examples/bayesian_transformation>`

1.4 Ensemble
~~~~~~~~~~~~~~~~~~~~~~~

Ensemble transformations create multiple independent models trained on different data subsets or with
different initializations. By aggregating predictions across ensemble members, you obtain diverse uncertainty
estimates that often perform well in practice.

:doc:`Ensemble Transformation <notebooks/examples/ensemble_transformation>`

1.5 Evidential
~~~~~~~~~~~~~~~~~~~~~~~

Evidential methods model uncertainty by learning the parameters of a higher-order distribution
over predictions. Unlike dropout or ensemble methods that require multiple forward passes,
evidential models provide uncertainty estimates in a **single forward pass** by outputting
the parameters of a Dirichlet distribution (classification) or Normal-Inverse-Gamma distribution
(regression).

This approach provides a principled framework for capturing both **aleatoric uncertainty**
(inherent data noise) and **epistemic uncertainty** (model uncertainty due to limited data)
simultaneously :cite:`sensoyEvidentialDeep2018`.

1.5.1 Implemented Methods
----------------------------

``probly`` supports seven evidential deep learning methods through a unified interface:

**Classification Methods**

- **Evidential Deep Learning (EDL)** :cite:`sensoyEvidentialDeep2018` — The foundational
  method that trains networks to output Dirichlet concentration parameters directly.

- **Prior Networks (PrNet)** :cite:`malininPredictiveUncertaintyEstimation2018` — Uses both
  in-distribution and out-of-distribution data during training to learn sharper
  uncertainty estimates.

- **Information Robust Dirichlet Networks (IRD)** :cite:`tsiligkaridisInformationRobustDirichlet2019` —
  Adds adversarial robustness through entropy maximisation on perturbed inputs.

- **Posterior Networks (PostNet)** :cite:`charpentierPosteriorNetwork2020` — Uses normalizing flows
  to estimate density in latent space, converting density to pseudo-counts.

- **Natural Posterior Networks (NatPostNet)** :cite:`charpentierNaturalPosteriorNetwork2022` — Extends
  Posterior Networks to exponential family distributions with a certainty budget.

**Regression Methods**

- **Deep Evidential Regression (DER)** :cite:`aminiDeepEvidential2020` — Outputs parameters
  of a Normal-Inverse-Gamma distribution for regression with uncertainty.

- **Regression Prior Networks (RPN)** :cite:`malininRegressionPriorNetworks2020` — Combines
  evidential regression with OOD-aware training using KL divergence to a prior.


1.5.2 Unified Evidential Training
----------------------------------

The ``unified_evidential_train`` function in ``probly.train.evidential.common`` provides
a single interface for training all seven evidential methods:

.. code-block:: python

   from probly.train.evidential.common import unified_evidential_train
   from probly.train.evidential.torch import evidential_log_loss

   unified_evidential_train(
       mode="EDL",
       model=model,
       dataloader=train_loader,
       loss_fn=evidential_log_loss,
       epochs=5,
       lr=1e-3,
       device="cuda"
   )

**Function Signature**

.. code-block:: python

   def unified_evidential_train(
       mode: Literal["PostNet", "NatPostNet", "EDL", "PrNet", "IRD", "DER", "RPN"],
       model: nn.Module,
       dataloader: DataLoader,
       loss_fn: Callable = None,
       oodloader: DataLoader = None,
       flow: nn.Module = None,
       class_count: Tensor = None,
       epochs: int = 5,
       lr: float = 1e-3,
       device: str = "cpu",
   ) -> None:

**Parameters**

- ``mode``: Training approach identifier. One of: ``"PostNet"``, ``"NatPostNet"``, ``"EDL"``, ``"PrNet"``, ``"IRD"``, ``"DER"``, ``"RPN"``
- ``model``: The neural network to be trained (``nn.Module``)
- ``dataloader``: PyTorch DataLoader providing in-distribution training samples
- ``loss_fn``: Loss function for training (mode-dependent)
- ``oodloader``: DataLoader for out-of-distribution samples (required for ``"PrNet"``, ``"RPN"``)
- ``flow``: Normalizing flow module (required for ``"PostNet"``)
- ``class_count``: Tensor containing number of samples per class (used by ``"PostNet"``)
- ``epochs``: Number of training epochs (default: 5)
- ``lr``: Learning rate (default: 1e-3)
- ``device``: Device for training, e.g. ``"cpu"`` or ``"cuda"`` (default: ``"cpu"``)

**Mode-Specific Requirements**

+-------------+------------------+--------------------------------------------------+
| Mode        | Required Params  | Notes                                            |
+=============+==================+==================================================+
| ``EDL``     | loss_fn          | Basic evidential classification                  |
+-------------+------------------+--------------------------------------------------+
| ``IRD``     | loss_fn          | Generates adversarial examples internally        |
+-------------+------------------+--------------------------------------------------+
| ``NatPostNet``| loss_fn        | Model must return ``(alpha, z, log_pz)``         |
+-------------+------------------+--------------------------------------------------+
| ``PrNet``   | oodloader        | Uses ``pn_loss`` internally                      |
+-------------+------------------+--------------------------------------------------+
| ``PostNet`` | flow, class_count| Optimizes both model and flow parameters         |
+-------------+------------------+--------------------------------------------------+
| ``DER``     | —                | Uses ``der_loss`` internally; model returns 4 values |
+-------------+------------------+--------------------------------------------------+
| ``RPN``     | oodloader        | Uses ``rpn_loss`` internally                     |
+-------------+------------------+--------------------------------------------------+

**Example: Training with Prior Networks**

.. code-block:: python

   from probly.train.evidential.common import unified_evidential_train

   # PrNet requires both ID and OOD data
   unified_evidential_train(
       mode="PrNet",
       model=model,
       dataloader=id_train_loader,
       oodloader=ood_train_loader,
       epochs=10,
       lr=1e-3,
       device="cuda"
   )


1.5.3 Evidential Layers
------------------------

The ``probly.layers.evidential.torch`` module provides specialized layers for evidential models.

**RadialFlowDensity**

Normalizing flow for density estimation, used by Posterior Networks and Natural Posterior Networks.
Uses a stack of radial flow layers to transform a base Gaussian distribution:

.. code-block:: python

   from probly.layers.evidential.torch import RadialFlowDensity

   # Create flow with 4 radial layers over a 2D latent space
   flow = RadialFlowDensity(dim=2, flow_length=4)

   # Compute log probability of latent vectors
   log_prob = flow.log_prob(z)  # Shape: [B]

**NatPNClassifier**

Complete Natural Posterior Network classifier that combines an encoder, classifier head,
and normalizing flow for density-based uncertainty:

.. code-block:: python

   from probly.layers.evidential.torch import NatPNClassifier

   model = NatPNClassifier(
       num_classes=10,
       latent_dim=2,
       flow_length=4,
       certainty_budget=2.0,  # Scales density into evidence (default: latent_dim)
       n_prior=10.0           # Prior pseudo-count (default: num_classes)
   )

   # Forward pass returns three values
   alpha, z, log_pz = model(x)
   # alpha: Posterior Dirichlet parameters [B, num_classes]
   # z: Latent representation [B, latent_dim]
   # log_pz: Log density from flow [B]

**EvidentialRegression**

MLP model for evidential regression that outputs Normal-Inverse-Gamma parameters:

.. code-block:: python

   from probly.layers.evidential.torch import EvidentialRegression

   model = EvidentialRegression()

   # Returns four parameters for 1D regression
   mu, kappa, alpha, beta = model(x)
   # mu: Predicted mean
   # kappa: Observation count (>= 0, controls epistemic uncertainty)
   # alpha: Shape parameter (> 1)
   # beta: Rate parameter (> 0, controls aleatoric uncertainty)


1.5.4 Loss Functions
---------------------

Loss functions are available in ``probly.train.evidential.torch``.

**Classification Losses**

.. code-block:: python

   from probly.train.evidential.torch import (
       evidential_log_loss,      # EDL log loss (Sensoy et al., 2018)
       evidential_ce_loss,       # Cross-entropy variant using digamma
       evidential_mse_loss,      # MSE variant with variance term
       evidential_kl_divergence, # KL divergence regularizer
   )

   # Basic usage - inputs are raw model outputs, targets are class indices
   loss = evidential_log_loss(inputs, targets)

**ird_loss** :cite:`tsiligkaridisInformationRobustDirichlet2019`

Information Robust Dirichlet loss combining Lp calibration, regularization
on incorrect classes, and entropy maximization for adversarial robustness:

.. code-block:: python

   from probly.train.evidential.torch import ird_loss

   # alpha: predictions on clean inputs [B, K]
   # y: one-hot encoded labels [B, K]
   # adversarial_alpha: predictions on perturbed inputs [B, K] (optional)
   loss = ird_loss(
       alpha=alpha,
       y=y_onehot,
       adversarial_alpha=adversarial_alpha,
       p=2.0,      # Lp norm exponent
       lam=0.15,   # Regularization weight
       gamma=1.0   # Entropy weight
   )

**pn_loss** :cite:`malininPredictiveUncertaintyEstimation2018`

Prior Networks loss for paired ID and OOD training:

.. code-block:: python

   from probly.train.evidential.torch import pn_loss

   # Computes ID KL + OOD KL + cross-entropy term
   loss = pn_loss(model, x_in, y_in, x_ood)

**natpn_loss** :cite:`charpentierNaturalPosteriorNetwork2022`

Natural Posterior Network loss (expected NLL minus entropy regularization):

.. code-block:: python

   from probly.train.evidential.torch import natpn_loss

   loss = natpn_loss(alpha, y, entropy_weight=1e-4)

**postnet_loss** :cite:`charpentierPosteriorNetwork2020`

Posterior Networks loss using flow-based density estimation:

.. code-block:: python

   from probly.train.evidential.torch import postnet_loss

   loss, alpha = postnet_loss(z, y, flow, class_counts, entropy_weight=1e-5)

**Regression Losses**

**der_loss** :cite:`aminiDeepEvidential2020`

Deep Evidential Regression loss (Student-t NLL + evidence regularization):

.. code-block:: python

   from probly.train.evidential.torch import der_loss

   loss = der_loss(y, mu, kappa, alpha, beta, lam=0.01)

**rpn_loss** :cite:`malininRegressionPriorNetworks2020`

Regression Prior Networks loss combining DER on ID data with KL to prior on OOD:

.. code-block:: python

   from probly.train.evidential.torch import rpn_loss

   loss = rpn_loss(model, x_id, y_id, x_ood, lam_der=0.01, lam_rpn=50.0)

**Helper Functions**

.. code-block:: python

   from probly.train.evidential.torch import (
       kl_dirichlet,                # KL(Dir(α_p) || Dir(α_q))
       make_in_domain_target_alpha, # Sharp Dirichlet target for ID (α=10 on correct class)
       make_ood_target_alpha,       # Flat Dirichlet target for OOD
       rpn_prior,                   # Zero-evidence Normal-Gamma prior
       rpn_ng_kl,                   # KL divergence for Normal-Gamma distributions
       predictive_probs,            # Expected probabilities: α / Σα
   )


1.5.5 Uncertainty Interpretation
---------------------------------

**Classification (Dirichlet)**

For evidential classification, the model outputs Dirichlet concentration parameters
α = (α₁, ..., αₖ) where K is the number of classes.

**Key quantities:**

- **Dirichlet strength** S = Σαᵢ — total evidence; higher means more confident
- **Expected probabilities** p = α/S — the predicted class distribution
- **Epistemic uncertainty** K/S — uncertainty due to lack of evidence

.. code-block:: python

   from probly.train.evidential.torch import predictive_probs

   alpha = model(x)
   S = alpha.sum(dim=1)              # Total evidence
   probs = predictive_probs(alpha)   # Class probabilities
   uncertainty = num_classes / S     # Epistemic uncertainty

**Regression (Normal-Inverse-Gamma)**

For evidential regression, the model outputs (μ, κ, α, β):

- **Predicted mean**: μ
- **Aleatoric uncertainty** (data noise): β / (α - 1)
- **Epistemic uncertainty** (model uncertainty): β / (κ(α - 1))

.. code-block:: python

   mu, kappa, alpha, beta = model(x)

   aleatoric = beta / (alpha - 1)
   epistemic = beta / (kappa * (alpha - 1))


1.5.6 Tutorial Notebooks
-------------------------

Comprehensive tutorials are available in ``notebooks/examples/unified_evidential_train/``:

- ``unified_evidential_function_notebook.ipynb`` — Unified training function demo with EDL on MNIST, including OOD detection using FashionMNIST
- ``Prior_Networks.ipynb`` — Dirichlet Prior Networks (Malinin & Gales, 2018) with KL-based training, entropy analysis, and OOD detection AUC
- ``posterior_network.ipynb`` — Posterior Networks (Charpentier et al., 2020) using normalizing flows for density-based pseudo-counts without OOD training data
- ``Natural_Posterior_Network.ipynb`` — Natural Posterior Networks (Charpentier et al., 2022) with radial flows, Dirichlet posteriors, and certainty budget
- ``information_robust_dirichlet_networks_notebook.ipynb`` — IRD Networks (Tsiligkaridis, 2019) with Lp calibration, regularization, and adversarial entropy
- ``deep_evidential_regression_summary.ipynb`` — Deep Evidential Regression (Amini et al., 2020) with Normal-Inverse-Gamma distributions and aleatoric/epistemic uncertainty
- ``Regression_Prior_Networks (4).ipynb`` — Regression Prior Networks (Malinin et al., 2020) with Normal-Wishart distributions and unified DER+RPN loss

For the basic evidential transformation tutorials, see:

:doc:`Evidential Regression Transformation <notebooks/examples/evidential_regression_transformation>`


1.5.2 Evidential Classification
--------------------------------------
For classification, evidential classification learns a Dirichlet distribution over class probabilities,
enabling sophisticated uncertainty quantification for multi-class prediction tasks.

:doc:`Evidential Classification Transformation <notebooks/examples/evidential_classification_transformation>`


2. Utilities and Layers
================================


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
========================================

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
==============
By providing a comprehensive suite of tools across these three pillars—Transformations, Utilities & Layers,
and Evaluation & Quantification—``probly`` aims to make uncertainty estimation accessible and practical
for a wide range of machine learning applications, from research prototyping to production deployment.
