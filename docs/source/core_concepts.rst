Core Concepts
=============

1. Understanding Uncertainty in Machine Learning
---------------------------------------------------

This section explains what uncertainty means in machine learning, why it naturally
arises in real-world problems, and why handling it correctly is essential for
building trustworthy models. Probly provides tools to work with uncertainty in a
structured and unified way.

1.1 What Is Uncertainty?
^^^^^^^^^^^^^^^^^^^^^^^^

In standard machine learning pipelines a model outputs a **single prediction**
a class label, a probability, or a regression value. However this number does
not tell us **how confident** the model actually is.

In machine learning, uncertainty refers to the **degree of confidence** a model
has in its outputs. There are two fundamental types :cite:`Hullermeier2021`:

**Epistemic Uncertainty**

Uncertainty caused by lack of knowledge or insufficient training data.
The model may never have seen anything similar before, for example
a rare medical anomaly or an unusual object in autonomous driving.
This uncertainty can be reduced with more or better data.

.. figure:: Epistemische.png
   :alt: Epistemic uncertainty illustration
   :width: 35%
   :align: center


**Aleatoric Uncertainty**

Uncertainty caused by noise in the data itself.
Labels may be ambiguous, sensors may be unreliable, or images may be blurry.
This uncertainty cannot be eliminated simply by collecting more data.

Most classical ML models such as neural networks or random forests
ignore both forms of uncertainty and return only a single output, often
leading to overconfident predictions.

.. figure:: Aleatorische.png
   :alt: Aleatoric uncertainty illustration
   :width: 35%
   :align: center

probly addresses this by offering unified tools to represent and quantify both
epistemic and aleatoric uncertainty across different methods.

1.2 Sources of Uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uncertainty appears naturally throughout the ML pipeline. Common sources include:

**Limited or Biased Training Data**

Small, imbalanced, or unrepresentative datasets cause poor generalization.
When the model encounters unfamiliar examples, predictions become unreliable.

**Out of Distribution Inputs**

Inputs that differ significantly from the training data, such as new environments,
novel objects, or corrupted images. Models often give confident but wrong
predictions for such samples.

**Label Noise and Ambiguity**

Human annotators may disagree or produce inconsistent labels.
Some domains, for example medicine or law, inherently contain subjective judgments.

**Model Architecture Limitations**

Certain architectures cannot express uncertainty well.
A deterministic network without any probabilistic layers, for example,
will always output a single best guess regardless of how unsure it is.

probly provides mechanisms to model all these uncertainty sources explicitly
instead of ignoring them, aligning with the common epistemic/aleatoric framing
:cite:`Hullermeier2021`.

1.3 Why Overconfidence Is a Problem?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modern ML models are often overconfident. They produce strong high probability
predictions even when they should be unsure. This causes serious issues in
real world systems. Misjudging whether uncertainty is epistemic or aleatoric
is a common driver of such overconfidence :cite:`Hullermeier2021`:

**Safety Critical Failures**

- A diagnostic model reporting 0.99 confidence despite being unsure.
- An autonomous vehicle misreading a rare obstacle but still reacting as if it were certain.

**Miscalibration**

The model’s predicted probabilities do not match reality.
For example predictions marked as 90 percent confident may be correct only
60 percent of the time.

**Poor Decision Making**

Downstream systems such as doctors, financial engines, or controllers
may rely on predictions that look certain but are actually unstable.

**Erosion of Trust**

Professionals and regulators increasingly require models not only to
provide predictions but also to communicate how reliable those
predictions are.

probly directly addresses these challenges by offering consistent tools to
express, compare, and act on model uncertainty, helping prevent dangerous
overconfidence.

2. Representing Uncertainty
-----------------------------

2.1 What Is an Uncertainty Representation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An uncertainty representation describes the form in which a machine-learning model
expresses not only its prediction but also its confidence in that prediction.
Instead of returning a single label such as “cat,” an uncertainty-aware model
produces additional information that reflects how sure or unsure it is about
its output.

Such representations can take many forms, including probability distributions,
repeated stochastic samples, raw logits, or evidence values for higher-level
distributions. In practice, they may appear as sets of sampled outputs, vectors
of class probabilities, parameters of a distribution, or structured intervals.
All of these formats serve the same purpose: they quantify how uncertain the
model is about its own prediction :cite:`sensoy2018evidential`.

2.2 Common Representation Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Dropout-based representations**

Dropout-based representations arise when a model is evaluated multiple times
with stochastic dropout activated. Each pass yields a slightly different output,
and the collection of these outputs represents the model’s uncertainty.

**Ensemble-based representations**

Ensemble-based representations rely on several independently trained models
whose predictions are combined; the variability across models expresses the
epistemic uncertainty.

**Evidential representations**

Evidential representations work by predicting the parameters of a higher-order
distribution rather than explicit samples. This allows the model to express
both a belief and uncertainty about that belief through a single forward pass.
This family of methods is closely related to evidential deep learning as
introduced by Sensoy et al. :cite:`sensoy2018evidential`.

**Bayesian sampling representations**

Bayesian sampling representations describe uncertainty by drawing samples from
distributions placed over the model’s weights, leading to sampled predictions
that approximate the full predictive distribution.

**Predictive distribution representations**

Predictive distribution representations output parameters of a probability
distribution or predictive intervals directly, allowing uncertainty to be
expressed in a compact parametric form.

2.3 Why Representations Must Be Unified
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different uncertainty methods produce outputs that vary widely in dimensionality,
structure, and meaning. Some provide many samples, others return explicit
distribution parameters, and others supply intervals or evidence values.
Without a unifying framework, these heterogeneous outputs cannot be compared
or processed consistently.

2.4 How probly Standardizes Representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Probly standardizes uncertainty by wrapping all forms of outputs into a single,
unified representation object. This object provides a consistent interface for
accessing samples, distribution parameters, evidence, or interval information,
regardless of the underlying method that produced them.

3. Quantifying and Using Uncertainty
------------------------------------

3.1 What is Uncertainty Quantification?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Models after being made uncertainty-aware can generate various forms of
uncertainty representations (e.g., samples, credal sets, distributions over
distributions). Uncertainty quantification means converting these representations
into numerical measures of uncertainty.

3.2 Why Quantification is Important
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uncertainty quantification is essential for making uncertainty-aware models
comparable, testable, and operable.

3.3 Downstream Tasks
^^^^^^^^^^^^^^^^^^^^

Quantified uncertainty enables OOD detection, selective prediction, calibration,
and risk-aware decision making.

3.4 How everything is connected
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**1. Model Transformation**
A standard ML model is transformed into an uncertainty-aware version using techniques
such as MC Dropout :cite:`Gal2016`, Bayesian Layers :cite:`Tran2019`, or Ensembles
:cite:`Lakshminarayanan2017`.

**2. Uncertainty Representation**
The resulting model produces samples, credal sets, interval predictions, or
distributions over distributions :cite:`Hullermeier2021`.

**3. Uncertainty Quantification**
From these structures, entropy, variance, or scoring-rule-based measures are computed
:cite:`Gneiting2007`, :cite:`Abdar2021`.

**4. Downstream Tasks & Visualization**
OOD detection, selective prediction, calibration, and risk-aware decisions depend
directly on quantified uncertainty.

See examples
^^^^^^^^^^^^

- :ref:`sphx_glr_auto_examples_plot_epistemic_vs_aleatoric.py`
- :ref:`sphx_glr_auto_examples_plot_uncertainty_quantification.py`
- :ref:`sphx_glr_auto_examples_plot_selective_prediction.py`
