Core Concepts
=============

1. Understanding Uncertainty in Machine Learning
---------------------------------------------

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
has in its outputs. There are two fundamental types:

**Epistemic Uncertainty**

Uncertainty caused by lack of knowledge or insufficient training data.
The model may never have seen anything similar before, for example
a rare medical anomaly or an unusual object in autonomous driving.
This uncertainty can be reduced with more or better data.

**Aleatoric Uncertainty**

Uncertainty caused by noise in the data itself.
Labels may be ambiguous, sensors may be unreliable, or images may be blurry.
This uncertainty cannot be eliminated simply by collecting more data.

Most classical ML models such as neural networks or random forests
ignore both forms of uncertainty and return only a single output, often
leading to overconfident predictions.

probly addresses this by offering unified tools to represent and quantify both
epistemic and aleatoric uncertainty across different methods.

1.2 Sources of Uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^

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
instead of ignoring them.

1.3 Why Overconfidence Is a Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modern ML models are often overconfident. They produce strong high probability
predictions even when they should be unsure. This causes serious issues in
real world systems:

**Safety Critical Failures**

- A diagnostic model reporting 0.99 confidence despite being unsure.
- An autonomous vehicle misreading a rare obstacle but still reacting as if it were certain.

**Miscalibration**

The model’s predicted probabilities do not match reality.
For example predictions marked as 90 percent confident may be correct only 60 percent of the time.

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
