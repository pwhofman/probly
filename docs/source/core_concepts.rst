Core Concepts
=============

1. Understanding Uncertainty in Machine Learning
---------------------------------------------

This section explains what uncertainty means in machine learning, why it naturally 
arises in real-world problems, and why handling it correctly is essential for 
building trustworthy models. Probly provides tools to work with uncertainty in a 
structured and unified way.

1.1 What Is Uncertainty?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
