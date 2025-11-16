Introduction
============

This page gives a gentle, high-level overview of what `probly` is and why 
uncertainty is an important topic in modern machine learning. It is meant as
a starting point for new users before they continue reading the rest of the User Guide, which 
covers everything from installation and quickstart examples to core concepts, main components,
advanced topics, and hands-on tutorials. The aim is to equip readers with the conceptual background needed
to understand and apply the more detailed material presented in later sections.

1. What is Probly?
---------------

`probly` is a Python library for **uncertainty-aware machine learning**.

In a typical machine-learning project you train a model (for example a neural
network or a random forest) and then use it to make predictions. Most models
only return a **single number or label**: a class, a score, a regression
value. However, in many applications we also want to know **how sure** the
model is about this prediction.

`probly` helps with exactly this. It provides:

* tools to turn standard models into **uncertainty-aware models**,
* a common interface to represent different kinds of uncertainty,
* functions to **quantify** uncertainty numerically, and
* utilities for downstream tasks such as out-of-distribution detection
  or selective prediction.

Instead of forcing you to use a completely new framework, `probly` is designed
to work together with existing libraries such as PyTorch, Flax/JAX and
scikit-learn. You can keep your usual training code and add uncertainty on top
of it with a thin `probly` wrapper.

1.1 The Problem: Overconfident Machine Learning Models
------------------------------------------------------

Most machine–learning models today are **overconfident**.  
They output a single prediction — a class label, a score, or a regression value —
but they do **not** tell us *how uncertain* they are about that prediction.

This becomes a problem in practical applications:

* A medical classifier might give the wrong diagnosis but still report a confident 0.98 probability.
* An autonomous vehicle might misinterpret a rare object because it has never seen anything similar before.
* A financial model may make predictions far outside the training distribution without realizing it.

Standard ML tools (e.g., PyTorch, TensorFlow, scikit-learn) do not provide a unified way
to represent uncertainty. Each method — dropout, ensembles, Bayesian networks,
evidential models, etc. — uses different outputs, shapes, and conventions.
This makes it difficult to compare or combine uncertainty methods.

**This is exactly the gap that `probly` fills.**

`probly` provides:

* **a common interface** for different uncertainty methods,  
* **a unified representation** for uncertainty-aware predictions,  
* **tools to quantify uncertainty** such as epistemic or aleatoric measures, and  
* **ready-to-use transformations** that turn existing models into uncertainty-aware models
  without rewriting the training pipeline.

Instead of forcing users to choose one uncertainty approach and rewrite their whole codebase,
`probly` offers a consistent way to apply uncertainty methods across modern ML frameworks
like PyTorch, Flax/JAX, and scikit-learn.

In short:  
**Machine-learning models are often overconfident — and Probly is designed to make
their uncertainty explicit, comparable, and usable.**
