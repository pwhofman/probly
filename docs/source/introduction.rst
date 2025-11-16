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

