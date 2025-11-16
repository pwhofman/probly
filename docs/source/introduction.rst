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

Most machine learning models today are overconfident.  
They output a single prediction such as a class label, a score, or a regression value,  
but they do not tell us how uncertain they are about that prediction.

This becomes a problem in practical applications:

* A medical classifier might give the wrong diagnosis but still report a confident 0.98 probability.  
* An autonomous vehicle might misinterpret a rare object because it has never seen anything similar before.  
* A financial model may make predictions far outside the training distribution without realizing it.

Standard ML tools such as PyTorch, TensorFlow, and scikit learn do not provide a unified way  
to represent uncertainty. Each uncertainty method such as dropout, ensembles, Bayesian networks,  
and evidential models uses different outputs, shapes, and conventions.  
This makes it difficult to compare or combine uncertainty methods.

This is exactly the gap that probly fills.

probly provides:

* a common interface for different uncertainty methods  
* a unified representation for uncertainty aware predictions  
* tools to quantify uncertainty such as epistemic or aleatoric measures  
* ready to use transformations that turn existing models into uncertainty aware models  
  without rewriting the training pipeline

Instead of requiring users to pick one uncertainty method and redesign their entire training code,  
probly offers a consistent way to apply uncertainty methods across modern ML frameworks  
such as PyTorch, Flax JAX, and scikit learn.

In short, machine learning models are often overconfident and probly is designed to make  
their uncertainty explicit, comparable, and usable.

1.2 Why Uncertainty Matters?
------------------------------------------------------

In many real world situations, the correctness of a prediction is not the only thing that matters.  
We also want to understand how confident the model is in its output.  
A prediction with low confidence should be treated very differently from the same prediction made with high confidence.

Uncertainty awareness is important for several reasons:

* It helps detect inputs that are very different from the training data, which is useful for out of distribution detection.  
* It allows systems to decide when a model should answer and when it should ask for human help.  
* It supports safer and more transparent decision making in fields such as medicine, finance, and autonomous systems.  
* It makes model behavior easier to interpret and debug by showing where the model is unsure.  

When a model communicates its uncertainty, users and downstream systems can make more careful choices.  
A low confidence prediction might trigger a safety mechanism, a manual review, or a fallback strategy.  
A high confidence prediction might allow the system to act automatically.

Without uncertainty information, a model can appear confident even when it is wrong.  
With uncertainty awareness, the model becomes more reliable, more transparent, and more useful in real world applications.

1.3 What Probly Provides?
------------------------------------------------------

probly is designed to make uncertainty aware machine learning simple, practical, and consistent.  
It provides a unified way to work with uncertainty so that users do not need to implement  
multiple methods from scratch or worry about incompatible output formats.

probly offers several core components:

* Representations that describe uncertainty information in a clear and structured way  
* Transformations that can turn standard models into uncertainty aware models with minimal changes  
* Quantification tools that compute numerical measures of uncertainty  
* Tasks that use uncertainty information for practical purposes such as out of distribution detection or selective prediction  

A key idea behind probly is that users should not be forced to adopt one specific uncertainty method.  
Different models and different applications may require different approaches.  
probly provides a common interface so that these methods can be used and compared in a consistent manner.

Another important advantage is that probly integrates with modern machine learning frameworks  
such as PyTorch, Flax JAX, and scikit learn.  
This means that users can keep their existing training pipelines and add uncertainty awareness on top of them  
without rewriting their entire code.

Overall, probly gives users the building blocks needed to make machine learning models more  
transparent, reliable, and informative by exposing how confident the model is in its predictions.



