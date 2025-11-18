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
to work together with existing libraries such as PyTorch, and Flax/JAX. You can keep your usual training code and add uncertainty on top
of it with a thin `probly` wrapper.

1.1 The Problem: Overconfident Machine Learning Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

2. Key Ideas Behind Probly
------------------------------------------------------

probly is built around a few central ideas that make uncertainty aware machine learning easier to use.  
Instead of treating uncertainty as something separate or difficult, probly organizes the process into  
clear steps that work together and build on one another.

These ideas form the foundation of the entire library.  
They help users understand how uncertainty is represented, how it is created,  
and how it can be used to solve practical machine learning problems.

The main ideas are:

* Representations that describe uncertainty information  
* Transformations that create uncertainty aware models  
* Quantification tools that measure uncertainty  
* Tasks that make use of uncertainty in real applications  

The following sections explain each of these ideas in more detail.

2.1 Uncertainty Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An uncertainty representation describes the information a model returns when it predicts with uncertainty.  
Instead of giving a single output, the model provides additional information that shows how unsure it is.  
This information can look different depending on the method, but probly provides one consistent way  
to work with all of them.

A representation is the structure that stores the uncertainty information.  
It is the foundation that probly uses for everything that follows,  
including quantification and downstream tasks.

Different uncertainty methods create different types of representations.  
Some examples include:

* Multiple stochastic forward passes, which appear when using dropout  
* Predictions from several independently trained models, which form an ensemble  
* Parameters of a predictive distribution that come from evidential models  
* Collections of outputs that describe a distribution of possible predictions  

Each method creates uncertainty in its own way, but probly unifies them  
so they can all be handled through a single interface.  
This makes it easier to compare methods, switch between them,  
and build systems that use uncertainty consistently.

The key idea behind uncertainty representations is simple.  
They capture how the model behaves when it is unsure,  
and probly uses this representation as the base for all later steps.

2.2 Uncertainty Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An uncertainty transformation changes how a model makes predictions so that it can express uncertainty.  
Instead of building a new model from the beginning, the transformation wraps the existing model  
and makes it produce uncertainty information in addition to normal outputs.

Transformations are one of the core ideas in probly.  
They let users add uncertainty to their models without rewriting the training process.  
The model can be trained exactly as usual.  
The transformation then controls how the model behaves during prediction.

Common examples of uncertainty transformations include:

* Using dropout during prediction to generate multiple different outputs  
* Combining predictions from several independently trained models in an ensemble  
* Adding evidential layers that output distribution parameters instead of single values  
* Sampling from probabilistic or stochastic layers to produce a range of predictions  

Even though each transformation works differently, probly treats all of them in a unified way.  
The result of any transformation becomes a consistent uncertainty representation  
that probly can analyze, quantify, and use for downstream tasks.

The idea behind uncertainty transformations is simple.  
They modify the prediction behavior of a model so that uncertainty becomes visible.  
probly then provides all tools needed to work with this uncertainty in a reliable and practical way.

2.3 Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a model has an uncertainty representation, the next step is to measure  
how uncertain the model actually is.  
This step is called uncertainty quantification.

Quantification turns the raw uncertainty information into numerical values  
that describe how confident or uncertain the model is about each prediction.  
These values make uncertainty easy to interpret and compare across different models  
and different uncertainty methods.

probly provides a unified set of tools for quantifying uncertainty.  
This means users can switch between different uncertainty methods  
without changing the way they compute uncertainty scores.

Common examples of uncertainty quantities include:

* epistemic uncertainty, which reflects what the model does not know  
  for example because it has not seen similar data during training  
* aleatoric uncertainty, which reflects noise or ambiguity in the data itself  
* predictive entropy, which measures how spread out the predictions are  
* mutual information, which separates model uncertainty from data uncertainty  

These quantities help answer practical questions such as:

* How confident is the model in this prediction  
* Is this input outside the model’s training distribution  
* Should the system rely on the model or ask for human input  

probly makes uncertainty quantification simple and consistent.  
Regardless of whether the model uses dropout, ensembles, evidential methods,  
or any other transformation, the uncertainty representation produced by probly  
can be passed directly into its quantification tools.

The key idea is that quantification turns uncertainty into meaningful numbers  
that can be used in evaluation, decision making, or downstream tasks.

2.4 Downstream Tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a model has an uncertainty representation and the uncertainty  
has been quantified, this information can be used to perform  
important downstream tasks.  
These tasks help make machine learning systems safer, more reliable,  
and more interpretable in real applications.

probly provides built in tools for several common downstream tasks.  
Because all uncertainty methods in probly share the same unified  
representation, these tasks work the same way no matter which  
uncertainty method is used.

Examples of downstream tasks include:

* out of distribution detection, where uncertainty is used to decide  
  whether an input is very different from the training data  
* selective prediction, where the model abstains from a prediction  
  if uncertainty is too high  
* calibration evaluation, which measures how well predicted  
  probabilities match reality  
* risk based decision making, where predictions are combined  
  with uncertainty to take safer actions  

These tasks are essential for deploying machine learning models  
in environments where mistakes are costly or where the system should  
recognize when it does not know enough.

probly makes it easy to apply these tasks by providing simple functions  
that operate directly on the uncertainty representations and  
quantification results.  
This allows users to evaluate and improve model reliability  
without rewriting their training or inference pipelines.

In summary, downstream tasks use the uncertainty information  
provided by probly to make models not only accurate but also safe,  
transparent, and reliable.

