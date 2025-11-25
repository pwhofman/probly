Advanced Topics
===============

1. Overview
-----------

1.1 Purpose of this chapter
This chapter explains:

- what “advanced” means in the context of ``probly``,
- when you should read this chapter (recommended after Core Concepts and Main Components).

1.2 Prerequisites & notatin
Before reading this chapter, the reader should already be familiar with:

- the concepts introduced in Core Concepts,  
- the basic workflows described in Main Components,  
- foundational ideas such as uncertainty representations, transformations, and inference.

For clarity, this chapter follows the same notation conventions used throughout the ``probly`` documentation.

1.3 Typical advanced use cases
This chapter is intended for scenarios where users go beyond simple toy examples, such as:

- training or evaluating large or real-world models,  
- dealing with tight performance or memory constraints,  
- integrating ``probly`` into existing machine-learning pipelines.

These use cases often require a deeper understanding of transformations, scalability, and framework interoperability, which this chapter provides.

.. seealso::

    For background material, see :doc:`Core Concepts <core_concepts>`.

    For the main bulding blocks of ``probly``, like the main transofrmations, utilities & layers, and evaluation tools, see :doc:`Main Components <main_components>`.

2. Custom Transformations
-------------------------

2.1 Recall: What is a transformation?

In ``probly``, a **transformation** is a small building block that maps values between two spaces:

- an **unconstrained space**, where optimisation and inference algorithms can work freely, and  
- a **constrained space**, which matches the natural domain of your parameters or predictions
  (for example positive scales, probabilities on a simplex, or bounded intervals).

Instead of forcing you to design models directly in a complicated constrained space, you write
your model in terms of meaningful parameters. The transformation then takes care of the math that
keeps everything inside the valid domain.

In practice this means that transformations:

- provide a *short, reusable recipe* for how to turn raw latent variables into valid parameters,
- enable **reparameterisation**, which can make optimisation easier and gradients better behaved,
- automatically enforce **constraints** such as positivity, bounds, or simplex structure.

You can think of a transformation as an adapter between “nice for the optimiser” coordinates and
“nice for the human” coordinates.

2.2 When to implement your own?

The built-in transformations in ``probly`` cover many common situations, such as positive scales,
simple box constraints, or mappings to probability vectors. In many projects these are sufficient
and you never have to write your own.

There are, however, important cases where a **custom transformation** is the better choice:

- **Limitations of built-in transformations**  
  Your model uses a parameter space that is not covered by the standard transforms. Examples
  include structured covariance matrices, ordered variables, monotone functions, or parameters
  that must satisfy several coupled constraints at once.

- **Custom distributions or domain constraints**  
  Domain knowledge may require that parameters follow a particular shape or relationship
  (“these values must always sum to one”, “this parameter must stay in a problem-specific
  range”, “these two variables share a common scale”). A custom transformation lets you encode
  these rules explicitly instead of relying on ad-hoc clipping.

- **Cleaner uncertainty behaviour**  
  Some parameterisations produce more interpretable or numerically stable uncertainty estimates,
  for example working on a log-scale for strictly positive variances. A custom transformation can
  make the connection to the uncertainty representations from :doc:`Core Concepts <core_concepts>`
  more transparent.

- **Integration with existing code or libraries**  
  When you plug ``probly`` into an existing ML pipeline, external code often expects parameters
  in a fixed format. A transformation can serve as a bridge: ``probly`` works in its preferred
  unconstrained space, while the surrounding code still “sees” the familiar domain-specific
  representation.

As a practical rule: if you frequently add manual clamps, min/max operations, or ad-hoc
post-processing to keep parameters valid, it is a strong signal that a dedicated custom
transformation would make the model cleaner and more robust.

2.3 API & design principles

Custom transformations should follow a **small and predictable interface**. Conceptually, each
transformation is responsible for three things:

- a **forward mapping** from an unconstrained input to the constrained parameter space,
- an **inverse mapping** that recovers the unconstrained value from a constrained one (where
  this is well-defined),
- any **auxiliary quantities** that inference algorithms may need, such as Jacobians or
  log-determinants, depending on the method used.

Beyond this minimal interface, good transformations follow a few design principles:

- **local and self-contained**  
  All logic that enforces a particular constraint should live inside the transformation. The rest
  of the model should not need to know which reparameterisation is used internally.

- **clearly documented domain and range**  
  It should be obvious which inputs are valid, what shapes are expected, and which constraints the
  outputs satisfy. This makes debugging and reuse much easier.

- **numerically stable**  
  The implementation should avoid unnecessary overflow, underflow, or extreme gradients. Small
  epsilons, stable variants of mathematical formulas, or safe clipping near the boundaries are
  often needed in practice.

- **composable**  
  Whenever possible, transformations should work well in combination with others, for example a
  scaling transform followed by a simplex transform, or a log-transform followed by a shift.

During **sampling and inference**, ``probly`` repeatedly calls the forward and inverse mappings of
your transformation to move between the internal unconstrained representation and the external
constrained parameters that appear in the model. A well-designed transformation therefore keeps
these operations cheap, stable, and easy to reason about.

2.4 Step-by-step tutorial: simple custom transformation

This section walks through a minimal example of implementing a custom transformation in ``probly``.
The goal is not to show every detail of the library API, but to illustrate the typical workflow
from idea to a working component that can be used inside a model.

**Problem description**

Suppose we want a parameter that must always be **strictly positive**, for example a scale or
standard deviation. Working directly with a positive variable is inconvenient for optimisation, so
we introduce an unconstrained real-valued variable and use a transformation to map it into the
positive domain.

Our transformation therefore needs to:

- take any real number as input,
- output a strictly positive value,
- be invertible (or at least approximately invertible) so that inference algorithms in ``probly``
  can move between the two spaces.

**Implementation**

At implementation time we translate this idea into a small transformation object. Conceptually, it
contains:

- a **forward** method that maps from the unconstrained real line to positive values
  (for example via an exponential or softplus mapping),
- an **inverse** method that maps positive values back to the real line,
- any additional helpers required by the inference backends, such as computing a log-determinant
  of the Jacobian if needed.

The concrete class and method names depend on the exact transformation base class used by
``probly``, but the conceptual structure is always the same.

**Registration / configuration**

Once implemented, the transformation must be **registered** so that ``probly`` can find and use it.
This usually means:

- making the class importable from the appropriate module,
- optionally adding it to a registry or configuration table,
- defining any configuration options (for example, whether to clamp values near the boundary, or
  which nonlinearity to use).

After registration, the transformation can be referred to by name or imported wherever it is needed.

**Using it in a model**

To use the transformation in a model, we introduce an unconstrained latent parameter and attach the
transformation to it. During model construction, ``probly`` will then:

- store the transformation together with the parameter,
- transparently apply the forward mapping whenever the constrained parameter is needed,
- keep track of the relationship so that gradients and uncertainty estimates remain consistent.

From the model author’s perspective, the parameter now behaves like a normal positive quantity, even
though internally it is represented by an unconstrained variable.

**Running inference and inspecting results**

When we run inference, optimisation, or sampling, ``probly`` operates in the unconstrained space but
uses the transformation to interpret results in the constrained space. After the run finishes, we
can:

- inspect posterior samples or point estimates of the constrained parameter,
- verify that all inferred values satisfy the desired constraints,
- compare behaviour with and without the custom transformation to understand its impact.

This simple workflow generalises to more complex transformations with multiple inputs, coupled
constraints, or additional structure.
