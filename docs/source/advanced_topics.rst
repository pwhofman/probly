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
This chapter is intended for scenarios where users go beyond simple examples, such as:

- training or evaluating large or real-world models,  
- dealing with tight performance or memory constraints,  
- integrating ``probly`` into existing machine-learning pipelines.

These use cases often require a deeper understanding of transformations, scalability, and framework interoperability, which this chapter provides.

.. seealso::

    For background material, see :doc:`Core Concepts <core_concepts.rst>`.

    For the main bulding blocks of ``probly``, like the main transofrmations, utilities & layers, and evaluation tools, see :doc:`Main Components <main_components>`.

2. Custom Transformations
-------------------------

2.1 Recall: What is a transformation?

In ``probly``, a **transformation** is a small building block that maps values between two spaces,
similar in spirit to the bijectors used in TensorFlow Probability (TensorFlow Probability, 2023;
Rezende & Mohamed, 2015):

- an **unconstrained space**, where optimisation and inference algorithms can work freely, and  
- a **constrained space**, which matches the natural domain of your parameters or predictions
  (for example positive scales, probabilities on a simplex, or bounded intervals)
  (TensorFlow Probability, 2023).

Instead of forcing you to design models directly in a complicated constrained space, you write
your model in terms of meaningful parameters, and the transformation then takes care of the math
that keeps everything inside the valid domain (TensorFlow Probability, 2023; Rezende & Mohamed, 2015).

In practice this means that transformations:

- provide a *short, reusable recipe* for how to turn raw latent variables into valid parameters,  
- enable **reparameterisation**, which can make optimisation easier and gradients better behaved
  (Kingma & Welling, 2014),  
- automatically enforce **constraints** such as positivity, bounds, or simplex structure
  (TensorFlow Probability, 2023).

You can think of a transformation as an adapter between “nice for the optimiser” coordinates and
“nice for the human” coordinates (Kingma & Welling, 2014; Rezende & Mohamed, 2015).

2.2 When to implement your own?

The built-in transformations in ``probly`` are designed to cover many common cases,
such as positive scales, simple box constraints, or mappings to probability vectors.
This is similar in spirit to other probabilistic frameworks that provide default
constraint transforms for bounded, ordered, simplex, correlation, or covariance
parameters (Stan Development Team, 2025). In many projects these standard building
blocks are sufficient and you never need to write your own transformation.

There are, however, important situations where a **custom transformation** is the
better choice.

- **Limitations of built-in transformations**  
  Some models use parameter spaces that go beyond the usual catalogue of common constraints such as positive,
bounded, or simplex parameters. For example, you may need structured covariance matrices,
  ordered-but-positive sequences, monotone functions, or parameters that satisfy
  several coupled constraints at once. The Stan reference manual notes that
  “vectors may … be constrained to be ordered, positive ordered, or simplexes”
  and matrices “to be correlation matrices or covariance matrices” (Stan
  Development Team, 2025, Constraint Transforms section), but real applications
  often demand more specialised structures. In such cases, a custom
  transformation lets you explicitly encode the structure your model needs.

- **Custom distributions or domain constraints**  
  In many domains, prior knowledge is naturally expressed as constraints on
  parameters: certain probabilities must always sum to one, some effects must be
  monotone, or fairness and safety requirements restrict which configurations are
  admissible. Recent work on probabilistic circuits emphasises that domain
  constraints can “encode information about general trends in the domain and
  serve as effective inductive biases” (Karanam et al., 2024, p. 3). A custom
  transformation is a convenient way to build such domain-specific rules into the
  parameterisation instead of relying on ad-hoc clipping or post-processing.

- **Cleaner uncertainty behaviour and numerical stability**  
  Some parameterisations yield more interpretable and numerically stable
  uncertainty estimates than others. A classic example is working on a log or
  softplus scale for strictly positive parameters. Stan, for instance, uses a
  logarithmic transform for lower-bounded variables and applies the inverse
  exponential to map back to the constrained space (Stan Development Team, 2025).
  Practitioners have observed that replacing a naïve exponential with a softplus
  transform can substantially stabilise inference; one NumPyro user reports “a
  very substantial improvement in inference stability when I replace `exp`
  transformation with `softplus` for constraining `site_scale`” (vitkl, 2020,
  para. 31). In ``probly``, a custom transformation can encapsulate this kind of
  numerically robust parameterisation and make its effect on uncertainty
  representations easier to reason about.

- **Integration with existing code or libraries**  
  When you plug ``probly`` into an existing machine-learning pipeline, external
  code often expects parameters in a fixed, domain-specific representation. The
  internal unconstrained parameterisation that is convenient for inference may
  not match what a legacy training loop, a deep-learning framework, or a
  production system “expects to see.” A transformation can act as a bridge:
  ``probly`` operates in its preferred unconstrained space, while the surrounding
  code continues to work with familiar application-level parameters (cf. the use
  of constraint transforms to reconcile internal and external parameterisations
  in Stan; Stan Development Team, 2025).

As a practical rule of thumb: if you frequently add manual clamps, min/max
operations, or ad-hoc post-processing steps just to keep parameters valid, that is
a strong signal that a dedicated custom transformation would make the model
cleaner, more robust, and easier to maintain.

2.3 API & design principles

Custom transformations in ``probly`` should follow a **small and predictable interface**. Similar
interfaces appear in other probabilistic libraries. For example, TensorFlow Probability notes
that “A `Bijector` is characterized by three operations: 1. Forward … 2. Inverse … 3.
`log_det_jacobian(x)`” (TensorFlow Probability, 2023), and the Open Source Vizier guide adds
that “Each bijector implements at least 3 methods: `forward`, `inverse`, and (at least) one
of `forward_log_det_jacobian` and `inverse_log_det_jacobian`” (Open Source Vizier Authors, 2022).

Conceptually, each transformation in ``probly`` is responsible for three things:

- a **forward mapping** from an unconstrained input to the constrained parameter space,
  typically used to turn one random outcome into another (TensorFlow Probability, 2023),  
- an **inverse mapping** that recovers the unconstrained value from a constrained one,
  enabling probability and density computations,  
- any **auxiliary quantities** that inference algorithms may need, such as Jacobians or
  log-determinants, to account for the change of variables.

Stan’s transform system illustrates the same pattern: “every (multivariate) parameter in a Stan
model is transformed to an unconstrained variable behind the scenes by the model compiler”
and “the C++ classes also include code to transform the parameters from unconstrained to
constrained and apply the appropriate Jacobians” (Stan Development Team, 2025). In other
words, the model is written in terms of constrained parameters, while inference operates in an
unconstrained space connected by well-defined forward and inverse transforms.

Beyond this minimal interface, good transformations follow several design principles:

- **local and self-contained**  
  All logic that enforces a particular constraint should live inside the transformation. The rest
  of the model should not need to know which reparameterisation is used internally. This mirrors
  how libraries like Stan and NumPyro encapsulate constraints as self-contained objects that define
  where parameters are valid (Contributors to the Pyro Project, 2019; Stan Development Team, 2025).

- **clearly documented domain and range**  
  It should be obvious which inputs are valid, what shapes are expected, and which constraints the
  outputs satisfy. NumPyro’s ``Constraint`` base class explicitly states that “A constraint object
  represents a region over which a variable is valid, e.g. within which a variable can be
  optimized” (Contributors to the Pyro Project, 2019). Documenting domains and ranges for custom
  transformations in ``probly`` serves the same purpose.

- **numerically stable**  
  The implementation should avoid unnecessary overflow, underflow, or extreme gradients. Stan’s
  documentation on constraint transforms highlights numerical issues arising from floating-point
  arithmetic and the need for careful treatment of boundaries and Jacobian terms (Stan Development
  Team, 2025). In practice, this often means using stable variants of mathematical formulas,
  adding small epsilons, or applying safe clipping near boundaries.

- **composable**  
  Whenever possible, transformations should work well in combination with others. TensorFlow
  Probability, for example, provides composition utilities such as ``Chain`` to build complex
  mappings out of simpler bijectors (Open Source Vizier Authors, 2022). In ``probly``, the same
  idea applies: designing transformations to be composable makes it easier to express rich
  constraints while keeping each individual component small and testable.

During **sampling and inference**, ``probly`` repeatedly calls the forward and inverse mappings of
your transformation to move between the internal unconstrained representation and the external
constrained parameters that appear in the model. A well-designed transformation therefore keeps
these operations cheap, stable, and easy to reason about, in line with the goals of similar
transform systems in Stan and TensorFlow Probability (Stan Development Team, 2025;
TensorFlow Probability, 2023).

2.4 Step-by-step tutorial: simple custom transformation

This section walks through a minimal example of implementing a custom transformation in ``probly``.
The goal is not to show every detail of the library API, but to illustrate the typical workflow
from idea to a working component that can be used inside a model.

**Problem description:**

Suppose we want a parameter that must always be **strictly positive**, for example a scale or
standard deviation. Working directly with a positive variable is inconvenient for optimisation, so
we introduce an unconstrained real-valued variable and use a transformation to map it into the
positive domain.

Our transformation therefore needs to:

- take any real number as input,
- output a strictly positive value,
- be invertible (or at least approximately invertible) so that inference algorithms in ``probly``
  can move between the two spaces.

**Implementation:**

At implementation time we translate this idea into a small transformation object. Conceptually, it
contains:

- a **forward** method that maps from the unconstrained real line to positive values
  (for example via an exponential or softplus mapping),
- an **inverse** method that maps positive values back to the real line,
- any additional helpers required by the inference backends, such as computing a log-determinant
  of the Jacobian if needed.

The concrete class and method names depend on the exact transformation base class used by
``probly``, but the conceptual structure is always the same.

**Registration / configuration:**

Once implemented, the transformation must be **registered** so that ``probly`` can find and use it.
This usually means:

- making the class importable from the appropriate module,
- optionally adding it to a registry or configuration table,
- defining any configuration options (for example, whether to clamp values near the boundary, or
  which nonlinearity to use).

After registration, the transformation can be referred to by name or imported wherever it is needed.

**Using it in a model:**

To use the transformation in a model, we introduce an unconstrained latent parameter and attach the
transformation to it. During model construction, ``probly`` will then:

- store the transformation together with the parameter,
- transparently apply the forward mapping whenever the constrained parameter is needed,
- keep track of the relationship so that gradients and uncertainty estimates remain consistent.

From the model author’s perspective, the parameter now behaves like a normal positive quantity, even
though internally it is represented by an unconstrained variable.

**Running inference and inspecting results:**

When we run inference, optimisation, or sampling, ``probly`` operates in the unconstrained space but
uses the transformation to interpret results in the constrained space. After the run finishes, we
can:

- inspect posterior samples or point estimates of the constrained parameter,
- verify that all inferred values satisfy the desired constraints,
- compare behaviour with and without the custom transformation to understand its impact.

This simple workflow generalises to more complex transformations with multiple inputs, coupled
constraints, or additional structure.

2.5 Advanced patterns

Once you are comfortable with basic custom transformations, ``probly`` allows for more advanced
usage patterns that can make large or complex models easier to express.

**Composing multiple transformations**

Often it is easier to build a complex mapping by **composing several simple transformations** rather
than writing one large one. For example, you might:

- first apply a shift-and-scale transform,
- then map the result onto a simplex,
- finally enforce an ordering constraint.

When transformations are designed to be composable, ``probly`` can chain their forward and inverse
operations, giving you a flexible way to express rich constraints while keeping each component easy
to test and reason about.

**Sharing parameters across transformations**

In some models, several transformations depend on a **shared parameter** or hyperparameter (for
example a common scale or concentration parameter). Instead of duplicating this value, it is often
better to:

- define the shared quantity once,
- pass references to it into multiple transformations,
- ensure that updates to the shared parameter are consistently reflected in all dependent
  transformations.

This pattern encourages modular model design while keeping the statistical meaning of shared
structure explicit.

**Handling randomness vs determinism inside transformations**

Most transformations are deterministic mappings, but in some cases it is useful to include
controlled **randomness** inside a transformation (for example randomised rounding or stochastic
discretisation). When doing so, keep in mind:

- deterministic behaviour is usually easier for optimisation and debugging,
- if randomness is used, it should be driven by the same PRNG and seeding mechanisms as the rest
  of the ``probly`` model,
- the statistical interpretation of the model should remain clear even when transformations are
  stochastic.

2.6 Testing & debugging

Well-tested transformations are crucial for trustworthy models. Because transformations sit between
the internal representation and the visible parameters, subtle bugs can be hard to detect unless
you test them explicitly.

**Round-trip tests (forward + inverse)**

A basic but powerful test is the **round-trip check**:

- sample or construct a range of valid unconstrained inputs,
- apply the forward mapping followed by the inverse mapping,
- verify that the original inputs are recovered (up to numerical tolerance).

Similarly, you can test constrained values by applying inverse then forward. Systematic deviations
usually indicate mistakes in the formulas or shape handling.

**Numerical stability checks**

Transformations that operate near boundaries (very small or very large values, probabilities near
0 or 1, etc.) can suffer from numerical problems. It is good practice to:

- test extreme but valid inputs,
- check for overflow, underflow, or `nan`/`inf` values,
- monitor gradients if the transformation is used in gradient-based inference.

Where necessary, introduce small epsilons, safe clipping, or alternative parameterisations to keep
the transformation stable.

**Common pitfalls and how to recognise them**

Typical issues with custom transformations include:

- silently producing invalid outputs (for example negative values where only positives are allowed),
- mismatched shapes between forward and inverse mappings,
- forgetting to update the transformation when the model structure changes,
- inconsistent handling of broadcasting or batching.

Symptoms of these problems often show up later as:

- optimisation failing to converge,
- extremely large or unstable uncertainty estimates,
- runtime errors deep inside the inference code.

When such issues appear, it is often helpful to temporarily isolate the transformation in a small
test script, run the round-trip and stability checks described above, and only then reintegrate it
into the full ``probly`` model.



.. bibliography::
Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *Proceedings of the 2nd
International Conference on Learning Representations (ICLR).* https://arxiv.org/abs/1312.6114

Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. *Proceedings of
the 32nd International Conference on Machine Learning (ICML), 37*, 1530–1538.
https://proceedings.mlr.press/v37/rezende15.html

TensorFlow Probability. (2023). *Module: tfp.bijectors* [Computer software documentation].
TensorFlow. https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors

Karanam, A., Mathur, S., Sidheekh, S., & Natarajan, S. (2024).
A unified framework for human-allied learning of probabilistic circuits.
*arXiv.* https://arxiv.org/abs/2405.02413

Stan Development Team. (2025).
*Stan reference manual* (Version 2.37).
https://mc-stan.org/docs/reference-manual/

vitkl. (2020, December 31).
*Softplus transform as a more numerically stable way to enforce positive constraint
[Issue #855]*. GitHub. https://github.com/pyro-ppl/numpyro/issues/855

Contributors to the Pyro Project. (2019).
*NumPyro: numpyro.distributions.constraints* [Source code documentation].
NumPyro. https://num.pyro.ai/en/0.3.0/_modules/numpyro/distributions/constraints.html

Open Source Vizier Authors. (2022).
*Bijectors*. In *Open Source Vizier documentation*.
https://oss-vizier.readthedocs.io/en/latest/advanced_topics/tfp/bijectors.html

Stan Development Team. (2025).
*Constraint transforms*. In *Stan reference manual* (Version 2.37).
https://mc-stan.org/docs/reference-manual/transforms.html

TensorFlow Probability. (2023).
*tfp.bijectors.Bijector* [Computer software documentation].
TensorFlow. https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Bijector


