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
from an initial idea to a working component that can be used inside a model.

**Problem description**

Suppose we want a parameter that must always be **strictly positive**, for example a scale or
standard deviation. Many probabilistic frameworks enforce such constraints by transforming from an
unconstrained real variable into a positive domain. For instance, the Stan reference manual notes
that “Stan uses a logarithmic transform for lower and upper bounds” (Stan Development Team, n.d.),
and TensorFlow Probability’s Softplus bijector is documented as having “the domain [of] the
positive real numbers” (TensorFlow Probability, 2023). Following the same idea, we introduce an
unconstrained real-valued variable and use a transformation to map it into the positive domain.

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

Different libraries choose different specific transforms. Stan typically uses a log transform for
strictly positive parameters (Stan Development Team, n.d.), while TensorFlow Probability provides a
Softplus bijector, which “does not overflow as easily as the `Exp` Bijector” due to its asymptotic
behaviour (TensorFlow Probability, 2023). NumPyro implements a similar idea with a dedicated
Softplus-based transform from “unconstrained space to positive domain via softplus” in its
transforms module (Contributors to the Pyro Project, n.d.). In practice, this means you can choose
between an exponential-style mapping (simple but potentially less stable) and a softplus-style
mapping (slightly more complex but often more robust).

The concrete class and method names in a custom transformation depend on the transformation base
class used by ``probly``, but the conceptual structure is always the same: a forward map, an
inverse map, and (when required) the corresponding Jacobian terms.

**Registration / configuration**

Once implemented, the transformation must be **registered** so that ``probly`` can find and use it.
This usually means:

- making the class importable from the appropriate module,
- optionally adding it to a registry or configuration table,
- defining any configuration options (for example, whether to clamp values near the boundary, or
  which nonlinearity to use).

In other systems, something similar happens when new bijectors or constraint objects are added to
the library’s registry and then reused across models (Contributors to the Pyro Project, n.d.;
TensorFlow Probability, 2023). In ``probly``, registration plays the same role: it turns a single
implementation into a reusable building block.

After registration, the transformation can be referred to by name or imported wherever it is needed.

**Using it in a model**

To use the transformation in a model, we introduce an unconstrained latent parameter and attach the
transformation to it. During model construction, ``probly`` will then:

- store the transformation together with the parameter,
- transparently apply the forward mapping whenever the constrained parameter is needed,
- keep track of the relationship so that gradients and uncertainty estimates remain consistent.

This mirrors the way Stan and other packages internally work with unconstrained parameters while
presenting constrained parameters in the modelling language (Stan Development Team, n.d.; Stan
Development Team, 2015). From the model author’s perspective, the parameter now behaves like a
normal positive quantity, even though internally it is represented by an unconstrained variable.

**Running inference and inspecting results**

When we run inference, optimisation, or sampling, ``probly`` operates in the unconstrained space but
uses the transformation to interpret results in the constrained space. After the run finishes, we
can:

- inspect posterior samples or point estimates of the constrained parameter,
- verify that all inferred values satisfy the desired constraints,
- compare behaviour with and without the custom transformation to understand its impact.

Empirically, users have reported that carefully chosen positive transforms can significantly
improve numerical behaviour. For example, one NumPyro user notes “a very substantial improvement in
inference stability when I replace `exp` transformation with `softplus` for constraining
`site_scale`” (vitkl, 2020). This simple workflow generalises to more complex transformations with
multiple inputs, coupled constraints, or additional structure, and similar patterns appear across
modern probabilistic programming frameworks.

2.5 Advanced patterns

Once you are comfortable with basic custom transformations, ``probly`` allows for more advanced
usage patterns that can make large or complex models easier to express. In the wider literature,
normalizing flows show how powerful models can be obtained by composing simple invertible
transformations (Papamakarios et al., 2021; Rezende & Mohamed, 2015).

**Composing multiple transformations**

Often it is easier to build a complex mapping by **composing several simple transformations**
rather than writing one large one. For example, you might:

- first apply a shift-and-scale transform,
- then map the result onto a simplex,
- finally enforce an ordering constraint.

Normalizing-flow work explicitly argues that “we can build complex transformations by composing
multiple instances of simpler transformations” (Papamakarios et al., 2021, p. 3), while still
preserving invertibility and differentiability. Deep-learning libraries such as TensorFlow
Probability provide bijector APIs that implement this idea in practice, allowing chains of
transforms to be treated as a single object (TensorFlow Probability, n.d.).

Designing custom transformations in ``probly`` with this mindset keeps each piece simple and
testable: each small transform has a clear responsibility, and the full behaviour emerges from
their composition.

**Sharing parameters across transformations**

In some models, several transformations depend on a **shared parameter** or hyperparameter (for
example a common scale or concentration parameter). Instead of duplicating this value, it is often
better to:

- define the shared quantity once,
- pass references to it into multiple transformations,
- ensure that updates to the shared parameter are consistently reflected in all dependent
  transformations.

This pattern is closely related to hierarchical Bayesian modelling, where group-specific
parameters are tied together through common hyperparameters. In that context, “hierarchical models
allow for the pooling of information across groups while accounting for group-specific
variations” (Mittal, 2025, para. 2). Using shared parameters across transformations in ``probly``
has a similar effect: information is shared in a controlled way, and the structure of the model
remains explicit and interpretable.

**Handling randomness vs determinism inside transformations**

Most transformations are deterministic mappings, but in some cases it is useful to include
controlled **randomness** inside a transformation (for example randomised rounding or stochastic
discretisation). When you design such components, it helps to follow the discipline used by
modern functional ML frameworks.

For example, the JAX documentation emphasises that JAX “avoids implicit global random state, and
instead tracks state explicitly via a random `key`” (The JAX Authors, 2024, sec. “Explicit random
state”), and that “the crucial point is that you never use the same key twice” (The JAX Authors,
2024, sec. “Explicit random state”). Even if ``probly`` uses a different backend, the same
principles are useful:

- deterministic behaviour is usually easier for optimisation and debugging,
- if randomness is used, it should be driven by the same seeding and PRNG mechanisms as the rest
  of the model,
- the statistical meaning of the model should remain clear even when transformations are
  stochastic.

In practice, this means treating any random choices inside a transformation as part of the
probabilistic model, not as hidden side effects.

2.6 Testing & debugging

Well-tested transformations are crucial for trustworthy models. Because transformations sit
between the internal representation and the visible parameters, subtle bugs can be hard to
detect unless you test them explicitly. Large probabilistic frameworks such as Stan rely on
“extensive unit tests … for accuracy of values and derivatives, as well as error checking”
(Carpenter et al., 2017, p. 24), which is a good benchmark for how seriously this layer should
be treated.

**Round-trip tests (forward + inverse)**

A basic but powerful test is the **round-trip check**:

- sample or construct a range of valid unconstrained inputs,
- apply the forward mapping followed by the inverse mapping,
- verify that the original inputs are recovered (up to numerical tolerance).

From a mathematical point of view, this is just checking the fundamental property of a
bijective transform. Walton (2023) emphasises that “all bijective functions are invertible”
and satisfy :math:`f^{-1}(f(x)) = x`, which is exactly what round-trip tests are designed to
catch when your implementation or shape handling is wrong.

Similarly, you can test constrained values by applying inverse then forward. Systematic
deviations in either direction usually indicate mistakes in the formulas, inconsistencies in
broadcasting, or shape mismatches between forward and inverse.

**Numerical stability checks**

Transformations that operate near boundaries (very small or very large values, probabilities
near 0 or 1, etc.) can suffer from numerical problems. It is good practice to:

- test extreme but valid inputs,
- check for overflow, underflow, or `nan`/`inf` values,
- monitor gradients if the transformation is used in gradient-based inference.

Practical experience in differentiable simulation libraries shows why this matters. The
DiffeRT documentation notes that NaNs “tend to spread uncontrollably, making it difficult to
trace their origin” and therefore adopts a strict *no-NaN policy* for both outputs and
gradients (Eertmans, 2025, “No-NaN Policy” section). The same mindset works well in
``probly``: treat any appearance of NaNs or infinities as a bug in either the transformation
or its inputs, and add targeted tests to reproduce and eliminate it.

Where necessary, introduce small epsilons, safe clipping, or alternative parameterisations
to keep the transformation stable. For instance, many implementations replace naïve formulas
by numerically stable variants or custom Jacobians when differentiability and stability
conflict (see, e.g., Griewank & Walther, 2008, on stable automatic differentiation).

**Common pitfalls and how to recognise them**

Typical issues with custom transformations include:

- silently producing invalid outputs (for example negative values where only positives are allowed),
- mismatched shapes between forward and inverse mappings,
- forgetting to update the transformation when the model structure changes,
- inconsistent handling of broadcasting or batching.

Basic unit-testing advice for probabilistic code still applies here. As one practitioner
summarises, you should at least “assert that the returned value is not null and in the range
you expect” and then add stronger distributional checks where appropriate (hvgotcodes, 2012,
para. 3). For transformations, that means checking *both* the unconstrained and constrained
spaces for sanity (ranges, monotonicity, simple invariants).

Symptoms of problems with transformations often show up later as:

- optimisation failing to converge or getting stuck,
- extremely large or unstable uncertainty estimates,
- runtime errors or NaNs deep inside the inference code.

Empirical studies of probabilistic programming systems show that many real bugs are linked
to boundary conditions, dimension handling, and numerical issues (Dutta et al., 2018). Their
tool ProbFuzz, for example, “discovered 67 potential previously unknown bugs” across three
major systems and “caught at least one existing bug in 8 of 9 categories” they targeted
(Dutta et al., 2018, pp. 1, 7). This underlines that small mistakes in transform logic can
have large downstream effects.

When such issues appear in a ``probly`` model, it is often helpful to temporarily isolate
the transformation in a small test script, run the round-trip and stability checks described
above, and only then reintegrate it into the full model. This mirrors the way mature
probabilistic frameworks separate low-level tests of math functions and transforms from
high-level tests of full models (Carpenter et al., 2017).


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

Contributors to the Pyro Project. (2020). *NumPyro documentation* (Version 0.3.0). NumPyro.
https://num.pyro.ai/_/downloads/en/0.3.0/pdf/

Stan Development Team. (n.d.). *10.2 Lower bounded scalar*. In *Stan reference manual* (Version 2.22).
Stan. https://mc-stan.org/docs/2_22/reference-manual/lower-bound-transform-section.html

TensorFlow Probability. (2023). *tfp.bijectors.Softplus* [Computer software documentation]. TensorFlow.
https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Softplus

vitkl. (2020, December 30). *Softplus transform as a more numerically stable way to enforce positive
constraint [Issue #855]*. In *NumPyro* (GitHub repository). GitHub.
https://github.com/pyro-ppl/numpyro/issues/855

Mittal, B. (2025, July 23). *Bayesian hierarchical models*. GeeksforGeeks.  
https://www.geeksforgeeks.org/machine-learning/bayesian-hierarchical-models/

Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). Normalizing flows for probabilistic modeling and inference. *Journal of Machine Learning Research, 22*(57), 1–64.  
https://www.jmlr.org/papers/volume22/19-1028/19-1028.pdf

Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. In *Proceedings of the 32nd International Conference on Machine Learning* (pp. 1530–1538). PMLR.  
https://proceedings.mlr.press/v37/rezende15.html

TensorFlow Probability. (n.d.). *Module: tfp.bijectors*. In *TensorFlow Probability API documentation*. Retrieved December 5, 2025, from  
https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors

The JAX Authors. (2024). *Pseudorandom numbers*. JAX documentation.  
https://docs.jax.dev/en/latest/random-numbers.html

Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M.,
Brubaker, M. A., Guo, J., Li, P., & Riddell, A. (2017). Stan: A probabilistic programming
language. *Journal of Statistical Software, 76*(1), 1–32. https://doi.org/10.18637/jss.v076.i01

Dutta, S., Legunsen, O., Huang, Z., & Misailovic, S. (2018). Testing probabilistic programming
systems. In *Proceedings of the 26th ACM Joint European Software Engineering Conference and
Symposium on the Foundations of Software Engineering (ESEC/FSE ’18)* (pp. 574–586).
Association for Computing Machinery. https://doi.org/10.1145/3236024.3236057

Eertmans, J. (2025). *NaN and infinite values*. In *DiffeRT documentation*.  
https://differt.eertmans.be/latest/nans_and_infs.html

Griewank, A., & Walther, A. (2008). *Evaluating derivatives: Principles and techniques of
algorithmic differentiation* (2nd ed.). Society for Industrial and Applied Mathematics.  
https://doi.org/10.1137/1.9780898717766

hvgotcodes. (2012, March 29). *What are some good practices for unit testing probability
distributions?* Stack Overflow. https://stackoverflow.com/questions/9934903

Walton, S. (2023). *Isomorphism, normalizing flows, and density estimation: Preserving
relationships between data* (Area Exam Report No. AREA-202307-Walton). University of Oregon,
Department of Computer and Information Sciences.  
https://www.cs.uoregon.edu/Reports/AREA-202307-Walton.pdf



