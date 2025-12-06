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

3. Working with Large Models
----------------------------

3.1 What is a “large” model in practice?

What counts as a “large” model depends on the context, hardware, and goals of the project. In the
literature on large AI models, the term is often used for networks with hundreds of millions or
billions of parameters and complex computational structures (Tu, 2024). In everyday ``probly``
projects, you will usually encounter “large-model” issues much earlier, whenever memory, runtime,
or data handling start to dominate your workflow.

A model is typically “large in practice” when at least one of the following becomes a real
constraint:

- **Model dimensions and parameter count**  
  As parameter counts grow, the memory footprint for parameters, gradients, optimizer state, and
  activations can approach or exceed what a single device can handle comfortably. Practical guides
  on deploying large-scale deep learning models emphasise that model size, memory usage, and the
  need to split computation across multiple GPUs or machines are central challenges (Tyagi, 2025).

- **Dataset size**  
  A model can also feel “large” because the dataset is large. If the full dataset no longer fits
  into memory, you must rely on streaming, sharding, or mini-batch pipelines. Large-scale systems
  often report that I/O and data preprocessing can become bottlenecks as data volume increases
  (Tyagi, 2025), and the same effect appears in ``probly`` projects once you move beyond small,
  in-memory datasets.

- **Runtime and resource constraints**  
  Even a moderate-sized model may be “large” if a single training run takes many hours, if
  inference latency is critical, or if energy and hardware costs limit how many experiments you
  can run. Discussions of large-model deployments highlight the tension between model size,
  training time, and cost as a key obstacle in practice (Tyagi, 2025; Tu, 2024).

For the purposes of this chapter, we will treat a model as “large” whenever memory, data handling,
or runtime constraints force you to think about how the model is structured and how computations
are organised, rather than letting you focus solely on the modelling idea.

3.2 Model structuring strategies

As models and datasets grow, the structure of your code and model definition becomes just as
important as the choice of algorithm. Good project structure makes it easier to reason about large
systems, reuse components, and debug problems. Guides on organising Python and data science
projects consistently recommend a modular layout rather than a single monolithic script (Pati,
2025; The Hitchhiker’s Guide to Python Contributors, 2024).

**Modular design (sub-models and reusable components)**

For ``probly`` projects, a modular design usually means:

- separating data loading and preprocessing from model definition and inference,
- grouping related model components (for example a family of uncertainty-aware classifiers) into
  their own modules,
- encapsulating common patterns into reusable functions or classes.

Pati (2025) illustrates how splitting a data science project into modules such as
``preprocess.py``, ``train.py``, and ``predict.py`` improves maintainability and reuse. The same
idea applies to ``probly``: instead of one very large model file, you can create sub-models or
building blocks (for example, shared transformation libraries or common likelihood components)
that can be imported into multiple experiments.

**Naming conventions and file/project organisation**

Clear naming and consistent layout make large codebases easier to navigate. The “Structuring Your
Project” chapter in *The Hitchhiker’s Guide to Python* recommends short, lowercase module names
and discourages deeply nested, ad hoc layouts, because they tend to introduce confusing imports
and circular dependencies (The Hitchhiker’s Guide to Python Contributors, 2024).

In practice, this can mean:

- using descriptive filenames such as ``large_models/core_layers.py``,
  ``probly_transforms/constraints.py``, or ``pipelines/large_experiment.py``,
- keeping reusable library code (for example, generic ``probly`` models and transformations)
  separate from experiment-specific scripts and notebooks,
- adopting a simple, documented folder structure so that new contributors can quickly find model
  definitions, data processing code, configurations, and documentation (LuxDevHQ, 2023; Pati, 2025).

A clean structure does not make a model smaller, but it makes it **feel** smaller: you can focus on
one piece at a time, understand dependencies, and swap components in and out without losing track
of the overall system. The later sections in this chapter (3.3–3.6) assume that your ``probly``
code is at least partially modularised into sub-models, transformations, and pipelines rather than
being a single long script.

3.3 Memory management

For small toy examples, you can often ignore memory and just run the model. As soon as datasets
and models grow, memory becomes a central constraint: you may hit GPU out-of-memory errors, see
training slow down because of data loading, or notice that copying tensors between devices
dominates the runtime. This section outlines a few practical patterns for managing memory in
``probly`` projects.

**Batching and mini-batching**

The basic idea of mini-batching is simple: instead of processing the whole dataset at once, you
split it into smaller batches and process one batch at a time. Mini-batch gradient descent is
commonly described as balancing the stability of full-batch gradient descent with the efficiency
of stochastic updates by “processing small batches of data at a time” (Bhuva, 2025), which both
reduces memory usage and improves hardware utilisation (GeeksforGeeks, 2025; Jason Brownlee, 2019).

For memory management in ``probly``, this means:

- choose a batch size that comfortably fits into GPU or CPU memory,
- keep intermediate tensors (activations, representations) only for the current batch,
- avoid accidentally materialising the entire dataset as one huge tensor.

As long as your batch fits into device memory, you can usually scale to much larger datasets by
running more batches, rather than trying to fit everything at once.

**Streaming data**

When datasets no longer fit into RAM (or when reading them all at once would be too slow),
streaming becomes essential. Deep-learning input-pipeline tools such as TensorFlow’s ``tf.data``
API are explicitly designed to “handle large amounts of data, read from different data formats,
and perform complex transformations” in a streaming fashion (TensorFlow, 2024a). Performance
guides recommend combining operations such as ``prefetch``—which overlaps data loading and model
execution—and parallel ``map`` and ``interleave`` to ensure that the model is never waiting
idly for the next batch of data (TensorFlow, 2024b; Martire, 2022).

The same principles apply to ``probly``:

- build a data pipeline (using your framework of choice) that yields reasonably sized batches,
- overlap I/O and preprocessing with computation where possible,
- avoid repeatedly re-reading or re-decoding the same data if it can be cached safely.

From ``probly``’s perspective, it does not matter whether the batch comes from an in-memory list,
a generator, or a sophisticated ``tf.data`` or PyTorch ``DataLoader`` pipeline—only that each
batch fits in memory and arrives fast enough.

**Avoiding unnecessary copies and recomputations**

Memory pressure and slowdowns often come from hidden copies:

- moving tensors back and forth between CPU and GPU unnecessarily,
- creating new tensors instead of reusing existing buffers,
- recomputing large intermediate results instead of caching them.

The PyTorch CUDA semantics guide, for example, notes that host-to-GPU copies are “much faster
when they originate from pinned (page-locked) memory” and exposes a ``pin_memory()`` method on
CPU tensors for that reason (PyTorch Documentation, 2017). More recent tutorials explain that
with ``pin_memory=True``, data can be transferred asynchronously and “pre-loaded into the GPU”
while the previous batch is still being processed, reducing idle time (Hey Amit, 2024).

In practice for ``probly`` models this suggests:

- minimise the number of device transfers inside hot loops (move data once per batch, not once
  per operation),
- be deliberate about when you call operations such as ``.cpu()``, ``.numpy()`` or similar
  conversions, since each conversion can allocate new memory,
- cache expensive, reusable results (for example, fixed embeddings or precomputed features),
  rather than recomputing them every iteration if they do not change.

Careful profiling of memory and runtime (for example with your framework’s profiler) will usually
reveal whether copies, data loading, or the model itself are the main bottleneck. The strategies
above are then the first levers to try.

3.4 Scalability features in ``probly``

Even with good memory-management habits, some models will still push the limits of hardware
capacity. To cope with that, modern numerical libraries provide features such as vectorisation
and just-in-time (JIT) compilation. ``probly`` can take advantage of similar ideas when it is
built on top of frameworks like JAX or other XLA-based backends.

**Vectorisation**

Vectorisation means applying the same computation to many inputs in one go, using array
operations instead of Python loops. In JAX, for example, the ``vmap`` transformation “maps a
function over array axes” and lets you write a function once and apply it efficiently to a whole
batch of inputs (JAX Authors, n.d.-a; Foreman-Mackey, n.d.). Tutorials highlight that these tools
allow you “to apply functions to arrays of data in parallel” and thereby improve performance
(PyImageSearch, 2023).

In ``probly``, vectorisation typically shows up as:

- writing model code so that it naturally accepts batched inputs,
- leveraging backend tools (such as ``vmap`` in JAX or vectorised operations in NumPy/PyTorch)
  to evaluate many parameter settings or data points in a single call,
- avoiding Python ``for``-loops in critical performance paths when a batched array operation
  would suffice.

This reduces Python overhead, allows the compiler or backend to fuse operations, and often leads
to both faster and more memory-efficient code.

**JIT compilation and backend compilation options**

Just-in-time compilation takes a Python function and compiles it into an efficient, static
computation graph for accelerators. The JAX documentation describes ``jax.jit`` as a
transformation that “will perform Just In Time (JIT) compilation of a JAX Python function so it
can be executed efficiently in XLA” (JAX Authors, n.d.-b). Other tutorials show how combining
``jit`` with vectorisation can “make simulations faster” and fully exploit GPUs (Jaxley Team,
n.d.; Koul, 2023).

When ``probly`` runs on such backends, it can:

- trace parts of the model or inference loop into compiled functions,
- reuse compiled graphs across many calls (for example many batches or chains),
- use configuration flags to enable or disable JIT depending on debugging vs. production needs.

Typical configuration “knobs” you might expose in a ``probly`` project include:

- turning JIT on or off for the main log-likelihood or posterior evaluation,
- controlling which axes are vectorised (e.g. batching over data vs. batching over chains),
- switching between pure Python execution (for debugging) and fully compiled execution (for
  large experiments).

The underlying ideas come directly from libraries like JAX, which describe themselves as
“composable transformations of Python+NumPy programs” and use XLA to “compile and scale your
NumPy programs on TPUs, GPUs, and other hardware accelerators” (JAX Authors, n.d.-c). ``probly``
can build on these mechanisms to make large models feasible to run in practice, while still
giving users explicit control over when and how compilation is used.

3.5 Case study: scaling up a small example

This section sketches a typical journey: you start with a tiny, clean example on your laptop and
gradually grow it into a larger ``probly`` model and dataset. The goal is not to provide exact
code, but to show which design choices make scaling smoother.

**Step 1 – Start with a small, clean baseline**

Imagine you begin with a simple classification model on a small, well-known dataset (for example,
a few thousand examples stored in a single file). At this stage you:

- use a straightforward model structure,
- run on a single CPU or GPU,
- keep all data in memory,
- focus on correctness and clarity, not speed.

Guides on practical ML strongly recommend this style: start with a simple model and get the
pipeline right before adding complexity (Zinkevich, n.d.; Karpathy, 2019).
For a ``probly`` project, this means:

- verifying that the model compiles and runs end-to-end,
- checking that transformations, priors, and inference all behave as expected,
- establishing a small set of metrics (loss, accuracy, calibration, runtime per epoch, etc.).

At this phase, you deliberately avoid large datasets and complicated architectures so that bugs
are easy to spot.

**Step 2 – Increase data size and add batching**

Next, you replace the toy dataset with a larger one—perhaps ten or a hundred times bigger. A blog
on scaling up machine learning notes that once you move to “millions or even billions of rows,”
memory and I/O become major challenges and you need efficient strategies such as batching and
incremental loading (Khan, 2024).

In a ``probly`` context, this step usually includes:

- introducing **mini-batching** so only a subset of the data is in memory at once,
- moving from in-memory lists to a proper data pipeline (e.g. a generator, ``DataLoader``, or
  ``tf.data``),
- keeping the model structure almost the same, so you can attribute changes in behaviour
  primarily to the increased data size.

You now monitor:

- whether memory usage stays within limits,
- how runtime per training step changes,
- whether metrics behave similarly to the small-data case (if not, you investigate why).

If the model no longer fits easily in memory, you adapt batch sizes and streaming strategies
rather than rewriting the whole model at once.

**Step 3 – Scale the model and optimisation**

Once data handling is under control, you may want a more expressive model: deeper networks,
richer likelihoods, or more hierarchical structure. Research on scaling up algorithms suggests
that, as complexity grows, you need to be thoughtful about approximate methods, incremental
updates, and data subsampling (Domingos & Hulten, 2002).

Typical adjustments in this phase include:

- **richer model structure**: adding hierarchy, more parameters, or additional sub-models,
- **stronger regularisation**: to keep the larger model from overfitting,
- **more advanced optimisation**: switching from basic settings to tuned learning rates, schedulers,
  or different inference algorithms.

You also start to care about **hardware utilisation**: making use of vectorisation and JIT
compilation where the backend allows it (see Section 3.4). Proper profiling can tell you whether
time is spent in the model, the data pipeline, or in overhead.

**Step 4 – Move towards “production-like” runs**

In the final step of this case study, the same conceptual model is run under conditions closer to
a real large-scale experiment:

- full-sized training and validation sets,
- realistic batch sizes and number of epochs,
- logging, monitoring, and checkpointing turned on.

Checklists for training deep learning models emphasise that, at this stage, you need systematic
checks for data quality, experiment tracking, and evaluation (Alvi, 2024; Murtuzova, 2024).
Similarly, the ML Test Score rubric proposes dozens of concrete tests and monitors for assessing
production readiness (Breck et al., 2017).

For a large ``probly`` run, you now:

- track key metrics and system statistics over time,
- watch for instabilities (e.g. sudden spikes in loss or NaNs),
- ensure that checkpoints and seeds are stored so you can reproduce or resume the run if needed.

The important point is that you do **not** jump directly from a tiny prototype to the full
large-scale setup. Instead, you gradually expand data, model, and infrastructure, verifying at
each step that the system still behaves in a controlled way.

3.6 Checklist: preparing a large model run

Before you launch a big, potentially expensive model run, it helps to have a simple checklist.
Practical guides and checklists for deep learning stress that overlooking basic steps—like data
validation, metric definitions, or logging—often causes more trouble than the choice of
architecture itself (Alvi, 2024; Murtuzova, 2024).
Google’s “Rules of Machine Learning” and the ML Test Score similarly focus on robust pipelines and
testing rather than exotic algorithms (Breck et al., 2017; Zinkevich, n.d.).

The checklist below is adapted from these sources, but phrased in a way that fits ``probly`` use
cases.

**Data and problem definition**

- Have you clearly defined the prediction task, target, and evaluation metric(s)?  
- Is your training data validated for basic issues (missing values, label errors, obviously wrong
  ranges)? (Alvi, 2024)  
- Do you have separate training, validation, and test splits, and do you understand how they were
  created?  
- If you are using very large datasets, have you checked that your streaming/batching pipeline
  actually covers the full data, not just a subset?

**Model and code**

- Is the model architecture documented at a high level (inputs, key components, outputs)?  
- Have you already run the same model on a **smaller** dataset and confirmed that it trains and
  evaluates correctly? (Zinkevich, n.d.; Karpathy, 2019)  
- Are custom transformations, likelihoods, and priors covered by basic unit tests (e.g. round-trip
  checks, shape checks)? (see Section 2.6; Breck et al., 2017)  
- Is there a clear configuration mechanism (e.g. config files) that separates code from run-time
  settings (batch size, learning rate, number of epochs, etc.)?

**Infrastructure and resources**

- Do you know which hardware you will use (CPU, GPU(s), TPU), and have you confirmed that the
  model fits into memory with the planned batch size? (Khan, 2024)  
- Have you tested a short “smoke test” run (for example, a few batches or one epoch) on the target
  hardware?  
- Is checkpointing enabled so that long runs can be resumed after interruptions?  
- If running distributed or multi-GPU training, have you verified that all workers see consistent
  data and configurations?

**Monitoring, logging, and experiment tracking**

- Are you logging key training metrics (loss, accuracy, calibration metrics, etc.) and system
  metrics (runtime per step, memory usage)?  
- Do you have a central place (e.g. experiment tracker or dashboard) where runs and their configs
  are stored? MLOps best-practices emphasise systematic tracking of experiments and models as a
  foundation for reliable systems (Neptune.ai, 2021).  
- Is there a basic monitoring plan for long runs (alerts for divergence, NaNs, or stalled progress)?
  Guidance on ML system monitoring suggests that you should keep an eye on data quality, model
  performance, and operational metrics, not just final accuracy (Breck et al., 2017;
  Zinkevich, n.d.; Neptune.ai, 2021).  

**Reproducibility and governance**

- Are random seeds, library versions, and hardware details recorded so that important runs can be
  reproduced?  
- Is the training data snapshot (or at least its exact version and location) documented?  
- Do you know which model artefacts (weights, configuration, logs) must be kept for later analysis
  or deployment?

**Pre-launch sanity checks**

Before committing to a long and expensive run, it is useful to answer a few final questions:

- If this run fails or produces unusable results, do you have a clear next step?  
- Is there a simpler or cheaper “trial run” you can do first (fewer epochs, smaller dataset, fewer
  parameters)?  
- Do you have clear success criteria for this experiment (for example: “improve AUROC by at least
  2 percentage points on the validation set without degrading calibration”)?

Papers on ML production readiness note that having explicit tests and criteria greatly reduces the
risk of shipping fragile systems (Breck et al., 2017).
In the ``probly`` context, this checklist helps ensure that when you finally press “run” on a large
model, you are using your compute budget wisely and can trust the results.

4. Integration with Other Frameworks
------------------------------------

This chapter assumes that you often want to use ``probly`` together with other tools:
neural-network libraries, data pipelines, or classical ML components. The goal is not to cover
every integration pattern in detail, but to give you a mental model of how ``probly`` fits into
larger systems and what to watch out for when wiring different libraries together.

.. note::

   At the moment, ``probly`` provides first-class helpers and maintained examples for
   **PyTorch** and **Flax/JAX** only.

   References to other libraries in this section (such as TensorFlow, ``tf.data``,
   TensorFlow Probability, or scikit-learn) are intended as *conceptual* integration
   patterns or ideas for future extensions, **not** as built-in, officially supported
   backends.

4.1 General integration concepts

When integrating ``probly`` with other frameworks (Flax, TensorFlow, scikit-learn, etc.), three
recurring themes show up:

- how **data flows** between components,  
- how **types, shapes, and devices** are handled,  
- how **randomness and seeding** are coordinated.

**Data flow between our library and other libraries**

At a high level, ``probly`` consumes and produces arrays: batches of inputs, parameter vectors,
uncertainty representations, and so on. Other frameworks do the same, but often with their own
array types:

- JAX / Flax use JAX arrays and pytrees,  
- TensorFlow uses ``tf.Tensor`` and ``tf.data.Dataset`` objects (TensorFlow, 2024a),  
- scikit-learn expects NumPy arrays or array-like structures as inputs (scikit-learn Developers,
  2024).

The main integration task is to **convert between these representations in a controlled way**.
This typically means:

- deciding in which library the “main” computation lives (for example, running models in JAX and
  converting data from TensorFlow or NumPy when needed),  
- minimising the number of conversions (for example, not converting back and forth inside inner
  training loops),  
- keeping a clear boundary in your code where data moves from one framework to another.

**Type, shape, and device (CPU/GPU) considerations**

Array libraries are strict about shapes and dtypes:

- TensorFlow’s ``tf.data`` pipeline builds datasets as sequences of elements where each element
  has a fixed structure and shape (TensorFlow, 2024a),  
- scikit-learn’s estimator API assumes 2D feature matrices of shape
  ``(n_samples, n_features)`` for most supervised learning tasks (scikit-learn Developers, 2024),  
- Flax / JAX models typically assume explicit batch dimensions and well-defined dtypes.

When integrating with ``probly``, it helps to:

- decide on a **canonical shape convention** (for example: leading batch dimension, channel
  ordering, etc.),  
- standardise dtypes (e.g. always using ``float32`` unless there is a reason to do otherwise),  
- ensure that arrays live on the right **device** (CPU vs GPU/TPU) before calling library
  functions.

Moving data between devices (CPU ↔ GPU) is often more expensive than moving it between Python
functions in the same library, so it is worth designing your integration to minimise those hops.

**Randomness and seed management across frameworks**

Different frameworks handle randomness differently:

- JAX uses **explicit PRNG keys**. You create a key from an integer seed and then split it as you
  need more randomness, rather than relying on a global RNG (JAX Authors, n.d.).  
- Flax builds on this system and treats RNG streams as part of the module’s state and lifecycle
  (Flax Developers, n.d.-a, n.d.-b).  
- TensorFlow and NumPy use more traditional global or graph-local RNGs, controlled by functions
  such as ``tf.random.set_seed`` or ``numpy.random.seed`` (TensorFlow, 2024a; JAX Authors, n.d.).

When combining these with ``probly``, a good rule of thumb is:

- pick one library (often JAX in a Flax/``probly`` setup) as the “source of truth” for randomness,  
- derive and pass keys or seeds *into* other parts of the system, instead of letting each library
  silently manage its own global RNG,  
- log the seeds/keys used for important experiments so that runs can be reproduced.

4.2 Using ``probly`` with Flax

Flax is a neural-network library built on top of JAX. It provides a **Module** abstraction that
manages parameters, state, and randomness in a structured way (Flax Developers, n.d.-a,
n.d.-b). This makes it a natural companion for ``probly`` when you need neural networks as part
of a probabilistic model.

**Typical workflow: Flax for neural nets, ``probly`` for probabilistic parts**

A common pattern looks like this:

1. Define a Flax model (for example, an encoder or feature extractor) as a Linen Module.  
2. Initialise the Flax model to obtain a **variables dict** that contains parameters and any
   state (e.g. batch statistics) (Flax Developers, n.d.-a).  
3. Define a ``probly`` model that takes the Flax outputs (representations, logits, etc.) as
   inputs to probabilistic components (likelihoods, uncertainty representations, priors).  
4. Build a joint training or inference loop that updates both Flax parameters and ``probly``
   parameters in a consistent way.

Flax documentation emphasises the separation between **computation** and **parameters/state**:
“Modules offer a Pythonic abstraction … that have state, parameters and randomness on top of JAX”
(Flax Developers, n.d.-b). ``probly`` can treat these parameters as part of a larger probabilistic
model, while reusing Flax’s building blocks for the deterministic parts.

**Sharing parameters and state**

In a joint Flax+``probly`` setup, you often want to:

- keep all learnable quantities (Flax parameters, ``probly`` parameters) in a single **PyTree**
  so that optimisers can see and update everything,  
- clearly separate **deterministic state** (e.g. batch statistics) from **stochastic parameters**
  so that inference algorithms know what should be treated as random.

A practical approach is to:

- treat the Flax ``variables["params"]`` as one subset of the parameters,  
- treat ``probly``’s own parameters as another subset,  
- design a small utility that packs/unpacks these subsets from a combined structure passed into
  optimisers and inference routines.

**PRNG handling and common gotchas**

Both Flax and JAX rely on explicit PRNG keys; the JAX random-number documentation highlights that
keys are *pure values* and must be split to produce independent streams (JAX Authors, n.d.).
Flax modules provide helper methods such as ``make_rng`` to obtain new keys inside modules
(Flax Developers, n.d.-b).

Typical gotchas include:

- accidentally reusing the same key across multiple calls (leading to correlated “random” values),  
- mixing global RNGs (e.g. ``numpy.random``) with JAX/Flax keys in a way that is hard to
  reproduce,  
- forgetting to thread keys through ``probly``’s probabilistic components when they need
  randomness.

A robust integration treats PRNG keys just like any other part of the model state: they are
explicit, passed around deliberately, and included in experiment logs when reproducibility
matters.

4.3 Using ``probly`` with TensorFlow

.. note::

   ``probly`` does **not** currently ship an official TensorFlow backend or ready-made
   integration module. This subsection describes how you *could* wire ``probly`` together
   with TensorFlow and ``tf.data`` in your own projects, by analogy with other frameworks.

TensorFlow provides a powerful ecosystem for building data pipelines, training loops, and
serving infrastructure. A custom integration with ``probly`` would usually focus on **using
TensorFlow for data and training orchestration**, while letting ``probly`` handle probabilistic
modelling.

**Passing TensorFlow tensors and datasets into ``probly``**

TensorFlow’s ``tf.data`` API introduces a ``tf.data.Dataset`` abstraction that represents a
sequence of elements, where each element consists of one or more tensors (TensorFlow, 2024a).
You can create datasets from memory, TFRecord files, and many other sources, and then apply
transformations like ``map`` and ``batch`` to build efficient pipelines.

In a TensorFlow+``probly`` workflow, a typical pattern is:

- build a ``tf.data.Dataset`` that yields batches of inputs and labels (TensorFlow, 2024a),  
- inside a Python or ``tf.function`` training loop, convert each batch into the array type
  expected by ``probly`` (for example, NumPy arrays or JAX arrays),  
- call the ``probly`` model or log-likelihood on these batches,  
- optionally convert results (e.g. predictions, uncertainty measures) back to TensorFlow tensors
  if you want to integrate with other TF components.

**Integrating with TensorFlow training loops**

You can integrate ``probly`` with TensorFlow training in several ways:

- treat ``probly`` as a **black-box model**: in each training step, get a batch from
  ``tf.data``, convert it, run ``probly``, and update parameters using your chosen optimiser;  
- embed ``probly`` calls inside a ``tf.function`` for performance, as long as the conversions and
  control flow are compatible with TensorFlow’s tracing model;  
- use TensorFlow’s metrics and logging (e.g. TensorBoard) to monitor losses and uncertainty
  metrics produced by ``probly``.

Performance guides for ``tf.data`` stress the importance of overlapping input-pipeline work with
model execution using operations like ``prefetch`` and careful pipeline construction
(TensorFlow, 2024b). The same advice applies here: make sure your input pipeline is not the
bottleneck when calling into ``probly``.

**Known limitations and patterns**

Because TensorFlow and JAX/NumPy have different execution models and device handling, there are
trade-offs in any such custom integration:

- cross-framework calls introduce overhead and can complicate gradient computation,  
- some advanced TensorFlow features (e.g. distribution strategies) may not work smoothly if the
  core model lives outside of TensorFlow,  
- it is often simpler to use TensorFlow mainly for **data pipelines and infrastructure**, while
  keeping the heavy numerical work inside a single array framework that ``probly`` is built on.

4.4 Using ``probly`` with scikit-learn

.. note::

   ``probly`` does not currently include a built-in scikit-learn wrapper. The patterns in
   this subsection show how you could implement your **own** adapter class that follows the
   standard estimator API.

scikit-learn provides a standard **estimator interface**—objects with ``fit``, ``predict``,
and often ``score`` methods—plus tools like ``Pipeline`` and ``GridSearchCV`` for combining and
tuning models. The scikit-learn developer guide describes ``fit`` as the method “where the
training happens” and specifies that it should take the training data ``X`` and (for supervised
learning) ``y`` as inputs and return the estimator itself (scikit-learn Developers, 2024).

To integrate ``probly`` into this ecosystem, you can wrap a ``probly`` model in a thin
scikit-learn-style estimator.

**Wrapping a ``probly`` model as an estimator**

A minimal wrapper class might:

- accept configuration arguments in ``__init__`` (for example, model structure, prior settings,
  inference method),  
- implement ``fit(X, y=None)`` to run ``probly``’s training or inference on the data,  
- implement ``predict(X)`` or ``predict_proba(X)`` to return point predictions or uncertainty
  summaries,  
- optionally implement ``score(X, y)`` using scikit-learn’s metrics or your own metric.

This follows the standard estimator design described in scikit-learn’s developer documentation
(scikit-learn Developers, 2024) and allows your custom ``probly`` estimator to be used with tools
such as ``cross_val_score`` and ``cross_validate`` for evaluation (scikit-learn, 2024a).

**Using ``probly`` in pipelines and cross-validation**

Once wrapped, your ``probly`` estimator can be plugged into scikit-learn’s ``Pipeline``, which is
defined as “a sequence of data transformers with an optional final predictor” (scikit-learn,
2024b). Pipelines make it easy to:

- chain preprocessing steps (e.g. scaling, feature selection) with your ``probly`` estimator,  
- tune hyperparameters across all steps using ``GridSearchCV`` or ``RandomizedSearchCV``,  
- evaluate the whole pipeline with cross-validation, ensuring that preprocessing is learned only
  on training folds (scikit-learn, 2024a, 2024b).

From ``probly``’s perspective, the key requirement is simply that your wrapper behaves like a
scikit-learn estimator: support the right methods and follow the standard conventions for inputs
and attributes.

4.5 Interoperability best practices

When connecting ``probly`` to other frameworks, a few general best practices help avoid
frustrating bugs and performance issues.

**Device management (CPU/GPU)**

- Decide early which **devices** will run which parts of the system. For example, you might keep
  your Flax/``probly`` model entirely on GPU, while scikit-learn components run on CPU.  
- Minimise device transfers by grouping computations: move a batch *once* to GPU, perform all
  relevant model calls there, and only bring back aggregated results.  
- When using libraries like TensorFlow’s ``tf.data`` on GPU, follow their performance guidelines
  (e.g. prefetching, parallel mapping) to keep accelerators fully utilised (TensorFlow, 2024b).

**Version compatibility tips**

- Pin versions of core libraries (JAX, Flax, TensorFlow, scikit-learn, etc.) in your environment
  and CI configuration. Many of these libraries publish compatibility notes (for example, which
  JAX versions are supported by a given Flax release).  
- Avoid mixing very old and very new versions of closely related libraries, as subtle API changes
  (especially around randomness and device placement) can cause integration problems.  
- Document the versions used for key experiments so that collaborators can recreate your setup.

**Debugging errors across library boundaries**

Cross-library bugs are often about **mismatched assumptions**: incorrect shapes, wrong dtypes,
inconsistent devices, or misaligned RNG handling. When debugging:

- start with a very small example that uses the integration boundary only (for example, one batch
  flowing from a ``tf.data.Dataset`` into a ``probly`` model),  
- inspect and log shapes, dtypes, and devices right before and after conversion points,  
- temporarily disable advanced features (JIT compilation, complex pipelines) to reduce the search
  space,  
- re-run with fixed seeds and controlled randomness to see if errors are deterministic (JAX
  Authors, n.d.; TensorFlow, 2024a).

By treating the integration points as first-class components—carefully designed, tested, and
documented—you can combine ``probly`` with other frameworks without turning your project into a
black box.


5. Performance & Computational Efficiency
-----------------------------------------

5.1 Understanding performance bottlenecks

When a model feels “slow”, the first step is to understand **where the time is actually
spent**. For typical ``probly`` workflows, the main bottlenecks are:

- **CPU compute**: scalar Python loops, non-vectorised NumPy operations, or expensive
  Python-level bookkeeping.
- **GPU compute**: large matrix multiplications or convolutions that fully occupy the GPU.
- **I/O**: loading data from disk, the network, or very slow preprocessing.
- **Python overhead**: frequent Python function calls, dynamic graph construction, or
  heavy logging that prevents libraries from executing efficiently in compiled code.

Profiling tools help to diagnose these issues. For example, the standard Python profilers
collect “statistics that describe how often and for how long various parts of the program
executed” (Python Software Foundation, n.d.), which makes it easier to see whether time is
dominated by your model code, the data pipeline, or external libraries.

In practice, it is useful to adopt a simple routine:

- run a **small experiment** with realistic settings,
- profile the run to identify the **slowest functions and lines**,
- focus optimisation efforts only on the few hot spots that clearly dominate runtime.

5.2 Profiling your ``probly`` code

Profiling does not have to be complicated. For many questions, it is enough to:

- use a **function-level profiler** (e.g., ``cProfile``) to find the most expensive calls
  (Python Software Foundation, n.d.),
- complement this with a **line-level or memory profiler** when you suspect specific
  sections of code are responsible for high memory usage or unexpected slowdowns.

A practical workflow might look like this:

1. Wrap the main training / inference loop in a profiler context.
2. Run a short experiment on a subset of the data.
3. Sort the profiler output by **cumulative time** to find the most expensive functions.
4. For one or two of these functions, use a line profiler or targeted logging to drill
   down further.

The goal is not to micro-optimise everything, but to answer concrete questions such as:

- Is the time spent mostly in ``probly`` / NumPy / JAX, or in custom Python code?
- Is data loading or preprocessing slower than the actual model computations?
- Do a few functions account for most of the runtime?

Once you know this, the optimisation strategy usually becomes obvious.

5.3 Algorithmic improvements

Before tuning low-level details, it is often more effective to change the **algorithm**
itself:

- **Choose inference methods suited to your model.** Some models work well with simple
  optimisation-based approaches, while others require more expressive samplers. Methods
  with better convergence behaviour can dramatically reduce total runtime, even if each
  step is slightly more expensive.

- **Simplify or re-parameterise the model.** Alternative parameterisations can improve
  gradient flow, reduce pathological curvature, or make constraints easier to handle. This
  often leads to faster convergence and fewer required iterations.

- **Re-use previous runs.** Warm starts, cached results, or saved initialisations can
  avoid repeating expensive computations. For example, you might start a new experiment
  from the parameters of a previous run with similar settings instead of reinitialising
  from scratch.

Many performance problems disappear once the model and inference method are well aligned
with the task.

5.4 Vectorisation & parallelisation

Low-level performance often comes from **doing more work per call** rather than adding
more explicit loops. Numerical Python libraries such as NumPy are designed so that
vectorised operations “push” work into efficient compiled kernels instead of looping in
pure Python (Harris et al., 2020). In the context of ``probly``, this means:

- prefer **batch operations** over explicit Python loops,
- structure code so that entire arrays of parameters, samples, or observations can be
  processed at once,
- let the underlying backend (NumPy, JAX, etc.) take advantage of SIMD, multi-core, or
  GPU execution.

Vectorisation can be combined with **parallelisation**. For example, parallelising
independent chains or tasks across CPU cores or devices can further reduce wall-clock
time, provided that:

- the cost of launching parallel jobs and synchronising results is small compared to the
  work done per task,
- memory usage remains within the limits of each device,
- the random seeds and PRNG handling are designed to keep chains statistically
  independent (Open Data Science, 2019).

The trade-off is that more parallelism is not always better: beyond a certain point, the
overhead can outweigh the benefits, especially for small models or very short runs.

5.5 Reproducibility & randomness

Randomness is essential to many probabilistic methods, but it can also make performance
difficult to reason about. To keep experiments reproducible:

- **Set random seeds deliberately.** Using fixed seeds for NumPy, JAX, and other
  backends ensures that repeated runs with the same configuration produce comparable
  results (Open Data Science, 2019).

- **Log all relevant settings.** This includes seeds, dataset versions, batch sizes,
  hardware configuration, and important hyperparameters.

- **Balance reproducibility and exploration.** During debugging or benchmarking, fixed
  seeds are helpful. For final experiments, it may be preferable to run multiple
  independent seeds to understand variability.

Reproducibility is not just about fairness in comparison; it also makes performance
optimisation much easier, because you can be confident that changes in runtime are due to
code changes rather than random fluctuations.

5.6 Performance checklist

Before launching a large, expensive run, it is useful to walk through a short checklist:

- **Model & algorithm**
  - Is the chosen inference method appropriate for the model structure?
  - Are there unnecessary layers, parameters, or transformations that could be removed?

- **Implementation**
  - Are the main computations vectorised rather than written as Python loops?
  - Have you avoided repeated work, such as recomputing static quantities inside the main loop?

- **Data pipeline**
  - Is data loading and preprocessing fast enough compared to the model computation?
  - Are you using batching or mini-batching where appropriate?

- **Resources**
  - Is the model configured to use available hardware (CPU cores, GPU, memory) sensibly?
  - Is logging kept at a reasonable level so it does not become an I/O bottleneck?

- **Reproducibility**
  - Are random seeds set and logged?
  - Can you reliably reproduce a small profiling run before scaling up?

Answering these questions ahead of time helps avoid wasted compute and makes it easier to
interpret the results of large experiments.

6. Advanced Usage Patterns & Recipes
------------------------------------

6.1 Common advanced modeling patterns

This section sketches a few common “advanced” modelling patterns that often appear in real
projects. The goal is not to give full mathematical detail, but to show how these ideas fit
conceptually with ``probly`` and where they are typically useful.

**Hierarchical models**

Hierarchical (or multilevel) models are used when data are organised in groups, levels, or
contexts—for example students within classes, patients within hospitals, or measurements for
multiple machines. Instead of fitting a separate model to each group, a hierarchical model
shares information across groups via higher-level parameters. This “partial pooling” helps
stabilise estimates, especially when some groups have only a few observations (Gelman & Hill,
2007).

In ``probly``, hierarchical models can be expressed by:

- defining group-specific parameters (e.g. intercepts or slopes),
- tying them together through shared hyperparameters,
- using uncertainty representations to see how much information is borrowed across groups.

This pattern is particularly useful when you care about both overall trends and group-level
differences.

**Mixture models**

Mixture models assume that the data come from a combination of several latent components, such
as different customer types, regimes, or clusters. A simple example is a Gaussian mixture
model, where each data point is generated from one of several Gaussian components with its own
mean and variance (Bishop, 2006).

In ``probly``, you can:

- represent component-specific parameters and their mixing weights,
- use discrete or continuous latent variables to indicate which component generated each observation,
- quantify uncertainty about both the component assignments and the component parameters.

Mixture models are helpful when a single simple distribution is not flexible enough to describe
your data.

**Time-series and sequential models**

Time-series and sequential models capture data that arrive in order, such as sensor readings,
financial prices, or user activity over time. Typical goals include forecasting the future,
detecting regime changes, or understanding temporal structure (Hyndman & Athanasopoulos,
2018).

With ``probly``, you can:

- build models that include lagged variables, latent states, or dynamic parameters,
- express uncertainty about future trajectories, not just point forecasts,
- plug the resulting uncertainty into downstream decision-making or risk analyses.

Often, advanced time-series models combine ideas from hierarchies (e.g. many related series)
and mixtures (e.g. different regimes or behaviours).

6.2 Reusable templates

As your models become more complex, it is helpful to identify **reusable templates**—small
patterns that keep recurring across projects. Examples include:

- a standard hierarchical regression block for grouped data,
- a generic mixture-of-experts block that combines several prediction heads,
- a time-series forecasting head that can be attached to different feature extractors.

In ``probly``, these templates can be written as functions or modules that:

- take in model-specific pieces (e.g. feature networks, priors, or likelihood choices),
- expose a clean, well-documented interface,
- return both predictions and uncertainty representations in a consistent format.

By reusing such templates, you reduce boilerplate, keep designs more uniform across projects,
and make it easier for others to understand or extend your work.

6.3 Pointers to examples

To make these patterns concrete, it is useful to link each abstract idea to a **worked
example**:

- For hierarchical models, an example with grouped data (e.g. “schools”, “hospitals”, or
  “stores”) that walks through model specification, inference, and interpretation.
- For mixture models, a clustering or anomaly-detection example that shows how component
  responsibilities and uncertainty can be visualised.
- For time-series models, a forecasting example that compares point forecasts to predictive
  intervals over time.

In the long run, the aim is that each advanced pattern described here corresponds to at least
one notebook in the *Examples & Tutorials* section, so that readers can jump directly from the
conceptual description to runnable code.

7. Summary
----------

7.1 Key takeaways

This chapter pulled together the “advanced” parts of working with ``probly``. Here are the
most important ideas to remember:

- **Think in workflows, not one-off runs.**  
  You rarely get the model right on the first attempt. Start simple, run it, look at what
  goes wrong, and then refine. Advanced topics are mostly about having good tools for
  iterating in a controlled way.

- **Use transformations to tame tricky parameter spaces.**  
  Transformations let you express models in natural, human-friendly parameters while keeping
  inference in a convenient unconstrained space. Custom transforms are the place to encode
  constraints, reparameterisations, and numerical tricks so the rest of the model stays clean.

- **Structure your code for large models and datasets.**  
  As things grow, clear modular structure matters as much as the math: separate data loading,
  model definition, and inference; avoid giant monolithic scripts; and reuse building blocks
  across projects.

- **Lean on vectorisation, batching, and compilation.**  
  Performance usually comes from doing more work per call, not from clever loops. Writing
  models in a vectorised style and using backend compilation options (where available) can
  make the difference between a toy demo and a practical large-scale run.

- **Integrate carefully with other frameworks.**  
  When combining ``probly`` with Flax, TensorFlow, or scikit-learn, be explicit about how
  data, shapes, devices (CPU/GPU), and random seeds move across boundaries. Clear integration
  points make complex systems much easier to debug.

- **Test, profile, and document advanced pieces.**  
  Custom transformations, large-model setups, and multi-framework integrations deserve small
  dedicated tests and occasional profiling runs. A few well-placed checks (round-trip tests,
  shape checks, smoke tests) catch many subtle bugs before they become expensive.

- **Favour clarity and robustness over cleverness.**  
  An “advanced” model is only useful if people can understand, trust, and maintain it. Simple,
  well-structured models with honest uncertainty are usually more valuable than fragile,
  over-complicated constructions.

If you keep these principles in mind, the rest of the ``probly`` documentation—methods,
modules, and examples—should slot naturally into your own advanced models and experiments.


.. bibliography::
Alvi, F. (2024, November 6).
Deep learning model training checklist: Essential steps for building and deploying models.
OpenCV.
https://opencv.org/blog/deep-learning-model-training/

Bhuva, L. (2024, November 30).
Mini-batch gradient descent: A comprehensive guide.
Medium.
https://medium.com/@lomashbhuva/mini-batch-gradient-descent-a-comprehensive-guide-ba27a6dc4863

Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.
https://www.springer.com/gp/book/9780387310732

Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017).
The ML test score: A rubric for ML production readiness and technical debt reduction.
In *Proceedings of the IEEE International Conference on Big Data* (pp. 1123–1132).
IEEE.
https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/

Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M.,
Brubaker, M. A., Guo, J., Li, P., & Riddell, A. (2017). Stan: A probabilistic programming
language. *Journal of Statistical Software, 76*(1), 1–32. https://doi.org/10.18637/jss.v076.i01

Contributors to the Pyro Project. (2019).
*NumPyro: numpyro.distributions.constraints* [Source code documentation].
NumPyro. https://num.pyro.ai/en/0.3.0/_modules/numpyro/distributions/constraints.html

Contributors to the Pyro Project. (2020). *NumPyro documentation* (Version 0.3.0). NumPyro.
https://num.pyro.ai/_/downloads/en/0.3.0/pdf/

Domingos, P., & Hulten, G. (2002).
A general method for scaling up machine learning algorithms and its application to clustering.
In *Proceedings of the Eighteenth International Conference on Machine Learning* (pp. 106–113).
Morgan Kaufmann.
https://dl.acm.org/doi/10.5555/645530.658293

Dutta, S., Legunsen, O., Huang, Z., & Misailovic, S. (2018). Testing probabilistic programming
systems. In *Proceedings of the 26th ACM Joint European Software Engineering Conference and
Symposium on the Foundations of Software Engineering (ESEC/FSE ’18)* (pp. 574–586).
Association for Computing Machinery. https://doi.org/10.1145/3236024.3236057

Eertmans, J. (2025). *NaN and infinite values*. In *DiffeRT documentation*.  
https://differt.eertmans.be/latest/nans_and_infs.html

Flax Developers. (n.d.-a).
Managing parameters and state.
In *Flax Linen fundamentals*.
https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/state_params.html

Flax Developers. (n.d.-b).
The Flax Module lifecycle.
In *Flax Linen developer notes*.
https://flax-linen.readthedocs.io/en/latest/developer_notes/module_lifecycle.html

GeeksforGeeks. (2025, September 30).
Mini-batch gradient descent in deep learning.
GeeksforGeeks.
https://www.geeksforgeeks.org/deep-learning/mini-batch-gradient-descent-in-deep-learning/

Gelman, A., & Hill, J. (2007). *Data analysis using regression and multilevel/hierarchical models*.
Cambridge University Press.
https://www.cambridge.org/core/books/data-analysis-using-regression-and-multilevelhierarchical-models/0C1C3F8F5E6C5D7D5C7D40D5D6A50F5F

Griewank, A., & Walther, A. (2008). *Evaluating derivatives: Principles and techniques of
algorithmic differentiation* (2nd ed.). Society for Industrial and Applied Mathematics.  
https://doi.org/10.1137/1.9780898717766

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P.,
Cournapeau, D., … Oliphant, T. E. (2020). Array programming with NumPy.
*Nature*, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

Hey Amit. (2024, May 10).
When to set pin_memory to True in PyTorch.
Medium.
https://medium.com/data-scientists-diary/when-to-set-pin-memory-to-true-in-pytorch-75141c0f598d

hvgotcodes. (2012, March 29). *What are some good practices for unit testing probability
distributions?* Stack Overflow. https://stackoverflow.com/questions/9934903

Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and practice* (2nd ed.).
OTexts. https://otexts.com/fpp2/

JAX Authors. (n.d.-a).
A brief introduction to JAX.
JAX documentation.
https://jax.exoplanet.codes/en/latest/tutorials/introduction-to-jax/

JAX Authors. (n.d.-b).
Just-in-time compilation.
JAX documentation.
https://docs.jax.dev/en/latest/jit-compilation.html

JAX Authors. (n.d.-c).
JAX: Composable transformations of Python+NumPy programs.
GitHub.
https://github.com/jax-ml/jax

JAX Authors. (n.d.).
Pseudorandom numbers.
In *JAX documentation*.
https://docs.jax.dev/en/latest/random-numbers.html

Jason Brownlee. (2019, August 19).
A gentle introduction to mini-batch gradient descent and how to configure batch size.
Machine Learning Mastery.
https://www.machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/

Jaxley Team. (n.d.).
Speeding up simulations with JIT-compilation and GPUs.
Jaxley documentation.
https://jaxley.readthedocs.io/en/v0.4.0/tutorials/04_jit_and_vmap.html

Karanam, A., Mathur, S., Sidheekh, S., & Natarajan, S. (2024).
A unified framework for human-allied learning of probabilistic circuits.
*arXiv.* https://arxiv.org/abs/2405.02413

Karpathy, A. (2019, April 25).
A recipe for training neural networks.
https://karpathy.github.io/2019/04/25/recipe/

Khan, F. T. (2024, October 11).
Scaling up machine learning: Efficient strategies for handling large datasets.
Medium.
https://medium.com/@ftech/scaling-up-machine-learning-efficient-strategies-for-handling-large-datasets-1d329c608470

Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *Proceedings of the 2nd
International Conference on Learning Representations (ICLR).* https://arxiv.org/abs/1312.6114

Koul, N. (2023, March 3).
JAX — A beginner’s tutorial.
Medium.
https://medium.com/@nimritakoul01/jax-a-beginners-tutorial-ca09b25a3f56

LuxDevHQ. (2023, August 28).
Generic folder structure for your machine learning projects.
DEV Community.
https://dev.to/luxdevhq/generic-folder-structure-for-your-machine-learning-projects-4coe

Martire, S. (2022, March 12).
Optimizing a TensorFlow input pipeline: Best practices in 2022.
Medium.
https://medium.com/@virtualmartire/optimizing-a-tensorflow-input-pipeline-best-practices-in-2022-4ade92ef8736

Mittal, B. (2025, July 23). *Bayesian hierarchical models*. GeeksforGeeks.  
https://www.geeksforgeeks.org/machine-learning/bayesian-hierarchical-models/

Murtuzova, T. (2024, June 17).
Essential deep learning checklist: Best practices unveiled.
DEV Community.
https://dev.to/api4ai/essential-deep-learning-checklist-best-practices-unveiled-5gma

Neptune.ai. (2021, March 5).
MLOps checklist – 10 best practices for a successful model deployment.
Neptune Blog.
https://neptune.ai/blog/mlops-best-practices

Open Data Science. (2019, April 24). *Properly setting the random seed in ML experiments:
Not as simple as you might imagine*. OpenDataScience.com.
https://opendatascience.com/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine/

Open Source Vizier Authors. (2022).
*Bijectors*. In *Open Source Vizier documentation*.
https://oss-vizier.readthedocs.io/en/latest/advanced_topics/tfp/bijectors.html

Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). Normalizing flows for probabilistic modeling and inference. *Journal of Machine Learning Research, 22*(57), 1–64.  
https://www.jmlr.org/papers/volume22/19-1028/19-1028.pdf

Pati, S. K. (2025, March 27).
Best practices for organizing and coding data science projects — Part 1.
The Deep Hub.
https://medium.com/thedeephub/best-practices-for-organizing-and-coding-data-science-projects-part-1-72539e14a7a0

Python Software Foundation. (n.d.). *The Python profilers*. Python documentation.
https://docs.python.org/3/library/profile.html

PyImageSearch. (2023, February 27).
Learning JAX in 2023: Part 2 — JAX’s power tools: grad, jit, vmap, and pmap.
PyImageSearch.
https://pyimagesearch.com/2023/02/27/learning-jax-in-2023-part-2-jaxs-power-tools-grad-jit-vmap-and-pmap/

PyTorch Documentation. (2017, January 16).
CUDA semantics.
PyTorch.
https://docs.pytorch.org/docs/stable/notes/cuda.html

Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. *Proceedings of
the 32nd International Conference on Machine Learning (ICML), 37*, 1530–1538.
https://proceedings.mlr.press/v37/rezende15.html

Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. In *Proceedings of the 32nd International Conference on Machine Learning* (pp. 1530–1538). PMLR.  
https://proceedings.mlr.press/v37/rezende15.html

scikit-learn. (2024a).
3.1. Cross-validation: evaluating estimator performance.
In *scikit-learn 1.7.2 documentation*.
https://scikit-learn.org/stable/modules/cross_validation.html

scikit-learn. (2024b).
Pipeline.
In *scikit-learn 1.7.2 documentation*.
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

scikit-learn Developers. (2024).
Developing scikit-learn estimators.
In *scikit-learn 1.7.2 documentation*.
https://scikit-learn.org/stable/developers/develop.html

Stan Development Team. (n.d.). *10.2 Lower bounded scalar*. In *Stan reference manual* (Version 2.22).
Stan. https://mc-stan.org/docs/2_22/reference-manual/lower-bound-transform-section.html

Stan Development Team. (2025).
*Constraint transforms*. In *Stan reference manual* (Version 2.37).
https://mc-stan.org/docs/reference-manual/transforms.html

Stan Development Team. (2025).
*Stan reference manual* (Version 2.37).
https://mc-stan.org/docs/reference-manual/

TensorFlow. (2024a, August 15).
tf.data: Build TensorFlow input pipelines.
TensorFlow.
https://www.tensorflow.org/guide/data

TensorFlow. (2024a, August 15).
tf.data: Build TensorFlow input pipelines.
TensorFlow Guide.
https://www.tensorflow.org/guide/data

TensorFlow. (2024b, August 15).
Better performance with the tf.data API.
TensorFlow.
https://www.tensorflow.org/guide/data_performance

TensorFlow. (2024b, August 15).
Better performance with the tf.data API.
TensorFlow Guide.
https://www.tensorflow.org/guide/data_performance

TensorFlow Probability. (n.d.). *Module: tfp.bijectors*. In *TensorFlow Probability API documentation*. Retrieved December 5, 2025, from  
https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors

TensorFlow Probability. (2023). *Module: tfp.bijectors* [Computer software documentation].
TensorFlow. https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors

TensorFlow Probability. (2023).
*tfp.bijectors.Bijector* [Computer software documentation].
TensorFlow. https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Bijector

TensorFlow Probability. (2023). *tfp.bijectors.Softplus* [Computer software documentation]. TensorFlow.
https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Softplus

The Hitchhiker’s Guide to Python Contributors. (2024).
Structuring your project.
In *The Hitchhiker’s Guide to Python*.
https://docs.python-guide.org/writing/structure/

The JAX Authors. (2024). *Pseudorandom numbers*. JAX documentation.  
https://docs.jax.dev/en/latest/random-numbers.html

Tu, X. (2024).
An overview of large AI models and their applications.
*Visual Intelligence, 2*(1), Article 34.
https://doi.org/10.1007/s44267-024-00065-8

Tyagi, A. J. (2025).
Scaling deep learning models: Challenges and solutions for large-scale deployments.
*World Journal of Advanced Engineering Technology and Sciences, 16*(2), 10–20.
https://doi.org/10.30574/wjaets.2025.16.2.1252

vitkl. (2020, December 30). *Softplus transform as a more numerically stable way to enforce positive
constraint [Issue #855]*. In *NumPyro* (GitHub repository). GitHub.
https://github.com/pyro-ppl/numpyro/issues/855

vitkl. (2020, December 31).
*Softplus transform as a more numerically stable way to enforce positive constraint
[Issue #855]*. GitHub. https://github.com/pyro-ppl/numpyro/issues/855

Walton, S. (2023). *Isomorphism, normalizing flows, and density estimation: Preserving
relationships between data* (Area Exam Report No. AREA-202307-Walton). University of Oregon,
Department of Computer and Information Sciences.  
https://www.cs.uoregon.edu/Reports/AREA-202307-Walton.pdf

Zinkevich, M. (n.d.).
Rules of machine learning: Best practices for ML engineering.
Google Developers.
https://developers.google.com/machine-learning/guides/rules-of-ml
