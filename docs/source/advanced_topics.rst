Advanced Topics
===============

1. Overview
-----------

1.1 Purpose of this chapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This chapter explains:

- what “advanced” means in the context of ``probly``,
- when you should read this chapter (recommended after Core Concepts and Main Components).

1.2 Prerequisites & notation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before reading this chapter, the reader should already be familiar with:

- the concepts introduced in Core Concepts,
- the basic workflows described in Main Components,
- foundational ideas such as uncertainty representations, transformations, and inference.

For clarity, this chapter follows the same notation conventions used throughout the ``probly`` documentation.

1.3 Typical advanced use cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This chapter is intended for scenarios where users go beyond simple examples, such as:

- training or evaluating large or real-world models,
- dealing with tight performance or memory constraints,
- integrating ``probly`` into existing machine-learning pipelines.

These use cases often require a deeper understanding of transformations, scalability, and framework interoperability, which this chapter provides.

.. seealso::

    For background material, see :doc:`Core Concepts <core_concepts.rst>`.

    For the main building blocks of ``probly``, like the main transformations, utilities & layers, and evaluation tools, see :doc:`Main Components <main_components>`.

2. Custom Transformations
-------------------------

2.1 Recall: What is a transformation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``probly``, a **transformation** is a small building block that maps values between two spaces,
similar in spirit to the bijectors used in TensorFlow Probability :cite:`tfpBijectorSoftplus2023,rezendeVariationalFlows2015`:

- an **unconstrained space**, where optimisation and inference algorithms can work freely, and
- a **constrained space**, which matches the natural domain of your parameters or predictions
  (for example positive scales, probabilities on a simplex, or bounded intervals) :cite:`tfpBijectorSoftplus2023`.

Instead of forcing you to design models directly in a complicated constrained space, you write
your model in terms of meaningful parameters, and the transformation then takes care of the math
that keeps everything inside the valid domain :cite:`tfpBijectorSoftplus2023,rezendeVariationalFlows2015`.

In practice this means that transformations:

- provide a *short, reusable recipe* for how to turn raw latent variables into valid parameters,
- enable **reparameterisation**, which can make optimisation easier and gradients better behaved :cite:`kingmaAutoEncodingVB2014`,
- automatically enforce **constraints** such as positivity, bounds, or simplex structure :cite:`tfpBijectorSoftplus2023`.

You can think of a transformation as an adapter between “nice for the optimiser” coordinates and
“nice for the human” coordinates :cite:`kingmaAutoEncodingVB2014,rezendeVariationalFlows2015`.

2.2 When to implement your own?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built-in transformations in ``probly`` are designed to cover many common cases,
such as positive scales, simple box constraints, or mappings to probability vectors.
This is similar in spirit to other probabilistic frameworks that provide default
constraint transforms for bounded, ordered, simplex, correlation, or covariance
parameters :cite:`stanConstraintTransforms2025`. In many projects these standard building
blocks are sufficient and you never need to write your own transformation.

There are, however, important situations where a **custom transformation** is the
better choice.

- **Limitations of built-in transformations**

  Some models use parameter spaces that go beyond the usual catalogue of common constraints such as positive,
  bounded, or simplex parameters. For example, you may need structured covariance matrices,
  ordered-but-positive sequences, monotone functions, or parameters that satisfy
  several coupled constraints at once. The Stan reference manual notes that
  “vectors may … be constrained to be ordered, positive ordered, or simplexes”
  and matrices “to be correlation matrices or covariance matrices” in its section on
  constraint transforms :cite:`stanConstraintTransforms2025`, but real applications
  often demand more specialised structures. In such cases, a custom
  transformation lets you explicitly encode the structure your model needs.

- **Custom distributions or domain constraints**

  In many domains, prior knowledge is naturally expressed as constraints on
  parameters: certain probabilities must always sum to one, some effects must be
  monotone, or fairness and safety requirements restrict which configurations are
  admissible. Work on probabilistic circuits emphasises that domain
  constraints can encode general trends in the domain and
  serve as effective inductive biases :cite:`karanamHumanAlliedPCs2024`. A custom
  transformation is a convenient way to build such domain-specific rules into the
  parameterisation instead of relying on ad-hoc clipping or post-processing.

- **Cleaner uncertainty behaviour and numerical stability**

  Some parameterisations yield more interpretable and numerically stable
  uncertainty estimates than others. A classic example is working on a log or
  softplus scale for strictly positive parameters. Stan, for instance, uses a
  logarithmic transform for lower-bounded variables and applies the inverse
  exponential to map back to the constrained space :cite:`stanConstraintTransforms2025`.
  Practitioners have observed that replacing a naïve exponential with a softplus
  transform can substantially stabilise inference; one NumPyro user reports a
  very substantial improvement in inference stability when replacing an ``exp``
  transform with ``softplus`` for constraining ``site_scale`` :cite:`vitklSoftplusTransform2020`. In ``probly``, a custom transformation can encapsulate this kind of
  numerically robust parameterisation and make its effect on uncertainty
  representations easier to reason about.

- **Integration with existing code or libraries**

  When you plug ``probly`` into an existing machine-learning pipeline, external
  code often expects parameters in a fixed, domain-specific representation. The
  internal unconstrained parameterisation that is convenient for inference may
  not match what a legacy training loop, a deep-learning framework, or a
  production system “expects to see.” A transformation can act as a bridge:
  ``probly`` operates in its preferred unconstrained space, while the surrounding
  code continues to work with familiar application-level parameters, just as
  constraint transforms reconcile internal and external parameterisations in Stan :cite:`stanConstraintTransforms2025`.

As a practical rule of thumb: if you frequently add manual clamps, min/max
operations, or ad-hoc post-processing steps just to keep parameters valid, that is
a strong signal that a dedicated custom transformation would make the model
cleaner, more robust, and easier to maintain.

2.3 API & design principles
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom transformations in ``probly`` should follow a **small and predictable interface**. Similar
interfaces appear in other probabilistic libraries. For example, TensorFlow Probability notes
that a ``Bijector`` is characterised by three operations (forward, inverse, and a log-determinant
Jacobian) :cite:`tfpBijectorSoftplus2023`, and the Open Source Vizier guide adds
that each bijector implements at least three methods (``forward``, ``inverse``, and one of the
log-Jacobian methods) :cite:`ossvizierBijectors2022`.

Conceptually, each transformation in ``probly`` is responsible for three things:

- a **forward mapping** from an unconstrained input to the constrained parameter space,
  typically used to turn one random outcome into another :cite:`tfpBijectorSoftplus2023`,
- an **inverse mapping** that recovers the unconstrained value from a constrained one,
  enabling probability and density computations,
- any **auxiliary quantities** that inference algorithms may need, such as Jacobians or
  log-determinants, to account for the change of variables.

Stan’s transform system illustrates the same pattern: every (multivariate) parameter in a Stan
model is transformed to an unconstrained variable behind the scenes by the model compiler,
and the C++ classes include code to transform parameters from unconstrained to
constrained and apply the appropriate Jacobians :cite:`stanConstraintTransforms2025`. In other
words, the model is written in terms of constrained parameters, while inference operates in an
unconstrained space connected by well-defined forward and inverse transforms.

Beyond this minimal interface, good transformations follow several design principles:

- **local and self-contained**

  All logic that enforces a particular constraint should live inside the transformation. The rest
  of the model should not need to know which reparameterisation is used internally. This mirrors
  how libraries like Stan and NumPyro encapsulate constraints as self-contained objects that define
  where parameters are valid :cite:`numpyroConstraints2019,stanConstraintTransforms2025`.

- **clearly documented domain and range**

  It should be obvious which inputs are valid, what shapes are expected, and which constraints the
  outputs satisfy. NumPyro’s ``Constraint`` base class explicitly states that a constraint object
  represents a region over which a variable is valid and can be optimised :cite:`numpyroConstraints2019`. Documenting domains and ranges for custom
  transformations in ``probly`` serves the same purpose.

- **numerically stable**

  The implementation should avoid unnecessary overflow, underflow, or extreme gradients. Stan’s
  documentation on constraint transforms highlights numerical issues arising from floating-point
  arithmetic and the need for careful treatment of boundaries and Jacobian terms :cite:`stanConstraintTransforms2025`. In practice, this often means using stable variants of mathematical formulas,
  adding small epsilons, or applying safe clipping near boundaries.

- **composable**

  Whenever possible, transformations should work well in combination with others. TensorFlow
  Probability, for example, provides composition utilities such as ``Chain`` to build complex
  mappings out of simpler bijectors :cite:`tfpModuleBijectorsND`. In ``probly``, the same
  idea applies: designing transformations to be composable makes it easier to express rich
  constraints while keeping each individual component small and testable.

During **sampling and inference**, ``probly`` repeatedly calls the forward and inverse mappings of
your transformation to move between the internal unconstrained representation and the external
constrained parameters that appear in the model. A well-designed transformation therefore keeps
these operations cheap, stable, and easy to reason about, in line with the goals of similar
transform systems in Stan and TensorFlow Probability :cite:`stanConstraintTransforms2025,tfpBijectorSoftplus2023`.

2.4 Step-by-step tutorial: simple custom transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section walks through a minimal example of implementing a custom transformation in ``probly``.
The goal is not to show every detail of the library API, but to illustrate the typical workflow
from an initial idea to a working component that can be used inside a model.

**Problem description**

Suppose we want a parameter that must always be **strictly positive**, for example a scale or
standard deviation. Many probabilistic frameworks enforce such constraints by transforming from an
unconstrained real variable into a positive domain. For instance, the Stan reference manual notes
that Stan uses a logarithmic transform for lower and upper bounds :cite:`stanLowerBoundedScalarND`,
and TensorFlow Probability’s Softplus bijector is documented as having the positive real numbers
as its domain :cite:`tfpBijectorSoftplus2023`. Following the same idea, we introduce an
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
strictly positive parameters :cite:`stanLowerBoundedScalarND`, while TensorFlow Probability provides a
Softplus bijector which does not overflow as easily as the exponential bijector :cite:`tfpBijectorSoftplus2023`. NumPyro implements a similar idea with a dedicated
Softplus-based transform from unconstrained space to the positive domain in its
transforms module :cite:`numpyroTransforms2019`. In practice, this means you can choose
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
the library’s registry and then reused across models :cite:`numpyroTransforms2019,tfpBijectorSoftplus2023`. In ``probly``, registration plays the same role: it turns a single
implementation into a reusable building block.

After registration, the transformation can be referred to by name or imported wherever it is needed.

**Using it in a model**

To use the transformation in a model, we introduce an unconstrained latent parameter and attach the
transformation to it. During model construction, ``probly`` will then:

- store the transformation together with the parameter,
- transparently apply the forward mapping whenever the constrained parameter is needed,
- keep track of the relationship so that gradients and uncertainty estimates remain consistent.

This mirrors the way Stan and other packages internally work with unconstrained parameters while
presenting constrained parameters in the modelling language :cite:`stanLowerBoundedScalarND,stanProgrammingLanguage2p8p0`. From the model author’s perspective, the parameter now behaves like a
normal positive quantity, even though internally it is represented by an unconstrained variable.

**Running inference and inspecting results**

When we run inference, optimisation, or sampling, ``probly`` operates in the unconstrained space but
uses the transformation to interpret results in the constrained space. After the run finishes, we
can:

- inspect posterior samples or point estimates of the constrained parameter,
- verify that all inferred values satisfy the desired constraints,
- compare behaviour with and without the custom transformation to understand its impact.

Empirically, users have reported that carefully chosen positive transforms can significantly
improve numerical behaviour. For example, one NumPyro user notes a very substantial improvement in
inference stability when replacing an ``exp`` transformation with ``softplus`` for constraining
``site_scale`` :cite:`vitklSoftplusTransform2020`. This simple workflow generalises to more complex transformations with
multiple inputs, coupled constraints, or additional structure, and similar patterns appear across
modern probabilistic programming frameworks.

2.5 Advanced patterns
~~~~~~~~~~~~~~~~~~~~~

Once you are comfortable with basic custom transformations, ``probly`` allows for more advanced
usage patterns that can make large or complex models easier to express. In the wider literature,
normalizing flows show how powerful models can be obtained by composing simple invertible
transformations :cite:`papamakariosNormalizingFlows2021,rezendeVariationalFlows2015`.

**Composing multiple transformations**

Often it is easier to build a complex mapping by **composing several simple transformations**
rather than writing one large one. For example, you might:

- first apply a shift-and-scale transform,
- then map the result onto a simplex,
- finally enforce an ordering constraint.

Normalizing-flow work explicitly argues that we can build complex transformations by composing
multiple instances of simpler transformations :cite:`papamakariosNormalizingFlows2021`, while still
preserving invertibility and differentiability. Deep-learning libraries such as TensorFlow
Probability provide bijector APIs that implement this idea in practice, allowing chains of
transforms to be treated as a single object :cite:`tfpModuleBijectorsND`.

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
parameters are tied together through common hyperparameters. In that context, hierarchical models
allow for the pooling of information across groups while accounting for group-specific
variations :cite:`mittalBayesianHierarchicalModels2025`. Using shared parameters across transformations in ``probly``
has a similar effect: information is shared in a controlled way, and the structure of the model
remains explicit and interpretable.

**Handling randomness vs determinism inside transformations**

Most transformations are deterministic mappings, but in some cases it is useful to include
controlled **randomness** inside a transformation (for example randomised rounding or stochastic
discretisation). When you design such components, it helps to follow the discipline used by
modern functional ML frameworks.

For example, the JAX documentation emphasises that JAX avoids implicit global random state and
instead tracks state explicitly via a random key, and stresses that you should never use the same
key twice :cite:`jaxPseudorandomNumbers2024`. Even if ``probly`` uses a different backend, the same
principles are useful:

- deterministic behaviour is usually easier for optimisation and debugging,
- if randomness is used, it should be driven by the same seeding and PRNG mechanisms as the rest
  of the model,
- the statistical meaning of the model should remain clear even when transformations are
  stochastic.

In practice, this means treating any random choices inside a transformation as part of the
probabilistic model, not as hidden side effects.

2.6 Testing & debugging
~~~~~~~~~~~~~~~~~~~~~~~

Well-tested transformations are crucial for trustworthy models. Because transformations sit
between the internal representation and the visible parameters, subtle bugs can be hard to
detect unless you test them explicitly. Large probabilistic frameworks such as Stan rely on
extensive unit tests for accuracy of values and derivatives as well as error checking :cite:`carpenterStanProbabilistic2017`, which is a good benchmark for how seriously this layer should
be treated.

**Round-trip tests (forward + inverse)**

A basic but powerful test is the **round-trip check**:

- sample or construct a range of valid unconstrained inputs,
- apply the forward mapping followed by the inverse mapping,
- verify that the original inputs are recovered (up to numerical tolerance).

From a mathematical point of view, this is just checking the fundamental property of a
bijective transform. Walton emphasises that all bijective functions are invertible
and satisfy :math:`f^{-1}(f(x)) = x`, which is exactly what round-trip tests are designed to
catch when your implementation or shape handling is wrong :cite:`waltonIsomorphismNormalizing2023`.

Similarly, you can test constrained values by applying inverse then forward. Systematic
deviations in either direction usually indicate mistakes in the formulas, inconsistencies in
broadcasting, or shape mismatches between forward and inverse.

**Numerical stability checks**

Transformations that operate near boundaries (very small or very large values, probabilities
near 0 or 1, etc.) can suffer from numerical problems. It is good practice to:

- test extreme but valid inputs,
- check for overflow, underflow, or ``nan``/``inf`` values,
- monitor gradients if the transformation is used in gradient-based inference.

Practical experience in differentiable simulation libraries shows why this matters. The
DiffeRT documentation notes that NaNs tend to spread uncontrollably, making it difficult to
trace their origin, and therefore adopts a strict no-NaN policy for both outputs and
gradients :cite:`eertmansNaNInfinite2025`. The same mindset works well in
``probly``: treat any appearance of NaNs or infinities as a bug in either the transformation
or its inputs, and add targeted tests to reproduce and eliminate it.

Where necessary, introduce small epsilons, safe clipping, or alternative parameterisations
to keep the transformation stable. For instance, many implementations replace naïve formulas
by numerically stable variants or custom Jacobians when differentiability and stability
conflict, as discussed in the algorithmic differentiation literature :cite:`griewankWaltherEvaluatingDerivatives2008`.

**Common pitfalls and how to recognise them**

Typical issues with custom transformations include:

- silently producing invalid outputs (for example negative values where only positives are allowed),
- mismatched shapes between forward and inverse mappings,
- forgetting to update the transformation when the model structure changes,
- inconsistent handling of broadcasting or batching.

Basic unit-testing advice for probabilistic code still applies here. As one practitioner
summarises, you should at least assert that returned values are not null and lie in the
expected range, and then add stronger distributional checks where appropriate :cite:`hvgotcodesUnitTesting2012`. For transformations, that means checking *both* the unconstrained and constrained
spaces for sanity (ranges, monotonicity, simple invariants).

Symptoms of problems with transformations often show up later as:

- optimisation failing to converge or getting stuck,
- extremely large or unstable uncertainty estimates,
- runtime errors or NaNs deep inside the inference code.

Empirical studies of probabilistic programming systems show that many real bugs are linked
to boundary conditions, dimension handling, and numerical issues :cite:`duttaTestingProbabilistic2018`. Their
tool ProbFuzz, for example, discovered many previously unknown bugs across several
systems and caught at least one existing bug in most categories they targeted :cite:`duttaTestingProbabilistic2018`. This underlines that small mistakes in transform logic can
have large downstream effects.

When such issues appear in a ``probly`` model, it is often helpful to temporarily isolate
the transformation in a small test script, run the round-trip and stability checks described
above, and only then reintegrate it into the full model. This mirrors the way mature
probabilistic frameworks separate low-level tests of math functions and transforms from
high-level tests of full models :cite:`carpenterStanProbabilistic2017`.

3. Working with Large Models
----------------------------

3.1 What is a “large” model in practice?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What counts as a “large” model depends on your hardware and your goals. In the
research world, “large models” often mean networks with hundreds of millions or
billions of parameters :cite:`tuOverviewLargeAI2024`. In everyday ``probly`` projects, you will
usually run into “large-model” problems much earlier, as soon as memory, data
handling, or runtime start to become annoying.

In practice, a model is “large” when one or more of these become real limits:

- **Model size (number of parameters)**

  As you add layers and parameters, you need memory for parameters, gradients,
  optimiser state, and activations. If this no longer fits comfortably on a
  single device, you are in “large-model” territory :cite:`tyagiScalingDeepLearning2025`.

- **Dataset size**

  A model can also feel large because the **data** are large. If the full
  dataset does not fit in RAM, you have to switch to streaming or mini-batches
  instead of loading everything at once :cite:`tyagiScalingDeepLearning2025`.

- **Runtime and cost**

  Even a medium-sized model becomes “large” if one run takes many hours, or if
  GPU time is expensive and you can only afford a few runs :cite:`tuOverviewLargeAI2024,tyagiScalingDeepLearning2025`.

For this chapter, we call a model “large” whenever memory, data handling, or
runtime force you to think about structure and efficiency, instead of just
writing the most direct version of the model.

3.2 Model structuring strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As models and datasets grow, **code structure** becomes as important as the
choice of algorithm. A messy single file might work for a tiny example but
quickly becomes painful for larger projects. Guides on structuring data science
projects recommend a simple, modular layout instead of one big script :cite:`patiBestPracticesDSProjects2025,zinkevichRulesMLND`.

**Modular design (sub-models and reusable components)**

For ``probly`` projects, a modular design usually means:

- separating data loading and preprocessing from model definition and inference,
- grouping related model parts into their own modules (for example,
  ``uncertainty_heads.py`` or ``transforms/constraints.py``),
- turning common patterns into reusable functions or classes.

Splitting a project into files like ``preprocess.py``, ``train.py``, and ``evaluate.py``
makes it easier to maintain and reuse code :cite:`patiBestPracticesDSProjects2025`. The same idea applies to ``probly``: instead of one huge model
file, you build small building blocks (e.g. shared transformations or likelihood
components) and import them where you need them.

**Naming and project layout**

A clear layout makes a large codebase feel smaller. In practice, this can mean:

- using descriptive filenames such as ``large_models/core_layers.py`` or
  ``pipelines/experiment_large_01.py``,
- keeping reusable library code separate from experiment-specific scripts and
  notebooks,
- writing down a short “project structure” section in the README so new people
  can quickly find the important pieces :cite:`patiBestPracticesDSProjects2025,zinkevichRulesMLND`.

Good structure does not make the model mathematically simpler, but it makes it
much easier to find bugs, add new ideas, and run larger experiments without
getting lost.

3.3 Memory management
~~~~~~~~~~~~~~~~~~~~~

For small toy examples, you can often ignore memory and just run the model. As
soon as you start using bigger datasets or deeper networks, memory becomes a
real constraint. Typical symptoms are “out of memory” errors on the GPU, very
slow training, or code that spends a lot of time just moving data around.

**Batching and mini-batching**

Mini-batching means processing a subset of the data at a time instead of the
whole dataset. This is standard practice in large-scale deep learning: it
reduces memory usage and often makes hardware utilisation better :cite:`tyagiScalingDeepLearning2025`.

For ``probly`` models, this usually means:

- choosing a batch size that fits comfortably in GPU or CPU memory,
- keeping intermediate tensors only for the current batch,
- scaling to larger datasets by running more batches instead of making each
  batch bigger and bigger.

**Streaming data**

When the dataset does not fit into RAM, you need some form of **streaming**:

- a data loader that reads from disk in chunks,
- a generator that yields one batch at a time,
- sharded datasets that are loaded piece by piece.

The details depend on whether you use PyTorch, JAX, or something else, but the
idea is always the same: the model only ever sees a manageable batch, not the
entire dataset at once :cite:`tyagiScalingDeepLearning2025`.

**Avoiding unnecessary copies and recomputations**

Memory and runtime are often wasted by hidden copies and repeated work. Common
issues include:

- moving tensors between CPU and GPU more often than necessary,
- calling ``.cpu()``, ``.numpy()`` or similar conversions in tight loops,
- recomputing the same large intermediate results in every iteration.

A simple rule of thumb is:

- move data to the right device **once per batch**,
- cache expensive things that do not change,
- profile your code to see whether the main cost is in the model, the data
  pipeline, or device transfers :cite:`tyagiScalingDeepLearning2025`.

3.4 Scalability features in ``probly``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Even with good batching and streaming, some models will still push the limits
of your hardware. Modern numerical libraries provide features like
**vectorisation** and **just-in-time (JIT) compilation** to help with this.
``probly`` can benefit from these features when it runs on JAX or similar
backends.

**Vectorisation**

Vectorisation means writing code that works on whole arrays at once instead of
looping in Python. This lets the backend use fast compiled kernels and parallel
hardware :cite:`tuOverviewLargeAI2024,tyagiScalingDeepLearning2025`.

In ``probly``, vectorisation usually looks like:

- writing your model so it naturally accepts batches of inputs,
- evaluating many data points or parameter settings in one call,
- avoiding Python ``for``-loops in the hottest parts of the code when an array
  operation would do.

**JIT compilation and configuration knobs**

JIT compilation takes a Python function and compiles it into an efficient
accelerator program. Frameworks such as JAX use this to turn numerical Python
code into highly optimised kernels :cite:`tuOverviewLargeAI2024`.

When ``probly`` runs on such a backend, you can:

- JIT-compile the main log-likelihood or posterior function,
- reuse compiled functions across many batches or chains,
- switch JIT on or off depending on whether you are debugging or running a
  large experiment :cite:`tyagiScalingDeepLearning2025`.

Typical configuration “knobs” in a ``probly`` project include:

- enabling/disabling JIT for specific functions,
- deciding which dimension to batch over (data vs. chains),
- choosing between a slow, very transparent debug mode and a fast, compiled mode.

3.5 Case study: scaling up a small example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section sketches a typical path from a tiny prototype to a more serious
large-model setup in ``probly``. The exact code will differ, but the steps are
similar in most projects.

**Step 1 – Start small and simple**

You begin with a small dataset and a simple model on your laptop. At this
stage, you:

- run everything on a single device,
- keep all data in memory,
- focus on correctness and clarity, not speed.

Practical ML advice strongly recommends starting this way: get a simple
baseline working end-to-end before you add complexity :cite:`zinkevichRulesMLND`. For a
``probly`` model, this means checking that:

- the model compiles,
- transformations and priors behave sensibly,
- metrics such as loss and accuracy look reasonable.

**Step 2 – Add more data and batching**

Next, you switch to a larger dataset. Now you:

- introduce mini-batches so only part of the data is in memory at a time,
- replace ad-hoc loading with a proper data loader or generator,
- keep the model structure almost the same so you can tell whether problems
  come from the data size or from the model itself :cite:`tyagiScalingDeepLearning2025`.

You watch for memory errors, runtime per step, and whether the metrics still
behave similarly to the small-data case.

**Step 3 – Grow the model and use the hardware**

Once data handling is under control, you might want a bigger or more expressive
model. At this point, you:

- add layers or hierarchical structure where it helps,
- use regularisation to keep things stable,
- start using vectorisation and, where available, JIT compilation to make
  better use of the hardware :cite:`tuOverviewLargeAI2024,tyagiScalingDeepLearning2025`.

Profiling helps you see whether the time is spent in the model, the data
pipeline, or somewhere else.

**Step 4 – Run “production-like” experiments**

Finally, you run something closer to a real large-scale experiment:

- full training and validation sets,
- realistic batch sizes and number of epochs,
- logging, monitoring, and checkpointing turned on.

Guides for real-world ML systems stress the importance of data checks, clear
metrics, and experiment tracking at this stage :cite:`zinkevichRulesMLND,tyagiScalingDeepLearning2025`. For ``probly``, the idea is the same: you want runs that are not only fast,
but also traceable and reproducible.

3.6 Checklist: preparing a large model run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before starting a big and expensive run, it helps to walk through a short
checklist. Many common problems in production ML come from skipped basic steps,
not from exotic algorithms :cite:`zinkevichRulesMLND,tyagiScalingDeepLearning2025`.

**Data and problem**

- Is the prediction task clearly defined (inputs, target, evaluation metric)?
- Has the training data been checked for obvious issues (missing values, wrong
  ranges, label problems)?
- Are training, validation, and test splits clearly separated?
- If you stream data, are you sure the pipeline eventually covers the whole
  dataset?

**Model and code**

- Has the same model been run on a smaller dataset as a sanity check? :cite:`zinkevichRulesMLND`
- Are custom pieces (e.g. transformations) covered by at least basic tests
  (shapes, ranges, round-trip checks)?
- Is configuration (batch size, learning rate, etc.) separated from the code so
  you can easily rerun experiments with different settings?

**Resources and runtime**

- Does the model fit in memory on the planned hardware with the chosen batch
  size? :cite:`tyagiScalingDeepLearning2025`
- Have you done a short “smoke test” run (for example, one epoch or a few
  batches) on the real hardware?
- Is checkpointing enabled so that you can resume after interruptions?

**Monitoring and reproducibility**

- Are key metrics (loss, accuracy, calibration, runtime per step) being logged
  somewhere you can inspect later?
- Are random seeds, library versions, and important hyperparameters recorded
  so that important runs can be reproduced? :cite:`zinkevichRulesMLND`

**Before you press “run”**

Ask yourself:

- If this run fails, do I know what I will try next?
- Is there a cheaper or smaller version of this experiment I could run first?
- Do I have clear success criteria (for example, “validation accuracy improves
  by at least 2 points without worse calibration”)?

Walking through this checklist helps make sure that, when you finally launch a
large ``probly`` run, you use your compute budget wisely and can trust what the
results are telling you :cite:`tuOverviewLargeAI2024,tyagiScalingDeepLearning2025,zinkevichRulesMLND,patiBestPracticesDSProjects2025`.

4. Integration with Other Frameworks
------------------------------------

This chapter assumes that you sometimes want to use ``probly`` together with other
tools: neural-network libraries, data pipelines, or classic ML components.
The goal is not to cover every possible setup, but to give you an idea of how
``probly`` can fit into a larger system and what to watch out for at the
boundaries.

.. note::

   Right now, ``probly`` has first-class helpers and maintained examples for
   **PyTorch** and **Flax/JAX** only.

   References to other libraries in this section (such as TensorFlow, ``tf.data``,
   or scikit-learn) are meant as *conceptual* integration patterns or ideas for
   your own adapters, **not** as built-in, officially supported backends.

4.1 General integration concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you connect ``probly`` with other frameworks, three questions come up over
and over again:

- how **data** moves between components,
- how **types, shapes, and devices** are handled,
- how **randomness and seeds** are managed.

**Data flow between `probly` and other libraries**

``probly`` works with array-like objects: batches of inputs, parameter vectors,
and uncertainty representations. Other libraries do the same, but each has its
own types:

- JAX / Flax use JAX arrays and pytrees,
- TensorFlow uses ``tf.Tensor`` and ``tf.data.Dataset`` :cite:`tensorflowTfDataGuide2024`,
- scikit-learn expects NumPy arrays or “array-like” objects for ``fit`` and
  ``predict`` :cite:`sklearnDevelopingEstimators2024`.

When integrating, the main job is to **convert between these array types in a
controlled place**. In practice this usually means:

- deciding where the **main computation** lives (e.g. in JAX/Flax or PyTorch),
- converting data *once* at a clear boundary (not back and forth inside tight
  loops),
- keeping helper functions like ``to_jax_array(...)`` or ``to_numpy(...)`` in one
  module so you can change them later if needed.

**Types, shapes, and devices (CPU/GPU)**

Array libraries are quite strict about shapes and dtypes. For example,
``tf.data`` datasets produce elements with a fixed structure and shape :cite:`tensorflowTfDataGuide2024`,
and scikit-learn’s estimators assume 2D matrices of shape
``(n_samples, n_features)`` :cite:`sklearnDevelopingEstimators2024`.

To avoid surprises, it helps to:

- settle on a simple **shape convention** (e.g. “batch dimension first”),
- standardise dtypes (usually ``float32`` for model inputs),
- move arrays to the correct **device** (CPU vs GPU) *before* calling library
  functions.

Copying data between CPU and GPU is often more expensive than calling another
Python function, so you want to minimise those device hops.

**Randomness and seeds across frameworks**

Different frameworks treat randomness differently:

- JAX uses **explicit PRNG keys**: you create a key from a seed and then split
  it whenever you need fresh randomness :cite:`jaxPseudorandomNumbers2024`.
- Flax builds on this and treats RNG streams as part of a module’s state and
  lifecycle :cite:`flaxDevelopersLinenFundamentals2023`.
- Many other libraries (TensorFlow, NumPy, PyTorch) use global or graph-local
  RNGs with functions like ``set_seed`` :cite:`tensorflowTfDataGuide2024`.

A simple pattern for combined setups is:

- pick one library (often JAX in a Flax/``probly`` project) as the **main source
  of randomness**,
- derive keys or seeds from there and pass them into other parts of the system,
- log the seeds/keys you used for important runs so they can be reproduced later :cite:`jaxPseudorandomNumbers2024`.

4.2 Using ``probly`` with Flax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Flax is a neural-network library on top of JAX. It provides a **Module**
abstraction that cleanly separates parameters, state, and randomness :cite:`flaxDevelopersLinenFundamentals2023`. This makes it a natural match for ``probly`` when you
want neural nets inside a probabilistic model.

**Typical workflow: Flax for neural nets, ``probly`` for probabilistic parts**

A common setup looks like this:

1. Define a Flax model (for example, an encoder or feature extractor) as a Linen
   module.
2. Initialise the Flax model to get a **variables dict** that holds parameters
   and any extra state (e.g. batch-norm statistics) :cite:`flaxDevelopersLinenFundamentals2023`.
3. Define a ``probly`` model that takes the Flax outputs (features, logits,
   etc.) as inputs to probabilistic components (likelihoods, priors, uncertainty
   heads).
4. Build a training or inference loop that updates both the Flax parameters and
   the ``probly`` parameters together.

Flax’s design highlights the difference between **computation** and
**parameters/state**: modules define the computation, while parameters and state
live in separate data structures :cite:`flaxDevelopersLinenFundamentals2023`. ``probly`` can then
treat those parameters as just another part of the probabilistic model.

**Sharing parameters and state**

In a joint Flax+``probly`` model you typically want:

- all learnable parameters (Flax + ``probly``) inside one combined PyTree,
- deterministic state (e.g. running means) clearly separated from stochastic
  parameters.

A small helper that packs and unpacks these pieces makes it easy for optimisers
to see “one big parameter object” while still keeping a clear structure inside.

**PRNG handling**

Both JAX and Flax use explicit PRNG keys. The JAX docs emphasise that keys are
pure values and that you should never reuse the same key twice :cite:`jaxPseudorandomNumbers2024`. Flax
modules provide helpers like ``make_rng`` to get new keys when they
need randomness :cite:`flaxDevelopersLinenFundamentals2023`.

For a stable integration, treat keys just like any other input:

- thread them through your top-level training/inference functions,
- split them where you need extra randomness,
- store the initial seed in your experiment logs.

4.3 Using ``probly`` with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   ``probly`` does **not** currently ship an official TensorFlow backend.
   The patterns in this subsection describe how you *could* connect things in
   your own code, by analogy with other frameworks.

TensorFlow is often used for input pipelines and training infrastructure. A
typical custom integration would use **TensorFlow for data and orchestration**
and **``probly`` for the probabilistic core**.

**Passing TensorFlow data into ``probly``**

TensorFlow’s ``tf.data`` API represents datasets as streams of elements
(tensors) that you can map, batch, and shuffle :cite:`tensorflowTfDataGuide2024`. In a
TensorFlow+``probly`` workflow you might:

- build a ``tf.data.Dataset`` that yields batches of inputs and targets,
- inside the training loop, turn each batch into NumPy or JAX arrays in the
  format ``probly`` expects,
- call the ``probly`` model on these arrays,
- optionally convert results (e.g. predictions, uncertainties) back to tensors
  if you want to use TensorFlow tools like TensorBoard.

**Training loops and performance**

You can treat ``probly`` as a black-box model called from a TensorFlow training
loop. Performance guides for ``tf.data`` recommend overlapping input loading
with model execution using things like ``prefetch`` and parallel ``map`` :cite:`tensorflowTfDataGuide2024`. The same idea applies here: make sure the data pipeline
keeps the probabilistic model busy instead of letting it wait for I/O.

4.4 Using ``probly`` with scikit-learn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   ``probly`` does not currently include a built-in scikit-learn adapter.
   The code patterns here are suggestions for writing your **own** wrapper that
   follows scikit-learn’s estimator API.

scikit-learn defines a standard estimator interface with methods like ``fit``,
``predict``, and ``score``. The developer guide explains that ``fit`` is where
the training happens and that estimators should follow common rules for inputs
and attributes :cite:`sklearnDevelopingEstimators2024`.

To plug ``probly`` into this ecosystem, you can write a small wrapper class.

**Wrapping a ``probly`` model as an estimator**

A minimal wrapper might:

- take configuration options (model structure, priors, inference method) in
  ``__init__``,
- implement ``fit(X, y=None)`` to run ``probly``’s training or inference on the
  given data,
- implement ``predict(X)`` or ``predict_proba(X)`` to return point predictions
  or uncertainty summaries,
- optionally implement ``score(X, y)`` using scikit-learn metrics or your own
  custom metric.

If your wrapper follows the standard estimator rules, it can be used with
scikit-learn tools like cross-validation and grid search :cite:`sklearnDevelopingEstimators2024`.

**Pipelines and cross-validation**

Once wrapped, a ``probly`` estimator can be placed inside a scikit-learn
``Pipeline`` together with preprocessing steps, and then evaluated with
cross-validation. The advantage is that preprocessing and modelling are tuned
and evaluated together, using familiar tools from the scikit-learn ecosystem :cite:`sklearnDevelopingEstimators2024`.

4.5 Interoperability best practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A few habits make life much easier when ``probly`` and other frameworks meet:

**Device management**

- Decide early which parts run on CPU and which on GPU.
- Move a batch to the target device **once**, do all the work there, then move
  back only summary results if needed.
- Avoid hidden device transfers in utility functions.

**Version management**

- Pin versions of key libraries (JAX, Flax, PyTorch, TensorFlow, scikit-learn)
  so everyone runs the same stack.
- Note any known compatibility requirements (for example, which JAX version a
  given Flax release expects) :cite:`flaxDevelopersLinenFundamentals2023`.
- Record the versions used for important experiments.

**Debugging across library boundaries**

Cross-library bugs usually come from mismatched assumptions:

- shape or dtype mismatches,
- data accidentally on the wrong device,
- inconsistent random-number handling.

When debugging:

- start with a tiny example that only tests the hand-off between libraries,
- print/log shapes, dtypes, and devices right before and after conversion
  points,
- disable advanced features like JIT or complex pipelines until the basics
  work,
- re-run with fixed seeds so you can tell whether errors are deterministic :cite:`jaxPseudorandomNumbers2024,tensorflowTfDataGuide2024`.

If you treat integration points as “first-class citizens” and give them a bit
of structure and testing, you can combine ``probly`` with other frameworks
without turning the whole project into a black box.

5. Performance & Computational Efficiency
-----------------------------------------

5.1 Understanding performance bottlenecks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a model feels “slow”, the first step is to understand **where the time is
actually spent**. In typical ``probly`` workflows, bottlenecks usually fall into
a few simple categories:

- **CPU compute** – lots of Python loops, non-vectorised NumPy operations, or
  heavy bookkeeping in pure Python.
- **GPU compute** – large matrix multiplications or convolutions that fully
  load the GPU.
- **I/O** – reading data from disk or the network, or slow preprocessing.
- **Python overhead** – very frequent function calls, dynamic graph building,
  or extremely verbose logging.

Profiling tools help you see which of these dominates. The standard Python
profilers, for example, record how often and for how long various parts of the
program executed :cite:`pythonProfilersND`, so you can check whether
time goes into your model, the data pipeline, or external libraries.

A simple routine that works well in practice:

- run a **small experiment** with realistic settings,
- profile the run to find the **slowest functions/lines**,
- focus optimisation on the few places that clearly dominate runtime.

You do not need perfect measurements – just enough to see where the main time
sink is.

5.2 Profiling your ``probly`` code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Profiling your code can stay very simple. In many cases, it is enough to:

- use a **function-level profiler** (like ``cProfile``) to find the most
  expensive calls :cite:`pythonProfilersND`,
- add a **line-level or memory profiler** only when you suspect a specific
  block of code.

A practical workflow:

1. Wrap your main training or inference loop in a profiler context.
2. Run a short experiment on a subset of the data.
3. Sort the output by **cumulative time** to see the top few functions.
4. For one or two of those, use a line profiler or add logging to see what is
   really happening.

The goal is not to optimise every line. You just want to answer questions like:

- Is the time mostly in ``probly`` / NumPy / JAX, or in my own Python glue
  code?
- Is data loading slower than the model itself?
- Are there one or two functions that dominate runtime?

Once you know that, it is much easier to decide what to change.

5.3 Algorithmic improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before you tweak low-level details, it often helps more to change the
**algorithmic setup**:

- **Pick an inference method that fits the model.**
  Some models work fine with simple optimisation; others need richer samplers.
  A method with better convergence can cut total runtime a lot, even if each
  step is a bit slower.

- **Simplify or re-parameterise the model.**
  Better parameterisations can improve gradient flow, avoid extreme curvature,
  and make constraints easier to handle. That usually means fewer iterations
  and more stable training.

- **Re-use previous runs.**
  Warm-start from parameters that already work reasonably well, or cache
  expensive intermediate results. There is no need to recompute everything from
  scratch if a similar experiment has already been done.

Many “performance problems” disappear once the model and inference method are a
good match for the task.

5.4 Vectorisation & parallelisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Low-level speed usually comes from **doing more work per call**, not from
writing more loops. Array libraries like NumPy are designed so that you express
operations on whole arrays and they run in fast compiled code instead of pure
Python :cite:`harrisArrayProgramming2020`.

In ``probly``, this means:

- prefer **batch operations** over manual Python ``for``-loops,
- write code so that entire arrays of parameters, samples, or observations can
  be processed at once,
- let the backend (NumPy, JAX, etc.) use SIMD, multi-core CPUs, or GPUs.

You can combine this with **parallelisation**:

- run independent chains or tasks on different CPU cores or devices,
- make sure the work per task is large enough so that parallel overhead does
  not dominate,
- keep seeds and random-number streams clearly separated, so parallel chains
  really are independent :cite:`opendatascienceRandomSeed2019`.

More parallelism is not always better: if each task is tiny, the overhead of
starting and syncing workers can outweigh any speedup.

5.5 Reproducibility & randomness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Randomness is central to probabilistic modelling but can make performance
harder to debug if every run behaves differently. A few simple habits help:

- **Set random seeds on purpose.**
  Use fixed seeds for NumPy, JAX, and other backends so that runs with the same
  settings produce comparable results :cite:`opendatascienceRandomSeed2019`.

- **Log important settings.**
  Store seeds, dataset versions, batch sizes, hardware info, and key
  hyperparameters somewhere (config files, experiment tracker, or logs).

- **Balance reproducibility and exploration.**
  During debugging and profiling, fixed seeds are very helpful. For final
  experiments, you might run several seeds to see how stable the results are.

Good reproducibility is not just “nice for papers”; it makes performance tuning
much easier, because you know that changes in runtime or metrics are due to
your code changes, not random noise.

5.6 Performance checklist
~~~~~~~~~~~~~~~~~~~~~~~~~

Before you launch a big and expensive run, a quick checklist can save a lot of
time:

- **Model & algorithm**
  - Does the inference method make sense for this model?
  - Are there layers, parameters, or transforms you can remove without losing
    quality?

- **Implementation**
  - Are your main computations vectorised, or are there slow Python loops in
    the hot path?
  - Are you avoiding repeated work (e.g. recomputing static features inside the
    main loop)?

- **Data pipeline**
  - Is data loading fast enough compared to the model compute?
  - Are you using batching or mini-batching to keep memory usage under control?

- **Resources**
  - Is the model using available hardware (CPU cores, GPU, memory) in a
    sensible way?
  - Is logging set to a reasonable level so it does not become an I/O
    bottleneck?

- **Reproducibility**
  - Are seeds and key settings stored somewhere?
  - Can you reproduce a small profiling run before scaling up?

If you can honestly tick these boxes, you are much less likely to waste compute
and far more likely to understand what your large ``probly`` runs are doing.

6. Advanced Usage Patterns & Recipes
------------------------------------

6.1 Common advanced modeling patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section sketches a few “advanced” modelling patterns you will often see in
real projects. The goal is not to give full mathematical detail, but to show how
they fit conceptually with ``probly`` and when they are useful.

**Hierarchical models**

Hierarchical (or multilevel) models are used when data are organised in groups,
levels, or contexts – for example, students within classes, patients within
hospitals, or measurements for multiple machines. Instead of fitting a separate
model to each group, a hierarchical model shares information across groups using
higher-level parameters. This “partial pooling” stabilises estimates, especially
when some groups have only a few observations :cite:`gelmanHillDataAnalysis2007`.

In ``probly``, hierarchical models typically:

- define group-specific parameters (e.g. intercepts or slopes),
- tie them together through shared hyperparameters,
- use uncertainty representations to see how much information is borrowed across groups.

This pattern is especially helpful when you care about both overall trends and
group-level differences at the same time.

**Mixture models**

Mixture models assume that the data come from a combination of several latent
components, such as different customer types, regimes, or clusters. A classic
example is a Gaussian mixture model, where each data point is generated from one
of several Gaussian components, each with its own mean and variance :cite:`bishopPatternRecognition2006`.

In ``probly``, mixture models can:

- represent component-specific parameters and their mixing weights,
- use latent variables (discrete or continuous) to indicate which component
  generated each observation,
- quantify uncertainty about both the component assignments and the component
  parameters.

You would reach for a mixture model when a single simple distribution cannot
capture the shape of your data (for example, clearly multi-modal data).

**Time-series and sequential models**

Time-series and sequential models deal with data that arrive in order, such as
sensor readings, financial prices, or user activity over time. Typical goals are
to forecast future values, detect regime changes, or understand temporal
structure :cite:`hyndmanForecastingPrinciples2018`.

With ``probly``, you can:

- build models that include lagged variables, latent states, or time-varying
  parameters,
- express uncertainty about future trajectories, not just single point forecasts,
- feed these predictive distributions into downstream decisions or risk analysis.

More advanced time-series models often mix ideas from hierarchies (e.g. many
related series, like many stores over time) and mixtures (e.g. different
behavioural regimes).

6.2 Reusable templates
~~~~~~~~~~~~~~~~~~~~~~

As your models become more complex, it helps to recognise **reusable templates**:
small patterns that show up again and again. Examples include:

- a standard hierarchical regression block for grouped data (inspired by
  typical multilevel models in :cite:`gelmanHillDataAnalysis2007`),
- a generic mixture-of-experts block that combines several prediction heads :cite:`bishopPatternRecognition2006`,
- a time-series forecasting head that can be attached to different feature
  extractors :cite:`hyndmanForecastingPrinciples2018`.

In ``probly``, you can implement these templates as functions or modules that:

- take model-specific pieces as arguments (e.g. feature networks, priors, or
  likelihood choices),
- expose a clear, well-documented interface,
- return predictions and uncertainty representations in a consistent format.

By reusing such templates, you:

- reduce copy–paste boilerplate,
- keep projects more uniform,
- make it easier for other people (or future you) to understand and extend your
  models.

6.3 Pointers to examples
~~~~~~~~~~~~~~~~~~~~~~~~

To make these patterns easier to learn, it is useful to connect each idea to at
least one **worked example**:

- For hierarchical models, a grouped-data example (e.g. “schools”, “hospitals”,
  or “stores”) that walks through model specification, inference, and how to
  read the group-level posteriors :cite:`gelmanHillDataAnalysis2007`.
- For mixture models, a clustering or anomaly-detection example that shows both
  cluster responsibilities and uncertainty about the clusters themselves :cite:`bishopPatternRecognition2006`.
- For time-series models, a forecasting example that compares point forecasts to
  predictive intervals over time, and shows how to evaluate them :cite:`hyndmanForecastingPrinciples2018`.

For each advanced pattern in this chapter, there is at least one worked example in the
:doc:`Examples & Tutorials <examples_tutorials>` file.

7. Summary
----------

7.1 Key takeaways
~~~~~~~~~~~~~~~~~

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

If you keep these principles in mind, the rest of the ``probly`` documentation methods,
modules, and examples should slot naturally into your own advanced models and experiments.
