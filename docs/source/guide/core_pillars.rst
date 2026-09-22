.. _core_pillars:

======================
Core Pillars of Probly
======================

The previous part described uncertainty without calling a single function. This
part revisits its four questions and shows where the library answers each of
them. The mapping is one-to-one: every question has its own pillar, and every
pillar is one stage of a pipeline.

.. list-table::
    :header-rows: 1
    :widths: 30 30 40

    * - Question
      - Pillar
      - Entry point
    * - How does a model *carry* uncertainty at all?
      - :ref:`Transformation <pillar-transformation>`
      - ``probly.transformation`` / ``probly.method``
    * - What shape does a prediction have?
      - :ref:`Representation <pillar-representation>`
      - ``probly.representer``, ``probly.representation``
    * - How much uncertainty is that, and of which kind?
      - :ref:`Quantification <pillar-quantification>`
      - ``probly.quantification``
    * - Was any of this useful?
      - :ref:`Evaluation <pillar-evaluation>`
      - ``probly.evaluation``, ``probly.metrics``

.. image:: ../_static/readme/from_paper/paper_workflow_light.png
    :class: only-light
    :alt: The four-stage probly workflow: transform a model to make it
        uncertainty-aware, represent its predictive uncertainty, quantify and
        decompose it into total, epistemic and aleatoric parts, and evaluate
        the result in a downstream task.
    :width: 100%

.. image:: ../_static/readme/from_paper/paper_workflow_dark.png
    :class: only-dark
    :alt: The four-stage probly workflow: transform a model to make it
        uncertainty-aware, represent its predictive uncertainty, quantify and
        decompose it into total, epistemic and aleatoric parts, and evaluate
        the result in a downstream task.
    :width: 100%

The separation is deliberate. Any uncertainty library has to make these four
decisions, and many fuse them into a single object. Switching the method then
silently switches the measure as well, and a number computed under one method
can no longer be compared with a number computed under another. In ``probly``,
each stage is a separate import that can be swapped on its own, and two stages
communicate only through a :ref:`representation <uq-representing>`, never
through a framework-specific object.

.. _pillars-one-pipeline:

One Pipeline, Four Stages
=========================

The rest of this page elaborates on the following snippet.

.. code-block:: python

    from probly.method import dropout
    from probly.representer import representer
    from probly.quantification import quantify
    from probly.evaluation.ood import evaluate_ood

    net = ...  # any trained torch or flax network

    # 1. transform: keep dropout active at inference (MC dropout)
    model = dropout(net, p=0.25, predictor_type="logit_classifier")

    # 2. represent: turn stochastic forward passes into a second-order representation
    rep = representer(model, num_samples=50)
    out_id = rep.represent(data_id)
    out_ood = rep.represent(data_ood)

    # 3. quantify: reduce the representation to total/aleatoric/epistemic scalars
    eu_id = quantify(out_id).epistemic
    eu_ood = quantify(out_ood).epistemic

    # 4. evaluate: does the epistemic part separate in- from out-of-distribution?
    print(evaluate_ood(eu_id, eu_ood))

Only stage 1 is specific to the method. Replacing ``dropout`` with ``sngp``
changes that one line; replacing it with ``ensemble`` additionally drops
``num_samples``, since the members of an ensemble are iterated rather than
sampled. Stages 3 and 4 remain untouched in either case, because every
transformation leads to a representation that stage 3 already knows how to
handle.

.. _pillar-transformation:

Pillar 1: Transformation
========================

A transformation takes a model that predicts a point and returns a model that
predicts something richer. It is the only stage that touches the network.

Transformations come in two flavours, and the distinction determines how much
adopting one costs.

*Post-hoc* transformations wrap an already-trained model: ``dropout``
re-enables existing dropout layers, ``mahalanobis`` fits a class-conditional
Gaussian on the trained features, and ``temperature_scaling`` rescales the
logits on a held-out split. Nothing is retrained.

*Ante-hoc* transformations change the architecture or the loss, so the model
has to be trained afterwards: ``ensemble`` returns *N* models to fit,
``bayesian`` replaces deterministic layers with mean-field ones, and
``posterior_network`` attaches a normalizing-flow head.

Whether a transformation is post-hoc or ante-hoc is independent of the
namespace it lives in. The following example crosses the two axes:

.. code-block:: python

    from probly.transformation import dropout   # a primitive, applied per layer
    from probly.method import sngp              # a named method from the literature

    # post-hoc: re-enables the dropout layers the network already has
    mc = dropout(net, p=0.25, predictor_type="logit_classifier")

    # ante-hoc: swaps the last Linear for a random-feature GP head and
    # spectral-normalizes the rest, so the returned model has to be trained
    gp = sngp(net)

.. note::

    The two namespaces overlap, but they are not the same catalogue.
    ``probly.transformation`` holds the composable primitives, including the
    layer, head, and activation transforms that methods are built from:
    ``posterior_network``, ``normal_inverse_gamma_head``,
    ``dirichlet_exp_activation``, ``interval_classifier``.
    ``probly.method`` holds the named methods from the literature: ``sngp``,
    ``swag``, ``ddu``, ``duq``, ``mahalanobis``, and the ``credal_*`` and
    ``evidential_*`` families. The shared building blocks (``dropout``,
    ``ensemble``, ``bayesian``, ``batchensemble``, ``subensemble``,
    ``dropconnect``, ``masksembles``, ``cast``, and the ``conformal_*``
    scores) can be imported from either. Calibration is a subpackage rather
    than a top-level name: ``probly.transformation.calibration.temperature_scaling``.

Two design choices make transformations work across backends. First, a
transformation walks the layer tree with ``pytraverse`` and dispatches *per
layer type*, so it never needs to know what the model as a whole is. Second,
backends are registered lazily, so importing ``probly`` imports neither
``torch`` nor ``flax`` nor ``sklearn``. :ref:`methods` catalogues the methods
that ship today and the backends each of them supports.

.. _pillar-representation:

Pillar 2: Representation
========================

A representation is the object that crosses the boundary between stages. It is
what :ref:`uq-representing` describes in the abstract, made into a type:
``CategoricalDistribution``, ``Sample``, ``ConvexCredalSet``,
``ProbabilityIntervalsCredalSet``, and so on, each with an array, torch, and
JAX implementation.

A representation carries its own semantics: which axis is the sample axis,
whether it lives on the simplex, whether it is a set or a distribution. Stage 3
dispatches on exactly this information, and it neither knows nor cares that
stage 1 was MC dropout. This is the point at which the separation of stages
pays off.

The *representer* is the adapter that builds a representation from a
transformed model:

.. code-block:: python

    from probly.representer import representer

    rep = representer(mc, num_samples=50)
    out = rep.represent(x)   # or rep(x), or rep.predict(x)

For a sampling method, the representer runs the forward pass 50 times and
stacks the results into a ``Sample`` of categorical distributions. For an
ensemble, it runs each member once. For a method that already emits a
Dirichlet, it is the identity. The caller does not need to know which of these
happened.

The transformation does not fix the representation, however. ``representer``
dispatches on the predictor type and returns the representer *registered* for
it, and naming a representer class instead overrides that choice for the same
model:

.. code-block:: python

    from probly.representation.credal_set import create_convex_credal_set
    from probly.representer import ConvexCredalSetRepresenter, representer
    from probly.transformation import ensemble

    ens = ensemble(net, num_members=10, predictor_type="logit_classifier")

    rep = representer(ens)                                 # registered default -> a Sample
    cset = ConvexCredalSetRepresenter(ens).represent(x)    # same model -> a convex credal set

    # MC dropout is stochastic rather than iterable, so the credal reading
    # is built from the sample it produces
    cset_mc = create_convex_credal_set(representer(mc, num_samples=50).represent(x))

The credal representers read an *iterable* predictor. An ensemble can therefore
be handed to them directly, whereas a stochastic model first has to produce a
sample. ``ProbabilityIntervalsRepresenter`` makes the same move for probability
intervals. Beyond the choice of class, the arguments of the factory give a
second handle: whatever is passed to ``representer`` besides the model, such
as ``num_samples``, ``sample_axis``, or ``sample_factory``, is forwarded to the
constructor of whichever representer wins the dispatch.

Building several representations from one model is not a hypothetical need.
The benchmark does exactly this, because interval coverage and convex-hull
coverage each dispatch on a different reading of the same predictions.

.. seealso::

    :ref:`uq-representing` for the order ladder that the types mirror, and
    ``probly.decider`` for reducing a representation to a decision, for
    example collapsing a second-order distribution to a first-order one with
    ``categorical_from_mean``, or a credal set to its maximin-optimal member.

.. _pillar-quantification:

Pillar 3: Quantification
========================

Quantification maps a representation to a number. It offers two levels of entry
point, which differ in who decides what that number means:

.. code-block:: python

    from probly.quantification import decompose, entropy, measure

    h = entropy(out)        # a named measure, applied directly
    m = measure(out)        # the canonical notion for this representation
    uq = decompose(out)     # -> .total, .aleatoric, .epistemic

The named measures (``entropy``, ``mutual_information``, ``vacuity``,
``sample_variance``, ``spectral_entropy``, and others) are ordinary functions:
the caller decides, and the function computes exactly that number. ``measure``
and ``decompose`` form the dispatching layer, in which the representation
decides: they select the decomposition registered for *this* representation
and return, respectively, its canonical notion (usually the total) and the
full split. ``quantify``, which the pipeline above uses, sits on top of this
layer. In most cases it returns the same decomposition, but a representation
may register an entirely different notion of uncertainty in its place.

Which split is admissible depends on the representation. The entropy
decomposition of a second-order distribution is not the same object as the
upper and lower entropy of a credal set, and neither is defined for a bare
first-order distribution, which has nothing to split. The library encodes this
in the dispatch rather than in a docstring warning: requesting a decomposition
that does not exist for the given representation raises ``NotImplementedError``
instead of returning a quietly meaningless number.

For a second-order sample, the decomposition that ``decompose`` selects is
``SecondOrderEntropyDecomposition``. A different one is chosen by constructing
it directly:

.. code-block:: python

    from probly.quantification import (
        BrierLoss,
        EpistemicUncertainty,
        SecondOrderScoringRuleDecomposition,
    )

    uq = SecondOrderScoringRuleDecomposition(out, BrierLoss())
    uq.components              # which notions this split actually provides
    uq[EpistemicUncertainty]   # or uq["eu"], or uq.epistemic

This makes the rule of thumb from :ref:`uq-quantifying`, to pick the scoring
rule first, executable rather than advisory. ``LogLoss`` reproduces the
Shannon-entropy split exactly (its epistemic part *is* the mutual information),
and ``BrierLoss`` yields the Gini split. The credal counterpart is
``CredalSetEntropyDecomposition``, whose total is the upper entropy, whose
aleatoric part is the lower entropy, and whose epistemic part is the gap
between them.

A decomposition is a mapping keyed by the notion types ``TotalUncertainty``,
``AleatoricUncertainty``, and ``EpistemicUncertainty``. Because these notions
are first-class, stage 4 can ask for "the epistemic part" without knowing which
decomposition produced it, which is what allows the active-learning strategies
to be written once. A notion that the decomposition does not define raises
``KeyError`` rather than returning an answer, and ``uq.components`` states in
advance which notions are available.

:ref:`uq-quantifying` covers the measures themselves and the cases in which a
decomposition is undefined.

.. _pillar-evaluation:

Pillar 4: Evaluation
====================

A quantifier always returns a number, including when that number is
meaningless. The last pillar exists to catch this. ``probly.evaluation`` ships
the three downstream tasks by which uncertainty is usually justified:

:Out-of-distribution detection: ``evaluate_ood`` asks whether the uncertainty
    score separates in-distribution from out-of-distribution inputs, and
    reports the answer as AUROC.
:Selective prediction: ``selective_prediction`` abstains on the most uncertain
    fraction of inputs and measures the accuracy that remains. A useful
    uncertainty makes the risk-coverage curve fall.
:Active learning: uncertainty chooses the next labels, and the resulting
    learning curve is compared against random acquisition.

.. code-block:: python

    from probly.evaluation.ood import evaluate_ood

    print(evaluate_ood(eu_id, eu_ood))   # {'auroc': 0.94}

Both functions return more than the headline number on request:

.. code-block:: python

    from probly.evaluation.selective_prediction import selective_prediction

    # the operating point, not just the ranking
    evaluate_ood(eu_id, eu_ood, metrics=["auroc", "aupr", "fpr@0.95"])
    # -> {'auroc': 0.998, 'aupr': 0.998, 'fpr@0.95': 0.006}

    # the risk-coverage curve, not just its area
    aurc, risk_curve = selective_prediction(eu_id, losses, n_bins=50)

``evaluate_ood`` accepts ``metrics="all"`` or a list, and it understands
dynamic specifications such as ``"fpr@0.95"`` and ``"fnr@90%"``, so that a
claim can be stated at the operating point one would actually deploy rather
than as a single AUROC. ``selective_prediction`` returns the area together
with the per-bin losses; it is this curve that :ref:`uq-evaluating` calls the
honest artifact, since the scalar alone hides where the gain occurs.

``probly.metrics`` holds the intrinsic scores (calibration error, coverage, set
size), which ask a different question: not whether the derived score is
*useful*, but whether the predicted distribution is *right*. The proper
scoring rules, in turn, live with the quantifiers in
``probly.quantification.scoring_rule``, so that a rule chosen for evaluation is
the same object as the one that generated the decomposition.
:ref:`uq-evaluating` draws this distinction in detail.

.. seealso::

    Active learning needs a training loop rather than a snippet:
    :ref:`sphx_glr_auto_examples_active_learning_plot_active_learning_torch.py`
    compares uncertainty-driven acquisition against the random baseline.

    :ref:`pillars-composition` covers the dispatch mechanisms that keep the four
    stages independent of each other, and what that means for a new method.
