.. _core_pillars:

======================
Core Pillars of Probly
======================

The previous part described uncertainty without mentioning a single function.
This part does the opposite: it takes the same four questions and shows where
each of them lives in the library.

The mapping is deliberately one-to-one.

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

The pillars are separate on purpose. Every uncertainty library has to make the
same four decisions; most of them fuse the decisions into a single object, so
that switching the method also silently switches the measure, and a number
computed under one method cannot be compared with a number computed under
another. In ``probly`` each stage is one import, each stage is swappable
independently, and the interface between two stages is a
:ref:`representation <uq-representing>`, never a framework-specific object.

.. include:: /_includes/two_moons_setup.rst

.. _pillars-one-pipeline:

One Pipeline, Four Stages
=========================

Everything below is an elaboration of this snippet.

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

Every line but the first is fixed. Replacing ``dropout`` with ``ensemble``,
``laplace``, or ``sngp`` changes stage 1 only, because all of them hand back the
same kind of representation to stage 2. That is the whole design in one
sentence.

.. _pillar-transformation:

Pillar 1: Transformation
========================

A transformation takes a model that predicts a point and returns a model that
predicts something richer. It is the only stage that touches your network.

Transformations come in two flavours, and the distinction decides how much work
adopting one costs you:

*Post-hoc* transformations wrap an already-trained model. ``dropout`` re-enables
existing dropout layers, ``mahalanobis`` fits a class-conditional Gaussian on the
trained features, ``temperature_scaling`` rescales the logits on a held-out
split. Nothing is retrained.

*Ante-hoc* transformations change the architecture or the loss, so training has
to happen afterwards. ``ensemble`` gives you *N* models to fit, ``bayesian``
replaces deterministic layers with mean-field ones, ``posterior_network``
attaches a normalizing-flow head.

The two flavours are one axis; which namespace a name lives in is another. One
example of each, crossed:

.. jupyter-execute::

    from probly.transformation import dropout   # a primitive, applied per layer
    from probly.method import sngp              # a named method from the literature

    # post-hoc: re-enables the dropout layers the network already has
    mc = dropout(net, p=0.25, predictor_type="logit_classifier")

    # ante-hoc: swaps the last Linear for a random-feature GP head and
    # spectral-normalizes the rest, so the returned model has to be trained
    gp = sngp(net)

Two things make this work across backends. The transformation walks the layer
tree with ``pytraverse`` and dispatches *per layer type*, so it never needs to
know what the model as a whole is; and the backends are registered lazily, so
importing ``probly`` does not import ``torch``, ``flax``, or ``sklearn``.
:ref:`methods` catalogues what ships today and which backends each one supports.

.. note::

    The two namespaces overlap but are not the same catalogue.
    ``probly.transformation`` holds the composable primitives, including the
    layer, head, and activation transforms that methods are built out of ---
    ``posterior_network``, ``normal_inverse_gamma_head``,
    ``dirichlet_exp_activation``, ``interval_classifier``.
    ``probly.method`` holds the named methods from the literature ---
    ``sngp``, ``swag``, ``ddu``, ``duq``, ``mahalanobis``, the ``credal_*`` and
    ``evidential_*`` families. The shared building blocks --- ``dropout``,
    ``ensemble``, ``bayesian``, ``batchensemble``, ``subensemble``,
    ``dropconnect``, ``masksembles``, ``cast``, and the ``conformal_*`` scores
    --- are importable from either. Calibration is a subpackage rather than a
    top-level name: ``probly.transformation.calibration.temperature_scaling``.

.. _pillar-representation:

Pillar 2: Representation
========================

A representation is the object that crosses the boundary between stages. It is
what :ref:`uq-representing` describes in the abstract, made into a type:
``CategoricalDistribution``, ``Sample``, ``ConvexCredalSet``,
``ProbabilityIntervalsCredalSet``, and so on, each with an array, torch, and
JAX implementation.

The *representer* is the adapter that builds one:

.. jupyter-execute::

    from probly.representer import representer

    rep = representer(mc, num_samples=50)
    out = rep.represent(x)   # or rep(x), or rep.predict(x)

For a sampling method the representer runs the forward pass 50 times and stacks
the results into a ``Sample`` of categorical distributions. For an ensemble it
runs each member once. For a method that already emits a Dirichlet, the
representer is the identity. The caller does not need to know which of those
happened.

The transformation does not fix the representation, though. ``representer``
dispatches on the predictor type and returns the representer *registered* for
it; naming a representer class instead overrides that choice on the very same
model:

.. jupyter-execute::

    from probly.representation.credal_set import create_convex_credal_set
    from probly.representer import ConvexCredalSetRepresenter, representer
    from probly.transformation import ensemble

    ens = ensemble(net, num_members=10, predictor_type="logit_classifier")

    rep = representer(ens)                                 # registered default -> a Sample
    cset = ConvexCredalSetRepresenter(ens).represent(x)    # same model -> a convex credal set

    # MC dropout is stochastic rather than iterable, so the credal reading
    # is built from the sample it produces
    cset_mc = create_convex_credal_set(representer(mc, num_samples=50).represent(x))

The credal representers read an *iterable* predictor, so an ensemble-shaped
model can be handed to them directly while a stochastic one goes through the
sample it produces first. ``ProbabilityIntervalsRepresenter`` is the same move
for probability intervals. The second lever is the factory's own arguments:
whatever you pass to ``representer`` beyond the model --- ``num_samples``,
``sample_axis``, ``sample_factory`` --- is forwarded to the constructor of
whichever representer wins the dispatch.

This is not a hypothetical. The benchmark builds three representations from one
model for exactly this reason, because interval coverage and convex-hull
coverage each need the reading their own metric dispatches on.

This is where the library earns the separation. A representation knows its own
semantics --- which axis is the sample axis, whether it lives on the simplex,
whether it is a set or a distribution --- and stage 3 dispatches on exactly
that. It does not know or care that stage 1 was MC dropout.

.. seealso::

    :ref:`uq-representing` for the order ladder that the types mirror, and
    ``probly.decider`` for reducing a representation to a decision --- for
    example collapsing a second-order distribution to a first-order one with
    ``categorical_from_mean``, or to the maximin-optimal one for a credal set.

.. _pillar-quantification:

Pillar 3: Quantification
========================

Quantification maps a representation to a number. There are two levels of
entry point, and the difference between them matters:

.. jupyter-execute::

    from probly.decider import categorical_from_mean
    from probly.quantification import decompose, entropy, measure

    h = entropy(categorical_from_mean(out))  # a named measure, applied directly
    m = measure(out)                         # the canonical notion for this representation
    uq = decompose(out)                      # -> .total, .aleatoric, .epistemic

The named measures --- ``entropy``, ``mutual_information``, ``vacuity``,
``sample_variance``, ``spectral_entropy``, and the rest --- are ordinary
functions you call when you know exactly which number you want. ``measure`` and
``decompose`` are the dispatching layer: they pick the decomposition registered
for *this* representation and return, respectively, its canonical notion
(usually the total) and the full split.

*Which* split is admissible depends on the representation. The entropy
decomposition of a second-order distribution is not the same object as the
upper/lower entropy of a credal set, and neither is defined for a bare
first-order distribution --- there is nothing there to split. The library
encodes that in the dispatch rather than in a docstring warning: asking for a
decomposition that does not exist for your representation raises
``NotImplementedError`` instead of returning a quietly meaningless number.

``decompose`` picks the one decomposition registered for the representation ---
for a second-order sample that is ``SecondOrderEntropyDecomposition``.
Constructing a decomposition directly is how you choose a different one:

.. jupyter-execute::

    from probly.quantification import (
        BrierLoss,
        EpistemicUncertainty,
        SecondOrderScoringRuleDecomposition,
    )

    uq = SecondOrderScoringRuleDecomposition(out, BrierLoss())
    uq.components              # which notions this split actually provides
    uq[EpistemicUncertainty]   # or uq["eu"], or uq.epistemic

This is what makes the "pick the scoring rule first" rule of thumb from
:ref:`uq-quantifying` executable rather than advisory: ``LogLoss`` reproduces
the Shannon-entropy split exactly --- its epistemic part *is* the mutual
information --- and ``BrierLoss`` gives the Gini one. The credal counterpart is
``CredalSetEntropyDecomposition``, whose total is the upper entropy, aleatoric
the lower, and epistemic the gap between them.

The notions ``TotalUncertainty``, ``AleatoricUncertainty``, and
``EpistemicUncertainty`` are first-class, so a downstream stage can ask for
"the epistemic part" without knowing which decomposition produced it. That is
what lets the active-learning strategies in stage 4 be written once. A
decomposition is a mapping keyed by those notion types, so a stage can also ask
``uq.components`` what a given split provides instead of assuming --- a notion
the decomposition does not define raises ``KeyError`` rather than answering.

:ref:`uq-quantifying` covers the measures themselves and where decomposition is
undefined.

.. _pillar-evaluation:

Pillar 4: Evaluation
====================

A quantifier always returns a number, including when the number is meaningless.
The last pillar is what stops that from going unnoticed. ``probly.evaluation``
ships the three downstream tasks that uncertainty is usually justified by:

:Out-of-distribution detection: ``evaluate_ood`` --- does the uncertainty score
    separate in-distribution from out-of-distribution inputs? Reported as AUROC.
:Selective prediction: abstain on the most uncertain fraction and measure what
    accuracy remains. A useful uncertainty makes the risk-coverage curve fall.
:Active learning: use uncertainty to choose the next labels, and compare the
    resulting learning curve against random acquisition.

.. jupyter-execute::
    :hide-code:

    # The epistemic scores the evaluation blocks consume, from the MC dropout
    # model built above: one set per split, plus the 0/1 losses.
    with torch.no_grad():
        eu_id = decompose(representer(mc, num_samples=50).represent(data_id)).epistemic
        eu_ood = decompose(representer(mc, num_samples=50).represent(data_ood)).epistemic
        # a torch criterion pairs with torch losses: selective_prediction
        # dispatches on the backend and does not mix the two.
        losses = (net(data_id).argmax(-1) != labels).float()

.. jupyter-execute::

    from probly.evaluation.ood import evaluate_ood

    print(evaluate_ood(eu_id, eu_ood))

Both entry points return more than the headline number if you ask them to:

.. jupyter-execute::

    from probly.evaluation.selective_prediction import selective_prediction

    # the operating point, not just the ranking
    print(evaluate_ood(eu_id, eu_ood, metrics=["auroc", "aupr", "fpr@0.95"]))

    # the risk-coverage curve, not just its area
    aurc, risk_curve = selective_prediction(eu_id, losses, n_bins=50)

``evaluate_ood`` takes ``metrics="all"`` or a list, and understands dynamic
specs such as ``"fpr@0.95"`` and ``"fnr@90%"``, so the claim can be stated at
the operating point you would actually run instead of as one AUROC.
``selective_prediction`` returns the area *and* the per-bin losses, and it is
that curve which :ref:`uq-evaluating` calls the honest artifact --- the scalar
alone hides where the gain sits.

Alongside these, ``probly.metrics`` holds the intrinsic scores --- calibration
error, coverage, set size --- which ask a different question: whether the
predicted distribution is *right*, rather than whether the derived score is
*useful*. The proper scoring rules live with the quantifiers instead, in
``probly.quantification.scoring_rule``, which is what lets a rule chosen for
evaluation be the same object as the one that generated the decomposition.
:ref:`uq-evaluating` draws that distinction properly.

.. seealso::

    Active learning needs a training loop rather than a snippet:
    :ref:`sphx_glr_auto_examples_active_learning_plot_active_learning_torch.py`
    compares uncertainty-driven acquisition against the random baseline.

    :ref:`pillars-composition` covers the dispatch mechanisms that keep the four
    stages independent of each other, and what that means for a new method.
