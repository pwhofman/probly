.. _methods-conformal:

====================
Conformal Prediction
====================

.. currentmodule:: probly.transformation.conformal

Conformal prediction takes the opposite route to everything else in this part.
It does not try to describe the predictive distribution better; it gives up on
the distribution and returns a :ref:`set of outcomes <uq-sets>` instead --- a
set of labels, or an interval --- with a *finite-sample coverage guarantee*.
Under exchangeability of the calibration and test data, the true label falls
inside the returned set with probability at least ``1 - alpha``, whatever the
underlying model does. :cite:`angelopoulosGentleIntroduction2021`

The mechanism is the same for every variant on this page, and there are only
three steps. A **non-conformity score** ``s(x, y)`` measures how badly the
model's output fits a candidate label. That score is evaluated on a held-out
calibration split, and its empirical ``1 - alpha`` quantile is stored. At
prediction time the set is every label whose score falls below that quantile.

The guarantee is therefore free, and the score is where all the design work
lies: it does not affect *whether* coverage holds, only how large and how
adaptive the sets are. That is the axis on which the variants below differ, and
each entry follows the same fields so they can be read against each other.

.. important::

    Coverage is **marginal**, averaged over the whole distribution --- not
    conditional on the input. A predictor can hold 90% coverage overall while
    systematically undercovering a hard subgroup. Adaptive scores narrow that
    gap; none of them close it.

    The wrapper must be calibrated on data disjoint from training before it can
    predict; see :func:`probly.calibrator.calibrate`.

Classification
--------------

Sets of labels. The scores differ in how much probability mass they require
before they stop adding classes.

.. _m-conformal-lac:

:func:`conformal_lac <probly.transformation.conformal.conformal_lac>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Least ambiguous set-valued classifier.** The score is simply ``1 - p_y``: a
label is in the set if the model gave it enough probability.

:Idea: Non-conformity score ``1 - p_y`` --- keep a label if it received enough
    probability.
:Representation: A :ref:`set of labels <uq-sets>` with marginal coverage.
:Advantages: The smallest average set size of any score here, which is why it
    is the default choice.
:Disadvantages: Little adaptivity --- it hits its marginal target partly by
    undercovering hard inputs and overcovering easy ones.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_lac

.. _m-conformal-aps:

:func:`conformal_aps <probly.transformation.conformal.conformal_aps>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Adaptive prediction sets.** Sort the classes by probability and accumulate;
the score of a label is the cumulative mass down to it, with a uniform random
term breaking the discreteness of the last step.

:Idea: Accumulate sorted class probabilities; the score is the cumulative mass
    down to a label, with a random tie-break.
:Representation: A :ref:`set of labels <uq-sets>` with marginal coverage.
:Advantages: Sets grow when the distribution is flat and shrink when it is
    peaked, giving far better conditional coverage than LAC.
:Disadvantages: Noticeably larger sets on many-class problems.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_aps

.. _m-conformal-saps:

:func:`conformal_saps <probly.transformation.conformal.conformal_saps>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Sorted adaptive prediction sets.** Keeps only the top-1 probability as real
information and replaces the rest of the tail with a linear penalty in the
rank, weighted by ``lambda_val``.

:Idea: Keep only the top-1 probability and replace the tail with a rank-linear
    penalty weighted by ``lambda_val``.
:Representation: A :ref:`set of labels <uq-sets>` with marginal coverage.
:Advantages: Retains most of APS's adaptivity while cutting the set sizes the
    noisy tail causes.
:Disadvantages: Adds the ``lambda_val`` hyperparameter, and leans on the tail
    ordering being reliable.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_saps

.. _m-conformal-raps:

:func:`conformal_raps <probly.transformation.conformal.conformal_raps>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Regularized adaptive prediction sets.** APS plus an explicit penalty
``lambda_reg`` on every class included beyond rank ``k_reg``.

:Idea: APS with an explicit penalty ``lambda_reg`` on every class included
    beyond rank ``k_reg``.
:Representation: A :ref:`set of labels <uq-sets>` with marginal coverage.
:Advantages: The direct fix for APS's long tail --- small sets without
    abandoning adaptivity.
:Disadvantages: Two knobs to tune (``lambda_reg`` and ``k_reg``), and worth
    tuning.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_raps

Regression
----------

Intervals rather than label sets. The question becomes whether the interval
width is allowed to vary with the input.

.. _m-conformal-absolute-error:

:func:`conformal_absolute_error <probly.transformation.conformal.conformal_absolute_error>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Absolute residual.** The score is ``|y - y_hat|``, so the calibrated quantile
is added and subtracted around the point prediction.

:Idea: Score ``|y - y_hat|``; add and subtract the calibrated quantile around
    the point prediction.
:Representation: An :ref:`interval <uq-sets>` with marginal coverage.
:Advantages: The simplest possible conformal regressor, and the one to reach
    for first.
:Disadvantages: Constant width everywhere --- valid on average, but blind to
    the fact that some inputs are harder than others.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_absolute_error

.. _m-conformal-cqr:

:func:`conformal_cqr <probly.transformation.conformal.conformal_cqr>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Conformalized quantile regression.** Requires a model predicting a lower and
an upper quantile; the score ``max(q_lo - y, y - q_hi)`` measures how far
outside the predicted interval the truth fell, and the calibrated quantile then
shifts both endpoints outward (or inward) by a constant.

:Idea: Score ``max(q_lo - y, y - q_hi)`` shifts the predicted quantile interval
    endpoints by a constant.
:Representation: An :ref:`interval <uq-sets>` with marginal coverage.
:Advantages: Width varies with the input because the base model makes it vary;
    conformalization only corrects the level.
:Disadvantages: Needs a quantile-regression base model; the additive correction
    cannot rescale a badly-scaled interval.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_cqr

.. _m-conformal-cqr-r:

:func:`conformal_cqr_r <probly.transformation.conformal.conformal_cqr_r>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Normalized CQR.** The same score divided by the predicted interval width, so
the correction becomes multiplicative rather than additive.

:Idea: The CQR score divided by the predicted interval width, making the
    correction multiplicative.
:Representation: An :ref:`interval <uq-sets>` with marginal coverage.
:Advantages: Rewards a base model that admits when it is unsure; preferable to
    plain CQR when the base quantiles are right in shape but wrong in scale.
:Disadvantages: Still needs a quantile-regression base model, and degrades when
    the predicted widths are unreliable.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_cqr_r

.. _m-conformal-uacqr:

:func:`conformal_uacqr <probly.transformation.conformal.conformal_uacqr>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Uncertainty-aware CQR.** Takes an *ensemble* of quantile regressors and
normalizes the CQR score by the ensemble's standard deviation at each endpoint.

:Idea: Normalize the CQR score by an ensemble of quantile regressors' standard
    deviation at each endpoint.
:Representation: An :ref:`interval <uq-sets>` with marginal coverage.
:Advantages: The scaling factor is epistemic --- the interval widens where the
    members disagree about the quantiles, not merely where the target is noisy.
:Disadvantages: The most expensive option here --- it needs the ensemble.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_uacqr

Choosing a Score
----------------

For classification, start with :func:`conformal_lac` if you care about average
set size and :func:`conformal_raps` if you care about conditional coverage;
:func:`conformal_aps` is the reference point both are measured against. For
regression, :func:`conformal_absolute_error` unless you can predict quantiles,
in which case :func:`conformal_cqr_r`. Whatever you pick, evaluate it on both
axes at once --- coverage alone is uninformative, because it is guaranteed by
construction, so it is *set size at the target coverage* that separates the
scores.

Related: :ref:`m-conformal-credal-set` applies the same calibration machinery
to distributions instead of labels.

Full API
--------

.. autosummary::
    :nosignatures:

    conformal_lac
    conformal_aps
    conformal_saps
    conformal_raps
    conformal_absolute_error
    conformal_cqr
    conformal_cqr_r
    conformal_uacqr
