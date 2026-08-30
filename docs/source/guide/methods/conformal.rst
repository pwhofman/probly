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
adaptive the sets are. That is the axis on which the variants below differ.

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

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: :func:`conformal_lac <probly.transformation.conformal.conformal_lac>`

        **Least ambiguous set-valued classifier.** The score is simply
        ``1 - p_y``: a label is in the set if the model gave it enough
        probability.

        Produces the *smallest* average set size of any score here, which is
        why it is the default choice. The cost is adaptivity --- it hits its
        marginal target partly by undercovering hard inputs and overcovering
        easy ones.


    .. grid-item-card:: :func:`conformal_aps <probly.transformation.conformal.conformal_aps>`

        **Adaptive prediction sets.** Sort the classes by probability and
        accumulate; the score of a label is the cumulative mass down to it,
        with a uniform random term breaking the discreteness of the last step.

        Sets grow when the distribution is flat and shrink when it is peaked,
        which gives far better conditional coverage than LAC --- at the price
        of noticeably larger sets on many-class problems.


    .. grid-item-card:: :func:`conformal_saps <probly.transformation.conformal.conformal_saps>`

        **Sorted adaptive prediction sets.** Keeps only the top-1 probability
        as real information and replaces the rest of the tail with a linear
        penalty in the rank, weighted by ``lambda_val``.

        Built on the observation that the ordering of the tail is reliable but
        its probabilities are not. Retains most of APS's adaptivity while
        cutting the set sizes that the noisy tail causes.


    .. grid-item-card:: :func:`conformal_raps <probly.transformation.conformal.conformal_raps>`

        **Regularized adaptive prediction sets.** APS plus an explicit penalty
        ``lambda_reg`` on every class included beyond rank ``k_reg``.

        The direct fix for APS's long tail: the regularizer makes including a
        fifteenth class expensive, so the sets stay small without abandoning
        adaptivity. Two knobs to tune, and worth tuning.


Regression
----------

Intervals rather than label sets. The question becomes whether the interval
width is allowed to vary with the input.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: :func:`conformal_absolute_error <probly.transformation.conformal.conformal_absolute_error>`

        **Absolute residual.** The score is ``|y - y_hat|``, so the calibrated
        quantile is added and subtracted around the point prediction.

        The simplest possible conformal regressor, and the one to reach for
        first. Its interval has *constant width everywhere*: valid on average,
        but blind to the fact that some inputs are harder than others.


    .. grid-item-card:: :func:`conformal_cqr <probly.transformation.conformal.conformal_cqr>`

        **Conformalized quantile regression.** Requires a model predicting a
        lower and an upper quantile; the score
        ``max(q_lo - y, y - q_hi)`` measures how far outside the predicted
        interval the truth fell.

        The calibrated quantile then shifts both endpoints outward (or inward)
        by a constant. Width varies with the input because the *base model*
        makes it vary --- conformalization only corrects the level.


    .. grid-item-card:: :func:`conformal_cqr_r <probly.transformation.conformal.conformal_cqr_r>`

        **Normalized CQR.** The same score divided by the predicted interval
        width, so the correction becomes multiplicative rather than additive.

        Wide predicted intervals produce smaller normalized scores, which
        rewards a base model that admits when it is unsure. Preferable to plain
        CQR whenever the base quantiles are already roughly right in shape but
        wrong in scale.


    .. grid-item-card:: :func:`conformal_uacqr <probly.transformation.conformal.conformal_uacqr>`

        **Uncertainty-aware CQR.** Takes an *ensemble* of quantile regressors
        and normalizes the CQR score by the ensemble's standard deviation at
        each endpoint.

        The scaling factor is now epistemic: the interval widens where the
        members disagree about where the quantiles are, not merely where the
        target is noisy. The most expensive option here --- it needs the
        ensemble.


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
