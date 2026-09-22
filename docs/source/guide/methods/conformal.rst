.. _methods-conformal:

====================
Conformal Prediction
====================

.. currentmodule:: probly.transformation.conformal

Conformal prediction takes a route opposite to that of the other methods in
this part. Rather than trying to describe the predictive distribution more
faithfully, it gives up on the distribution and returns a :ref:`set of
outcomes <uq-sets>` instead, a set of labels in classification or an interval
in regression, equipped with a *finite-sample coverage guarantee*. More
specifically, if the calibration and test data are exchangeable, the true
outcome lies in the returned set with probability at least ``1 - alpha``,
however good or bad the underlying model may be
:cite:`angelopoulosGentleIntroduction2021`.

All variants on this page share the same mechanism, which consists of three
steps. First, a **non-conformity score** ``s(x, y)`` measures how poorly a
candidate outcome ``y`` conforms to the model's prediction for ``x``. Second,
this score is evaluated on a held-out calibration split, and its empirical
``1 - alpha`` quantile is stored. Third, at prediction time, the set comprises
every candidate whose score does not exceed this quantile.

Consequently, the guarantee holds for any choice of score, and the design
effort goes entirely into the score. The score does not affect *whether*
coverage holds, only how large the sets are and how well their size adapts to
the difficulty of the input; it is along this axis that the variants below
differ.

.. important::

    Coverage is **marginal**, i.e., it holds on average over the distribution
    of inputs, not conditionally on a particular input. A predictor may thus
    achieve 90% coverage overall while systematically undercovering a
    difficult subgroup. Adaptive scores can narrow this gap, but none of them
    closes it; without further assumptions, exact conditional coverage is in
    general unattainable.

    Before it can predict, the wrapper must be calibrated on data disjoint
    from the training data; see :func:`probly.calibrator.calibrate`.

Classification
--------------

In classification, the prediction is a set of labels, and the scores differ
essentially in how much probability mass they require before they stop adding
classes.

.. _m-conformal-lac:

:func:`conformal_lac <probly.transformation.conformal.conformal_lac>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Least ambiguous set-valued classifier.** The score is simply ``1 - p_y``, so
a label enters the set if the model assigns it sufficient probability. Since
this threshold on ``p_y`` is the same for every input, the resulting sets are
small on average, but the coverage target is met partly by overcovering easy
inputs and undercovering difficult ones.

:Idea: Non-conformity score ``1 - p_y``; a label is retained if it receives
    enough probability.
:Representation: A :ref:`set of labels <uq-sets>` with marginal coverage.
:Advantages: Typically the smallest average sets of all scores on this page,
    which is why it is the default.
:Disadvantages: Little adaptivity; the marginal target is met partly by
    undercovering hard inputs and overcovering easy ones.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_lac

.. _m-conformal-aps:

:func:`conformal_aps <probly.transformation.conformal.conformal_aps>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Adaptive prediction sets.** The classes are sorted by predicted probability,
and the score of a label is the cumulative probability mass up to and
including it, with a uniform random term that removes the discreteness of the
last step. As a result, the set grows when the predicted distribution is flat
and shrinks when it is peaked.

:Idea: Accumulate the sorted class probabilities; the score of a label is the
    cumulative mass up to it, with a random tie-break.
:Representation: A :ref:`set of labels <uq-sets>` with marginal coverage.
:Advantages: Sets grow when the distribution is flat and shrink when it is
    peaked, which yields noticeably better conditional coverage than LAC.
:Disadvantages: On many-class problems, the long tail of small probabilities
    can inflate the sets considerably.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_aps

.. _m-conformal-saps:

:func:`conformal_saps <probly.transformation.conformal.conformal_saps>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Sorted adaptive prediction sets.** Only the top-1 probability is treated as
reliable information; the rest of the tail is replaced by a penalty that is
linear in the rank and weighted by ``lambda_val``. The premise is that the tail
probabilities of a many-class classifier are largely noise, so that their
*order* carries information while their values do not.

:Idea: Keep only the top-1 probability and replace the tail by a penalty
    linear in the rank, weighted by ``lambda_val``.
:Representation: A :ref:`set of labels <uq-sets>` with marginal coverage.
:Advantages: Retains most of the adaptivity of APS while avoiding the large
    sets caused by the noisy tail.
:Disadvantages: Adds the hyperparameter ``lambda_val``, and relies on the
    order of the tail being reliable.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_saps

.. _m-conformal-raps:

:func:`conformal_raps <probly.transformation.conformal.conformal_raps>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Regularized adaptive prediction sets.** APS with an explicit penalty
``lambda_reg`` on every class included beyond rank ``k_reg``. The diagnosis is
the same as for SAPS, namely that the tail is what inflates the sets, but the
remedy is more conservative: the tail probabilities are kept, and only the
cost of reaching into the tail increases.

:Idea: APS with an explicit penalty ``lambda_reg`` on every class included
    beyond rank ``k_reg``.
:Representation: A :ref:`set of labels <uq-sets>` with marginal coverage.
:Advantages: Directly addresses the long tail of APS, yielding small sets
    without giving up adaptivity.
:Disadvantages: Two hyperparameters, ``lambda_reg`` and ``k_reg``, to both of
    which the set sizes are sensitive.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_raps

Regression
----------

In regression, the prediction is an interval, and the main question is whether
its width may vary with the input.

.. _m-conformal-absolute-error:

:func:`conformal_absolute_error <probly.transformation.conformal.conformal_absolute_error>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Absolute residual.** With the score ``|y - y_hat|``, the calibrated quantile
is simply added to and subtracted from the point prediction, so the interval
has the same width for every input.

:Idea: Score ``|y - y_hat|``; add and subtract the calibrated quantile around
    the point prediction.
:Representation: An :ref:`interval <uq-sets>` with marginal coverage.
:Advantages: The simplest conformal regressor, and a natural first choice.
:Disadvantages: Constant width everywhere: valid on average, but insensitive
    to the fact that some inputs are harder to predict than others.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

The figure below, taken from
:ref:`sphx_glr_auto_examples_conformal_plot_regression_sklearn.py`, shows the
resulting intervals for a random forest on the Diabetes data, with the test
points sorted by their true value. The bands only appear to vary because they
are centered on point predictions that do; their width is the same for every
input. About 95% of the true values fall inside, as calibrated for
``alpha = 0.05``, but the intervals are as wide for the points that the model
predicts well as for those it predicts poorly.

.. image:: /auto_examples/conformal/images/sphx_glr_plot_regression_sklearn_001.png
    :alt: Prediction intervals of constant width around the point predictions
        of a random forest on the Diabetes test set, sorted by true value, with
        almost all true values falling inside their interval.
    :width: 100%

.. minigallery:: probly.method.conformal.conformal_absolute_error

.. _m-conformal-cqr:

:func:`conformal_cqr <probly.transformation.conformal.conformal_cqr>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Conformalized quantile regression.** The base model predicts a lower and an
upper quantile, and the score ``max(q_lo - y, y - q_hi)`` measures how far the
true value falls outside the predicted interval, with negative values
indicating that it falls inside. The calibrated quantile then shifts both
endpoints outward, or inward, by the same constant.

:Idea: Shift the endpoints of a predicted quantile interval by a calibrated
    constant, using the score ``max(q_lo - y, y - q_hi)``.
:Representation: An :ref:`interval <uq-sets>` with marginal coverage.
:Advantages: The width varies with the input because the base model lets it
    vary; conformalization only corrects the coverage level.
:Disadvantages: Requires a quantile-regression base model, and an additive
    correction cannot rescale an interval whose width is off by a factor.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_cqr

.. _m-conformal-cqr-r:

:func:`conformal_cqr_r <probly.transformation.conformal.conformal_cqr_r>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Normalized CQR.** The CQR score is divided by the width of the predicted
interval, so that the correction becomes multiplicative rather than additive.
The difference matters because an additive correction widens every interval by
the same amount, whereas a multiplicative one widens each interval in
proportion to its predicted width. The latter is what a base model requires
whose intervals have the right shape but the wrong overall scale.

:Idea: Divide the CQR score by the predicted interval width, which makes the
    correction multiplicative.
:Representation: An :ref:`interval <uq-sets>` with marginal coverage.
:Advantages: Rewards a base model that signals where it is unsure; preferable
    to plain CQR when the predicted intervals are right in shape but wrong in
    scale.
:Disadvantages: Still requires a quantile-regression base model, and degrades
    when the predicted widths are unreliable.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_cqr_r

.. _m-conformal-uacqr:

:func:`conformal_uacqr <probly.transformation.conformal.conformal_uacqr>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Uncertainty-aware CQR.** Based on an *ensemble* of quantile regressors, the
CQR score is normalized by the ensemble's standard deviation at each endpoint.
Since this standard deviation reflects disagreement among the members rather
than noise in the target, the intervals widen where the model is uncertain
about the quantiles themselves, which one may interpret as an epistemic
component of the interval width.

:Idea: Normalize the CQR score by the standard deviation of an ensemble of
    quantile regressors at each endpoint.
:Representation: An :ref:`interval <uq-sets>` with marginal coverage.
:Advantages: The scaling factor can be read as epistemic: the interval widens
    where the members disagree about the quantiles, not merely where the
    target is noisy.
:Disadvantages: The most expensive option on this page, since it requires
    training an ensemble.
:Reference: :cite:`angelopoulosGentleIntroduction2021`

.. minigallery:: probly.method.conformal.conformal_uacqr

Choosing a Score
----------------

For classification, :func:`conformal_lac` is the natural choice when the
average set size matters most, and :func:`conformal_raps` when conditional
coverage does; :func:`conformal_aps` serves as the reference point against
which both are usually compared. For regression,
:func:`conformal_absolute_error` is the default unless the base model can
predict quantiles, in which case :func:`conformal_cqr_r` is preferable.

Whichever score is chosen, it should be evaluated on both axes at once.
Coverage alone is uninformative, since it is guaranteed by construction; what
separates the scores is the *set size at the target coverage*, together with
how evenly the coverage is distributed across inputs.

The same calibration machinery can also be applied to distributions instead
of labels, which leads to the :ref:`conformal credal sets
<m-conformal-credal-set>`.

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
