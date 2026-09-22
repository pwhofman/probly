.. _methods-calibration:

===========
Calibration
===========

.. currentmodule:: probly.transformation.calibration

Calibration is the least invasive intervention in this part. It leaves the
kind of representation untouched: a :ref:`first-order distribution
<uq-first-order>` goes in, and a first-order distribution comes out. What
changes is the *scale* of the probabilities, that is, the relation between the
confidence the model reports and the frequency with which it is correct. A
model that is right in 70% of the cases in which it predicts 0.99 is
miscalibrated in exactly this sense, however good its accuracy may be.

Modern deep networks have been observed to be systematically overconfident
:cite:`guoOnCalibration2017`, and calibration addresses this problem directly.
Note that it complements the other methods in this part rather than competing
with them: the members of an ensemble may each be miscalibrated, and averaging
their predictions need not remove the miscalibration.

All methods on this page are **post-hoc**. The base model is frozen, and a
small number of additional parameters are fitted on a held-out calibration
split via :func:`probly.calibrator.calibrate`. Fitting them on the training
split instead is a common mistake: since the model has been fitted to these
data, its predictions look well calibrated there even when they are not.

The methods differ mainly in how much freedom they give the recalibration map,
which amounts to the familiar bias-variance trade-off: a more flexible map can
fit more complex forms of miscalibration, but needs more calibration data to
do so without overfitting. Flexibility has a second price that is easily
overlooked. Only :ref:`temperature scaling <m-temperature-scaling>` is
guaranteed to leave the predicted class unchanged; all other methods may
change the accuracy of the model, not only its confidence.

.. _m-temperature-scaling:

:func:`temperature_scaling <probly.transformation.calibration.temperature_scaling>`
-----------------------------------------------------------------------------------

Divides the logits by a single learned scalar ``T`` before the softmax. A
temperature ``T > 1`` flattens the predicted distribution, and ``T < 1``
sharpens it. Since dividing all logits by the same positive constant cannot
change their order, the predicted class, and hence the accuracy, is provably
unaffected; only the confidence attached to it changes.

:Idea: Divide the logits by a single learned scalar ``T`` before the softmax.
:Representation: A :ref:`first-order distribution <uq-first-order>` with an
    unchanged ranking of classes, and therefore an unchanged accuracy.
:Advantages: A single parameter leaves essentially nothing to overfit, so a
    few hundred calibration points typically suffice. Despite its simplicity,
    it is often hard to improve on in practice, which is why it is the
    default.
:Disadvantages: A single scalar corrects the overall sharpness but not a
    systematic bias, and it is too coarse when the miscalibration differs
    across classes.
:Reference: :cite:`guoOnCalibration2017`

.. minigallery:: probly.method.calibration.temperature_scaling

.. _m-platt-scaling:

:func:`platt_scaling <probly.transformation.calibration.platt_scaling>`
-----------------------------------------------------------------------

The binary predecessor of temperature scaling fits a logistic regression
``sigmoid(a * s + b)`` that maps the model's score ``s`` to a probability.
Compared with temperature scaling, the intercept ``b`` makes it possible to
correct a systematic bias in addition to the sharpness. The same intercept
shifts the decision threshold, however, so the accuracy may change. In
``probly``, Platt scaling applies to binary classifiers only.

:Idea: Fit a logistic regression ``sigmoid(a * s + b)`` from the model's score
    to a probability.
:Representation: A :ref:`first-order distribution <uq-first-order>` over two
    classes, obtained from an uncalibrated score.
:Advantages: The intercept corrects a systematic bias as well as the
    sharpness; the natural choice when the base model outputs a score rather
    than a distribution.
:Disadvantages: Restricted to binary classification; two parameters can only
    realize a sigmoid-shaped correction, and the intercept may change the
    accuracy.
:Reference: :cite:`plattProbabilisticOutputs1999`

.. minigallery:: probly.method.calibration.platt_scaling

.. _m-vector-scaling:

:func:`vector_scaling <probly.transformation.calibration.vector_scaling>`
-------------------------------------------------------------------------

Extends temperature scaling to one temperature and one bias *per class*, which
amounts to a diagonal linear map on the logits. The additional freedom pays
off when the miscalibration is not uniform across classes, as is typically the
case under class imbalance, where the rare classes tend to be the poorly
calibrated ones. Unlike a single temperature, however, per-class parameters can
reorder the logits, so vector scaling may change the predicted class.

:Idea: One temperature and one bias per class, i.e., a diagonal linear map on
    the logits.
:Representation: A :ref:`first-order distribution <uq-first-order>`; the
    per-class parameters can reorder the logits, so the accuracy may change.
:Advantages: Handles miscalibration that varies across classes, as it
    typically arises under class imbalance.
:Disadvantages: May change the accuracy, and the number of parameters, and
    with it the amount of calibration data needed, grows with the number of
    classes.
:Reference: :cite:`guoOnCalibration2017`

.. minigallery:: probly.method.calibration.vector_scaling

.. _m-isotonic-regression:

:func:`isotonic_regression <probly.transformation.calibration.isotonic_regression>`
-----------------------------------------------------------------------------------

The non-parametric member of the family fits a monotone step function that
maps predicted to observed probabilities. Since it assumes nothing beyond
monotonicity, it can correct any monotone distortion of the probabilities,
including those that a sigmoid, and hence Platt scaling, cannot fit. This
flexibility comes at a price: on a small calibration split, the step function
overfits and produces piecewise-constant probabilities with visible plateaus.
In ``probly``, isotonic regression applies to binary classifiers only.

:Idea: Fit a monotone step function from predicted to observed probability.
:Representation: A :ref:`first-order distribution <uq-first-order>` over two
    classes. The ranking of inputs by score is preserved up to ties within a
    plateau, but predictions can move across the decision threshold, so the
    accuracy may change.
:Advantages: Non-parametric, so it corrects any monotone miscalibration,
    including shapes that no sigmoid can fit.
:Disadvantages: Restricted to binary classification, and overfits on a small
    calibration split, producing piecewise-constant probabilities with visible
    plateaus; advisable only when the split is large.
:Reference: :cite:`zadroznyTransformingClassifier2002`

.. minigallery:: probly.method.calibration.isotonic_regression

.. _m-dirichlet-calibration:

:func:`dirichlet_calibration <probly.transformation.calibration.dirichlet_calibration>`
---------------------------------------------------------------------------------------

The most general member of the family fits a full multinomial logistic
regression on the *log-probabilities*, ``q = softmax(W ln(p) + b)``, with a
dense ``num_classes x num_classes`` matrix ``W``. It contains temperature
scaling as a special case (and, for two classes, beta calibration), and it
recalibrates log-probabilities rather than logits, which distinguishes it from
matrix and vector scaling. The off-diagonal entries of ``W`` let the
calibrated probability of one class depend on the predicted probabilities of
the others, which makes it possible to correct systematic confusions between
classes.

:Idea: Multinomial logistic regression on the log-probabilities,
    ``q = softmax(W ln(p) + b)``.
:Representation: A recalibrated :ref:`first-order distribution
    <uq-first-order>`; the full map can reorder classes, so the accuracy may
    change.
:Advantages: The most general map on this page; contains temperature scaling
    as a special case and can correct systematic confusions between classes.
:Disadvantages: ``W`` grows quadratically in the number of classes, so on
    many-class problems the off-diagonal and intercept regularizers
    (``reg_lambda`` and ``reg_mu``) and additional calibration data become
    indispensable.
:Reference: :cite:`kullBeyondTemperatureScaling2019`

.. minigallery:: probly.method.calibration.dirichlet_calibration

Choosing a Method
-----------------

:ref:`temperature_scaling <m-temperature-scaling>` is the natural starting
point: it has a single parameter, cannot change the accuracy, and typically
captures most of the achievable improvement. A more flexible map is worth its
additional data requirements only when there is a specific reason to expect a
single temperature to fall short. Class imbalance is one such reason and
suggests :ref:`vector_scaling <m-vector-scaling>`; systematic confusion between
particular classes is another and suggests
:ref:`dirichlet_calibration <m-dirichlet-calibration>`. For binary problems,
:ref:`platt_scaling <m-platt-scaling>` corrects a bias that temperature
scaling cannot, and :ref:`isotonic_regression <m-isotonic-regression>` is the
method of choice when the miscalibration is not sigmoid-shaped and the
calibration split is large enough to support a non-parametric fit.

In any case, the result should be assessed with the calibration diagnostics
in :ref:`uq-evaluating`, on data held out from the calibration split as well
as from training.

Full API
--------

.. autosummary::
    :nosignatures:

    temperature_scaling
    platt_scaling
    vector_scaling
    isotonic_regression
    dirichlet_calibration
