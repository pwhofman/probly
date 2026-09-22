.. _methods-calibration:

===========
Calibration
===========

.. currentmodule:: probly.transformation.calibration

Calibration is the smallest intervention in this part. It does not change the
representation at all: a :ref:`first-order distribution <uq-first-order>` goes
in and a first-order distribution comes out, with the same ranking of classes
and therefore the same accuracy. What changes is the *scale* of the
probabilities --- the difference between a model that is right 70% of the time
when it says 0.99 and one that is right 99% of the time when it says 0.99.

Modern networks are systematically overconfident, and this is the direct fix.
It is also complementary to everything else here rather than an alternative:
the members of an ensemble can each be miscalibrated, and averaging them does
not repair it.

Every method on this page is **post-hoc**. The base model is frozen, and a
small number of parameters are fitted on a held-out calibration split via
:func:`probly.calibrator.calibrate`. Using the training split instead is the
classic mistake --- the model is already overfit to it, so it looks calibrated
there and is not.

The methods differ in how much freedom the reparameterization gets, which is
the usual bias-variance trade: more parameters fit the miscalibration better
and need more calibration data to do it without overfitting. Each entry below
follows the same fields, so they can be read against each other.

.. _m-temperature-scaling:

:func:`temperature_scaling <probly.transformation.calibration.temperature_scaling>`
-----------------------------------------------------------------------------------

Divides the logits by a single learned scalar ``T`` before the softmax. One
parameter for the entire model: ``T > 1`` softens the distribution, ``T < 1``
sharpens it, and because a monotone transform of all logits cannot reorder them
the accuracy is provably unchanged.

:Idea: Divide the logits by a single learned scalar ``T`` before the softmax.
:Representation: A :ref:`first-order distribution <uq-first-order>`; the
    ranking, and therefore the accuracy, is provably unchanged.
:Advantages: One parameter, essentially nothing to overfit, and a few hundred
    calibration points suffice; despite being the simplest method it is usually
    the best, and the default.
:Disadvantages: A single scalar corrects only the overall sharpness, not a
    systematic bias, and is too coarse when miscalibration varies across
    classes.
:Reference: :cite:`guoOnCalibration2017`

.. minigallery:: probly.method.calibration.temperature_scaling

.. _m-platt-scaling:

:func:`platt_scaling <probly.transformation.calibration.platt_scaling>`
-----------------------------------------------------------------------

The binary ancestor of temperature scaling: fit a logistic regression
``sigmoid(a * s + b)`` mapping the model's score to a probability. The added
intercept lets it correct a systematic bias as well as the sharpness, which
temperature scaling cannot do.

:Idea: Fit a logistic regression ``sigmoid(a * s + b)`` from the model's score
    to a probability.
:Representation: A :ref:`first-order distribution <uq-first-order>` from an
    uncalibrated score.
:Advantages: The intercept corrects a systematic bias as well as sharpness;
    still the right tool when the base model emits a score rather than a
    distribution.
:Disadvantages: Binary in origin; two parameters fit only a rigid sigmoid
    shape.
:Reference: :cite:`plattProbabilisticOutputs1999`

.. minigallery:: probly.method.calibration.platt_scaling

.. _m-vector-scaling:

:func:`vector_scaling <probly.transformation.calibration.vector_scaling>`
-------------------------------------------------------------------------

Temperature scaling with one temperature and one bias *per class*, i.e. a
diagonal linear map on the logits. This matters when miscalibration is not
uniform across classes --- typically under class imbalance, where the rare
classes are the badly calibrated ones.

:Idea: One temperature and one bias per class --- a diagonal linear map on the
    logits.
:Representation: A :ref:`first-order distribution <uq-first-order>`; the
    per-class bias can reorder logits, so accuracy can change.
:Advantages: Handles miscalibration that is not uniform across classes,
    typically under class imbalance.
:Disadvantages: Can change the accuracy, and needs meaningfully more
    calibration data than a single scalar.
:Reference: :cite:`guoOnCalibration2017`

.. minigallery:: probly.method.calibration.vector_scaling

.. _m-isotonic-regression:

:func:`isotonic_regression <probly.transformation.calibration.isotonic_regression>`
-----------------------------------------------------------------------------------

The non-parametric option: fit an arbitrary monotone step function from
predicted to true probability. Because it assumes only monotonicity, it can
correct miscalibration of any shape, including the non-monotone-in-temperature
kind that no scaling method can reach.

:Idea: Fit an arbitrary monotone step function from predicted to true
    probability.
:Representation: A :ref:`first-order distribution <uq-first-order>`; monotone,
    so the ranking and accuracy are preserved.
:Advantages: Non-parametric, so it corrects miscalibration of any shape,
    including what no scaling method can reach.
:Disadvantages: Overfits badly on a small calibration split, producing
    piecewise-constant probabilities with visible plateaus; use it only when
    the split is large.
:Reference: :cite:`zadroznyTransformingClassifier2002`

.. minigallery:: probly.method.calibration.isotonic_regression

.. _m-dirichlet-calibration:

:func:`dirichlet_calibration <probly.transformation.calibration.dirichlet_calibration>`
---------------------------------------------------------------------------------------

The most general of the family: a full multinomial logistic regression on the
*log-probabilities*, ``q = softmax(W ln(p) + b)`` with a dense
``num_classes x num_classes`` matrix ``W``. It generalizes temperature scaling
(and, for two classes, beta calibration) and recalibrates probabilities rather
than logits, which is what separates it from matrix and vector scaling.

:Idea: Multinomial logistic regression on the log-probabilities,
    ``q = softmax(W ln(p) + b)``.
:Representation: A recalibrated :ref:`first-order distribution <uq-first-order>`
    over probabilities rather than logits.
:Advantages: The most general reparameterization here; generalizes temperature
    scaling and corrects the confusion structure between classes.
:Disadvantages: ``W`` grows quadratically in the number of classes, so on
    many-class problems the two L2 regularizers and extra calibration data are
    not optional.
:Reference: :cite:`kullBeyondTemperatureScaling2019`

.. minigallery:: probly.method.calibration.dirichlet_calibration

Choosing a Method
-----------------

Start with :ref:`temperature_scaling <m-temperature-scaling>`; it is one
parameter, it cannot hurt accuracy, and it captures most of the available
improvement. Move to :ref:`vector_scaling <m-vector-scaling>` if the classes
are imbalanced, to :ref:`dirichlet_calibration <m-dirichlet-calibration>` if
the confusion structure between classes is what needs correcting, and to
:ref:`isotonic_regression <m-isotonic-regression>` if the miscalibration is not
a monotone rescaling and the calibration split is large enough to support it.

Measure the result with the calibration diagnostics in :ref:`uq-evaluating`,
always on data held out from the calibration split itself.

Full API
--------

.. autosummary::
    :nosignatures:

    temperature_scaling
    platt_scaling
    vector_scaling
    isotonic_regression
    dirichlet_calibration
