.. _methods-credal:

===========
Credal Sets
===========

.. currentmodule:: probly.method

A :ref:`credal set <uq-credal>` is a *set* of probability distributions
considered admissible, and a credal method is one that declines to pick a
single member of it. Where a second-order method says "the distribution is
probably around here, with this much spread", a credal method says "the
distribution is one of these, and I will not rank them". Every query is then
answered by an interval --- a lower and an upper probability --- instead of a
number.

``probly`` represents credal sets two ways. A **convex credal set** is stored
by its vertices: each ensemble member contributes one distribution, and the set
is their convex hull. **Probability intervals** store a lower and an upper
bound per class, which is coarser but far cheaper to reason with. The methods
below differ mostly in how they produce those vertices or bounds.

Every entry follows the same fields, so they can be read against each other,
and links to the worked example in the gallery.

.. _m-credal-wrapper:

:func:`credal_wrapper <probly.method.credal_wrapper>`
-----------------------------------------------------

The most direct construction: replicate the classifier into an ensemble, then
summarize the members' predictions as per-class probability intervals rather
than averaging them. Averaging is exactly the step that destroys the
information a credal set keeps --- two members predicting 0.1 and 0.9 and two
members both predicting 0.5 have the same mean and very different credal sets.

:Idea: Summarize ensemble members as per-class probability intervals instead of
    averaging them.
:Representation: :ref:`Probability intervals <uq-credal>` credal set.
:Advantages: Direct and simple; keeps the disagreement that averaging would
    discard.
:Disadvantages: Needs an ensemble; intervals admit class-probability
    combinations no member actually predicted.
:Reference: :cite:`wangCredalWrapper2024`

.. minigallery:: probly.method.credal_wrapper

.. _m-credal-ensembling:

:func:`credal_ensembling <probly.method.credal_ensembling>`
-----------------------------------------------------------

Also ensemble-based, but keeps the members as *vertices* of a convex credal set
instead of collapsing them to intervals. That preserves the dependence between
classes: probability intervals allow combinations of class probabilities that
no member actually predicted, whereas the convex hull only contains mixtures of
real member predictions.

:Idea: Keep ensemble members as the vertices of a convex credal set.
:Representation: :ref:`Convex credal set <uq-credal>` --- member predictions as
    vertices.
:Advantages: Preserves the dependence between classes; the hull contains only
    real mixtures of member predictions.
:Disadvantages: Needs an ensemble; reasoning over a vertex set is costlier than
    over intervals.
:Reference: :cite:`nguyenCredalEnsembling2025`

.. minigallery:: probly.method.credal_ensembling

.. _m-credal-bnn:

:func:`credal_bnn <probly.method.credal_bnn>`
---------------------------------------------

The same idea with Bayesian members. Rather than one variational posterior, an
ensemble of Bayesian neural networks is trained, and the resulting predictive
distributions become the vertices of a convex credal set. This addresses the
standard objection to a single BNN: the prior and the variational family are
themselves choices, and a credal set can carry several of them without
pretending one is correct.

:Idea: Ensemble Bayesian neural networks; their predictive distributions are
    the vertices.
:Representation: :ref:`Convex credal set <uq-credal>` from Bayesian members.
:Advantages: Carries several prior and variational-family choices without
    pretending one is correct.
:Disadvantages: Pays the cost of training several BNNs, each with a variational
    scheme.
:Reference: :cite:`caprioCredalBayesian2024`

.. minigallery:: probly.method.credal_bnn

.. _m-credal-net:

:func:`credal_net <probly.method.credal_net.credal_net>`
--------------------------------------------------------

Instead of ensembling, the *weights* become intervals: every layer is replaced
by an interval-arithmetic counterpart, and the credal set falls out of
propagating those intervals to the output. One network, one forward pass, no
members to train --- at the price of interval arithmetic's tendency to widen as
it goes deeper.

:Idea: Replace each layer with an interval-arithmetic counterpart and propagate
    the weight intervals to the output.
:Representation: :ref:`Probability intervals <uq-credal>` credal set from one
    network.
:Advantages: One network, one forward pass, no members to train.
:Disadvantages: Interval arithmetic tends to widen with depth, so deep networks
    give loose bounds.
:Reference: :cite:`saleSecondOrder2024`

.. minigallery:: probly.method.credal_net

.. _m-credal-relative-likelihood:

:func:`credal_relative_likelihood <probly.method.credal_relative_likelihood>`
-----------------------------------------------------------------------------

Builds its members deliberately rather than randomly: one member per class,
each initialized with a bias towards that class. Each member answers "how well
can the data be explained if I lean towards class ``k``?", and the relative
likelihood of those explanations bounds the credal set. It gives a systematic,
reproducible cover of the simplex instead of relying on random initialization
for diversity.

:Idea: One deliberately class-biased member per class; the relative likelihood
    of their explanations bounds the set.
:Representation: Credal set from relative-likelihood level sets.
:Advantages: A systematic, reproducible cover of the simplex rather than
    relying on random initialization for diversity.
:Disadvantages: Trains one member per class, so cost grows with the number of
    classes.
:Reference: :cite:`lohrCredalPrediction2025`

.. minigallery:: probly.method.credal_relative_likelihood

.. _m-efficient-credal-prediction:

:func:`efficient_credal_prediction <probly.method.efficient_credal_prediction>`
-------------------------------------------------------------------------------

The cheap option. The base classifier is left alone and keeps returning its
ordinary :ref:`categorical distribution <uq-first-order>`; the credal view is
constructed on demand from that distribution plus bounds calibrated on a
held-out split. No ensemble, no interval layers, no retraining --- ask for
``predict`` and you get the point distribution, ask the representer and you get
the credal set.

:Idea: Keep the base classifier's categorical output and build the credal view
    on demand from calibrated bounds.
:Representation: Credal set from a :ref:`first-order distribution
    <uq-first-order>` plus held-out bounds.
:Advantages: No ensemble, no interval layers, no retraining; ``predict`` still
    returns the point distribution.
:Disadvantages: Needs a held-out calibration split, and the set is only as good
    as the calibrated bounds.
:Reference: :cite:`hofmanEfficientCredal2026`

.. minigallery:: probly.method.efficient_credal_prediction

.. _m-conformal-credal-set:

Conformal Credal Sets
---------------------

The bridge between this family and :ref:`methods-conformal`. Rather than
deriving the credal set from disagreement inside the model, a ball is placed
around the predicted distribution and its radius is *calibrated* on held-out
data, so the resulting credal set inherits conformal prediction's coverage
guarantee. The five variants differ only in which distance defines the ball:
total variation, Kullback-Leibler, Wasserstein, inner product, or a Dirichlet
relative-likelihood level set. The choice of distance determines the shape of
the set on the simplex, and therefore which distributions near the boundary
survive.

:Idea: Place a ball around the predicted distribution and calibrate its radius
    on held-out data.
:Representation: Credal set carrying a conformal coverage guarantee.
:Advantages: Inherits conformal prediction's finite-sample coverage guarantee;
    the distance choice tunes the shape of the set.
:Disadvantages: Needs a held-out calibration split; coverage is marginal rather
    than conditional, as for all conformal methods.
:Reference: :cite:`saleSecondOrder2024`, :cite:`angelopoulosGentleIntroduction2021`

.. autosummary::
    :nosignatures:

    ~probly.transformation.conformal_credal_set.conformal_total_variation
    ~probly.transformation.conformal_credal_set.conformal_kullback_leibler
    ~probly.transformation.conformal_credal_set.conformal_wasserstein_distance
    ~probly.transformation.conformal_credal_set.conformal_inner_product
    ~probly.transformation.conformal_credal_set.conformal_dirichlet_relative_likelihood

.. minigallery:: probly.transformation.conformal_credal_set.conformal_total_variation probly.transformation.conformal_credal_set.conformal_kullback_leibler probly.transformation.conformal_credal_set.conformal_wasserstein_distance probly.transformation.conformal_credal_set.conformal_inner_product probly.transformation.conformal_credal_set.conformal_dirichlet_relative_likelihood

Full API
--------

.. autosummary::
    :nosignatures:

    credal_wrapper
    credal_ensembling
    credal_bnn
    ~credal_net.credal_net
    credal_relative_likelihood
    efficient_credal_prediction
    ~probly.transformation.conformal_credal_set
