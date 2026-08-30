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

Each entry links to the worked example in the gallery.

.. _m-credal-wrapper:

:func:`credal_wrapper <probly.method.credal_wrapper>`
-----------------------------------------------------

The most direct construction: replicate the classifier into an ensemble, then
summarize the members' predictions as per-class probability intervals rather
than averaging them. Averaging is exactly the step that destroys the
information a credal set keeps --- two members predicting 0.1 and 0.9 and two
members both predicting 0.5 have the same mean and very different credal sets.

:cite:`wangCredalWrapper2024`

.. minigallery:: probly.method.credal_wrapper

.. _m-credal-ensembling:

:func:`credal_ensembling <probly.method.credal_ensembling>`
-----------------------------------------------------------

Also ensemble-based, but keeps the members as *vertices* of a convex credal set
instead of collapsing them to intervals. That preserves the dependence between
classes: probability intervals allow combinations of class probabilities that
no member actually predicted, whereas the convex hull only contains mixtures of
real member predictions.

:cite:`nguyenCredalEnsembling2025`

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

:cite:`caprioCredalBayesian2024`

.. minigallery:: probly.method.credal_bnn

.. _m-credal-net:

:func:`credal_net <probly.method.credal_net.credal_net>`
--------------------------------------------------------

Instead of ensembling, the *weights* become intervals: every layer is replaced
by an interval-arithmetic counterpart, and the credal set falls out of
propagating those intervals to the output. One network, one forward pass, no
members to train --- at the price of interval arithmetic's tendency to widen as
it goes deeper.

:cite:`saleSecondOrder2024`

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

:cite:`lohrCredalPrediction2025`

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

:cite:`hofmanEfficientCredal2026`

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

:cite:`saleSecondOrder2024`, :cite:`angelopoulosGentleIntroduction2021`

.. autosummary::
    :nosignatures:

    conformal_total_variation
    conformal_kullback_leibler
    conformal_wasserstein_distance
    conformal_inner_product
    conformal_dirichlet_relative_likelihood

.. minigallery:: probly.method.conformal_total_variation probly.method.conformal_kullback_leibler probly.method.conformal_wasserstein_distance probly.method.conformal_inner_product probly.method.conformal_dirichlet_relative_likelihood

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
