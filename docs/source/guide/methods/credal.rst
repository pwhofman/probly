.. _methods-credal:

===========
Credal Sets
===========

.. currentmodule:: probly.method

A :ref:`credal set <uq-credal>` is a *set* of probability distributions that
are all considered admissible, and a credal method is one that declines to
single out a member of this set. Where a second-order method says, roughly,
"the distribution is probably around here, with this much spread", a credal
method says "the distribution is one of these, and I will not rank them".
Consequently, every probabilistic query is answered by an interval, bounded by
a lower and an upper probability, instead of a single number.

``probly`` represents credal sets in two ways. A **convex credal set** is
stored by its vertices, for instance one distribution per ensemble member, and
comprises all of their mixtures, i.e., their convex hull. **Probability
intervals** store a lower and an upper bound per class; this representation is
coarser, since it ignores the dependence between classes, but considerably
cheaper to reason with. The methods below differ mainly in how they produce
these vertices or bounds.

.. _m-credal-wrapper:

:func:`credal_wrapper <probly.method.credal_wrapper>`
-----------------------------------------------------

The most direct construction replicates the classifier into an ensemble and
summarizes the members' predictions by per-class probability intervals instead
of averaging them. Averaging is precisely the step that destroys the
information a credal set retains: in the binary case, two members predicting
0.1 and 0.9 and two members both predicting 0.5 have the same mean, but very
different credal sets.

:Idea: Summarize the ensemble members by per-class probability intervals
    instead of averaging them.
:Representation: A credal set given by
    :ref:`probability intervals <uq-credal>`.
:Advantages: Simple and direct; retains the disagreement that averaging would
    discard.
:Disadvantages: Requires an ensemble; the intervals also admit distributions
    that are not mixtures of any member predictions.
:Reference: :cite:`wangCredalWrapper2024`

.. minigallery:: probly.method.credal_wrapper

.. _m-credal-ensembling:

:func:`credal_ensembling <probly.method.credal_ensembling>`
-----------------------------------------------------------

Also ensemble-based, but the members' predictions are kept as the *vertices*
of a convex credal set rather than being collapsed into intervals, optionally
after discarding a fraction of the members whose predictions lie farthest from
the ensemble mean. Keeping the vertices preserves the dependence between
classes: whereas probability intervals also admit distributions that are not
mixtures of any member predictions, the convex hull contains nothing else.

:Idea: Keep the ensemble members' predictions as the vertices of a convex
    credal set.
:Representation: A :ref:`convex credal set <uq-credal>` with the member
    predictions as vertices.
:Advantages: Preserves the dependence between classes, since the hull
    contains only mixtures of actual member predictions.
:Disadvantages: Requires an ensemble; reasoning over a set of vertices is more
    costly than over intervals.
:Reference: :cite:`nguyenCredalEnsembling2025`

.. minigallery:: probly.method.credal_ensembling

.. _m-credal-bnn:

:func:`credal_bnn <probly.method.credal_bnn>`
---------------------------------------------

The same idea with Bayesian members. Instead of a single variational
posterior, an ensemble of Bayesian neural networks is trained, and their
predictive distributions form the vertices of a convex credal set. This
addresses a standard objection to a single BNN, namely that the prior is
itself a choice, and one that is rarely well justified; a credal set can
accommodate several priors without having to commit to one of them.

:Idea: An ensemble of Bayesian neural networks whose predictive distributions
    are the vertices.
:Representation: A :ref:`convex credal set <uq-credal>` with Bayesian members
    as vertices.
:Advantages: Accommodates several prior specifications without pretending
    that one of them is correct.
:Disadvantages: Requires training several BNNs, each with its own variational
    scheme.
:Reference: :cite:`caprioCredalBayesian2024`

.. minigallery:: probly.method.credal_bnn

.. _m-credal-net:

:func:`credal_net <probly.method.credal_net.credal_net>`
--------------------------------------------------------

Instead of ensembling models, this approach makes the network itself
interval-valued: every linear, convolutional, and batch-normalization layer is
replaced by an interval-arithmetic counterpart, and the credal set results
from propagating these intervals to the output. A single network and a single
forward pass suffice, and there are no members to train. The price is paid in
tightness, since the bounds produced by interval arithmetic tend to widen with
depth.

:Idea: Replace the layers by interval-arithmetic counterparts and propagate
    the weight intervals to the output.
:Representation: A credal set given by
    :ref:`probability intervals <uq-credal>`, obtained from a single network.
:Advantages: One network, one forward pass, and no members to train.
:Disadvantages: Interval arithmetic tends to widen with depth, so deep
    networks may yield loose bounds.
:Reference: :cite:`wangCredalDeepEnsembles2024`

.. minigallery:: probly.method.credal_net

.. _m-credal-relative-likelihood:

:func:`credal_relative_likelihood <probly.method.credal_relative_likelihood>`
-----------------------------------------------------------------------------

Instead of relying on random initialization for diversity, this method
constructs its members deliberately: one member per class, each initialized
with a bias towards that class. Roughly speaking, each member answers the
question of how well the data can be explained when leaning towards class
``k``, and only explanations whose likelihood is sufficiently close to that of
the best one, i.e., whose relative likelihood is high enough, delimit the
credal set. The result is a systematic and reproducible cover of the simplex.

:Idea: One class-biased member per class; the relative likelihood of their
    explanations of the data delimits the set.
:Representation: A credal set given by
    :ref:`probability intervals <uq-credal>` spanned by the class-biased
    members.
:Advantages: A systematic, reproducible cover of the simplex that does not
    rely on random initialization for diversity.
:Disadvantages: Trains one member per class, so the cost grows with the number
    of classes.
:Reference: :cite:`lohrCredalPrediction2025`

.. minigallery:: probly.method.credal_relative_likelihood

.. _m-efficient-credal-prediction:

:func:`efficient_credal_prediction <probly.method.efficient_credal_prediction>`
-------------------------------------------------------------------------------

Unlike the methods above, this one leaves the base classifier untouched. The
classifier keeps returning its ordinary :ref:`categorical distribution
<uq-first-order>`, and the credal set is constructed on demand by shifting each
logit within offsets calibrated on a held-out split. There is no ensemble, no
interval layer, and no retraining: ``predict`` returns the point distribution,
and the representer returns the credal set.

:Idea: Keep the base classifier's categorical output and construct the credal
    set on demand from calibrated logit offsets.
:Representation: A credal set given by
    :ref:`probability intervals <uq-credal>`, derived from a
    :ref:`first-order distribution <uq-first-order>` and held-out bounds.
:Advantages: No ensemble, no interval layers, and no retraining; ``predict``
    still returns the point distribution.
:Disadvantages: Requires a held-out calibration split, and the credal set is
    only as good as the calibrated bounds.
:Reference: :cite:`hofmanEfficientCredal2026`

.. minigallery:: probly.method.efficient_credal_prediction

.. _m-conformal-credal-set:

Conformal Credal Sets
---------------------

These methods connect the present family with :ref:`methods-conformal`.
Rather than deriving the credal set from disagreement within the model, they
place a ball around the predicted distribution and *calibrate* its radius on
held-out data, so that the resulting credal set inherits the coverage
guarantee of conformal prediction. The five variants differ only in how the
ball is defined: by the total variation distance, the Kullback-Leibler
divergence, the Wasserstein distance, an inner product, or a level set of a
Dirichlet relative likelihood. This choice determines the shape of the set on
the simplex, and hence which distributions near its boundary are included.

:Idea: Place a ball around the predicted distribution and calibrate its radius
    on held-out data.
:Representation: A credal set carrying a conformal coverage guarantee.
:Advantages: Inherits the finite-sample coverage guarantee of conformal
    prediction; the choice of distance controls the shape of the set.
:Disadvantages: Requires a held-out calibration split; as for all conformal
    methods, coverage is marginal rather than conditional.
:Reference: :cite:`saleSecondOrder2024`, :cite:`angelopoulosGentleIntroduction2021`

.. autosummary::
    :nosignatures:

    ~probly.transformation.conformal_credal_set.conformal_total_variation
    ~probly.transformation.conformal_credal_set.conformal_kullback_leibler
    ~probly.transformation.conformal_credal_set.conformal_wasserstein_distance
    ~probly.transformation.conformal_credal_set.conformal_inner_product
    ~probly.transformation.conformal_credal_set.conformal_dirichlet_relative_likelihood

.. minigallery:: probly.transformation.conformal_total_variation probly.transformation.conformal_kullback_leibler probly.transformation.conformal_wasserstein_distance probly.transformation.conformal_inner_product probly.transformation.conformal_dirichlet_relative_likelihood

.. admonition:: Categorical score targets
    :class: note

    The inner-product, KL-divergence, total-variation, and Wasserstein
    conformal scores interpret targets consistently across NumPy, Torch, and
    JAX:

    * Integer arrays contain class indices, interpreted as point-mass targets.
    * Floating-point arrays contain probability vectors with a trailing class
      axis.
    * ``CategoricalDistribution`` targets supply their normalized
      probabilities, regardless of their storage dtype or whether they store
      logits.
    * Boolean and complex target arrays are rejected.

    Types and dtypes determine the interpretation. Shapes only validate the
    class count and establish ordinary batch broadcasting. For example,
    predictions of shape ``(members, batch, classes)`` accept floating targets
    of shape ``(batch, classes)`` or integer labels of shape ``(batch,)``.
    Singleton batch dimensions are preserved according to normal broadcasting
    rules.

    Integer one-hot arrays must be cast to floating point or wrapped in a
    categorical distribution to be interpreted as probability vectors.
    Conversely, class labels stored as floats must be explicitly converted to
    an integer dtype. There is no special reshaping of row-vector labels and no
    rank-based inference of target semantics.

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
