.. _methods-second-order:

==========================
Second-Order Distributions
==========================

.. currentmodule:: probly.method

These methods turn a point predictor into one that outputs a
:ref:`distribution over distributions <uq-second-order>`. The extra order is
what makes the :ref:`aleatoric/epistemic split <uq-decomposition>` computable:
the spread *between* the predicted distributions is the epistemic part, the
average spread *within* them is the aleatoric part.

They differ along one axis above all others, and it is worth naming before the
catalogue starts:

*Sampled* methods represent the second-order distribution implicitly, by a
finite set of first-order distributions you have to draw --- ``T`` forward
passes, ``N`` ensemble members, ``S`` posterior weight samples. You pay for the
resolution of the representation at every prediction.

*Parameterized* methods emit the second-order distribution in closed form,
typically as a Dirichlet over the simplex, from a single forward pass. They are
cheap at inference and expensive at training: the architecture, the loss, or
both have to change.

A third, smaller group is *distance-aware*: instead of a posterior over
parameters they measure where a point falls relative to the training data in
feature space. They produce excellent out-of-distribution scores and, by
construction, say little about aleatoric uncertainty.

Every entry below follows the same four fields, so they can be read against
each other.

.. _m-dropout:

:func:`dropout <probly.method.dropout>`
---------------------------------------

Keeps dropout layers active at inference and treats each stochastic forward
pass as one draw from an approximate posterior. It is the cheapest possible
retrofit: if the trained network already contains dropout, nothing needs to be
retrained at all. The transform inserts dropout after linear and convolutional
layers; the torch backend additionally offers ``shared_mask`` to draw one mask
per forward pass instead of one per element.

:Idea: Monte Carlo sampling of sub-networks by keeping dropout on at test time.
:Representation: :ref:`Sampled second order <uq-second-order>` --- a ``Sample``
    of categorical distributions.
:Cost: No retraining. ``T`` forward passes per prediction.
:Reference: :cite:`galDropoutBayesian2016`

.. _m-dropconnect:

:func:`dropconnect <probly.method.dropconnect>`
-----------------------------------------------

The same Monte Carlo argument applied one level down: DropConnect zeroes
individual *weights* rather than whole activations, so each forward pass
samples a different sparse weight matrix. The sub-network space is larger than
dropout's, which tends to give more diverse samples at the same drop rate.

:Idea: Monte Carlo sampling over randomly dropped weights.
:Representation: :ref:`Sampled second order <uq-second-order>`.
:Cost: No retraining. ``T`` forward passes per prediction.
:Reference: :cite:`mobinyDropConnectEffective2021`

.. _m-bayesian:

:func:`bayesian <probly.method.bayesian>`
-----------------------------------------

A genuine variational Bayesian neural network: every weight becomes a Gaussian
with a learned mean and standard deviation, fitted by Bayes-by-Backprop against
a prior. Unlike dropout, the posterior is explicit and trained for, which is
also why it cannot be bolted onto a finished model.

:Idea: Mean-field Gaussian variational posterior over the weights.
:Representation: :ref:`Sampled second order <uq-second-order>`, drawn by
    sampling weights.
:Cost: Requires training with the ELBO; roughly doubles the parameter count.
    ``S`` forward passes per prediction.
:Reference: :cite:`blundellWeightUncertainty2015`

.. _m-laplace:

:mod:`laplace <probly.method.laplace>`
--------------------------------------

A post-hoc Bayesian treatment: fit the network as usual, then place a Gaussian
around the trained MAP weights whose covariance is the inverse Hessian of the
loss. ``probly`` integrates the ``laplace-torch`` package and exposes its
``predictive_samples`` through the standard representer interface
(classification only).

:Idea: Second-order Taylor expansion of the loss around the MAP estimate.
:Representation: :ref:`Sampled second order <uq-second-order>`.
:Cost: No change to training. One Hessian approximation after fitting, then
    ``S`` forward passes per prediction.
:Reference: Wraps the external ``laplace-torch`` package.

.. _m-ensemble:

:func:`ensemble <probly.method.ensemble>`
-----------------------------------------

Train ``N`` copies of the same architecture from different initializations and
treat their predictions as samples. Deep ensembles remain the strongest and
most robust baseline in the literature, and the reason is diversity of loss
basins rather than any Bayesian argument.

:Idea: Independent retraining; disagreement between members is the epistemic
    signal.
:Representation: :ref:`Sampled second order <uq-second-order>` with ``N``
    members.
:Cost: ``N`` times the training cost, ``N`` times the memory, ``N`` forward
    passes.
:Reference: :cite:`lakshminarayananSimpleScalable2017`

.. _m-batchensemble:

:func:`batchensemble <probly.method.batchensemble>`
---------------------------------------------------

An ensemble that fits in roughly one model's memory. Each member owns only a
rank-one factor pair ``(r, s)`` that modulates a shared "slow" weight matrix,
so ``N`` members cost ``N`` extra vectors per layer rather than ``N`` extra
matrices. Members are still trained jointly, in one pass over tiled inputs.

:Idea: Rank-one per-member perturbations of a shared weight matrix.
:Representation: :ref:`Sampled second order <uq-second-order>` with ``N``
    members.
:Cost: Close to one model in memory; one training run; inputs are tiled ``N``
    times per batch.
:Reference: :cite:`wenBatchEnsemble2020`

.. _m-subensemble:

:func:`subensemble <probly.method.subensemble>`
-----------------------------------------------

Shares the expensive backbone and ensembles only the head. The split is either
taken from an existing model (the last ``head_layer`` layers become the head)
or supplied explicitly. Diversity is lower than a full ensemble, since all
members see the same features, but the cost is close to a single model.

:Idea: One shared feature extractor, ``N`` independently initialized heads.
:Representation: :ref:`Sampled second order <uq-second-order>` with ``N``
    heads.
:Cost: One backbone plus ``N`` heads; one training run.
:Reference: :cite:`valdenegro-toroDeepSub2019`

.. _m-dare:

:func:`dare <probly.method.dare>`
---------------------------------

An ensemble whose members are actively pushed apart. Anti-regularization
rewards weight diversity instead of penalizing weight magnitude, which widens
the ensemble's spread away from the training data --- where a conventional
ensemble tends to collapse into agreement --- and so improves epistemic
estimates under shift.

:Idea: Deep anti-regularized ensembling: increase member diversity by
    regularizing *against* weight shrinkage.
:Representation: :ref:`Sampled second order <uq-second-order>` with ``N``
    members.
:Cost: ``N`` times the training cost, plus the anti-regularization term.
:Reference: :cite:`demathelinDeepAntiRegularized2023`

.. _m-duq:

:func:`duq <probly.method.duq>`
-------------------------------

Replaces the linear classification head with a set of learned per-class
centroids in feature space and scores a point by its RBF distance to them. A
single deterministic forward pass yields both the prediction and the
uncertainty; the centroids are updated by an exponential moving average during
training, and a gradient penalty keeps the feature map from collapsing.

:Idea: Distance to learned class centroids in feature space, in place of a
    softmax head.
:Representation: RBF kernel scores, presented as a
    :ref:`categorical distribution <uq-first-order>`; total uncertainty only.
:Cost: One deterministic forward pass. Requires training with a gradient
    penalty and a changed head.
:Reference: :cite:`vanAmersfoortUncertaintyEstimation2020`

.. _m-ddu:

:func:`ddu <probly.method.ddu>`
-------------------------------

Also distance-aware, but it separates the two jobs: spectral normalization
keeps the feature extractor sensitive to input changes (so distances in feature
space remain meaningful), and a Gaussian mixture density fitted post-hoc on
those features supplies the epistemic score. The softmax entropy is kept for
aleatoric uncertainty.

:Idea: Spectral normalization for a well-behaved feature space plus a Gaussian
    density estimate over it.
:Representation: Feature-space density as the epistemic score, softmax as the
    :ref:`first-order <uq-first-order>` predictive.
:Cost: One forward pass. Spectral normalization during training; the density
    head is fitted afterwards.
:Reference: :cite:`mukhotiDeepDeterministicUncertainty2023`

.. _m-deup:

:func:`deup <probly.method.deup.deup>`
--------------------------------------

Rather than deriving uncertainty from a posterior, DEUP *predicts the model's
own error*. A second head is trained on held-out data to regress the frozen
main model's per-sample loss; at inference that predicted loss is the epistemic
score. Training is explicitly two-phase --- first the classifier, then the
error head on data the classifier has not seen.

:Idea: Learn a direct regressor of the main model's generalization error.
:Representation: A scalar error score; the decomposition assigns it entirely to
    the epistemic term.
:Cost: One extra head and a second training phase on a held-out split. One
    forward pass at inference.
:Reference: :cite:`lahlouDirectEpistemic2023`

.. _m-sngp:

:func:`sngp <probly.method.sngp>`
---------------------------------

Combines the two ingredients that make a single deterministic network
distance-aware: spectral normalization on the hidden layers to make the
representation approximately distance-preserving, and a Gaussian process output
layer (random Fourier features with a Laplace-approximated posterior) that
widens away from the training data.

:Idea: Spectral-normalized backbone with a GP output layer approximated by
    random Fourier features.
:Representation: Gaussian over logits, giving a
    :ref:`second-order <uq-second-order>` predictive in closed form.
:Cost: One forward pass. Changed head, spectral normalization, and a covariance
    update pass at the end of training.
:Reference: :cite:`liuSimplePrincipled2020`

.. _m-evidential-classification:

:func:`evidential_classification <probly.method.evidential_classification>`
---------------------------------------------------------------------------

The first of the parameterized methods: the network outputs the concentration
parameters of a Dirichlet directly, interpreting them as accumulated
*evidence* for each class. The whole second-order distribution comes out of one
forward pass, and total evidence (rather than spread across samples) carries
the epistemic signal --- low evidence means "I have seen nothing like this".

:Idea: Predict Dirichlet concentrations as per-class evidence.
:Representation: :ref:`Parameterized second order <uq-second-order>` ---
    a ``DirichletDistribution``.
:Cost: One forward pass. Needs an evidential loss and an activation that keeps
    concentrations positive.
:Reference: :cite:`sensoyEvidentialDeep2018`

.. _m-posterior-network:

:func:`posterior_network <probly.method.posterior_network.posterior_network>`
-----------------------------------------------------------------------------

Fixes the main weakness of purely evidential losses --- that nothing forces
evidence to fall off away from the data --- by making the concentration
parameters proportional to an estimated *density*. A normalizing flow over a
low-dimensional latent space provides that density, so regions with little
training data receive little evidence by construction.

:Idea: Density-based Dirichlet concentrations via a normalizing flow in latent
    space.
:Representation: :ref:`Parameterized second order <uq-second-order>` ---
    a ``DirichletDistribution``.
:Cost: One forward pass. Requires an encoder, per-class normalizing flows, and
    a Bayesian-loss training scheme.
:Reference: :cite:`charpentierPosteriorNetwork2020`

.. _m-mahalanobis:

:func:`mahalanobis <probly.method.mahalanobis>`
-----------------------------------------------

A post-hoc out-of-distribution detector rather than a predictive-uncertainty
method. Class-conditional Gaussians with a tied covariance are fitted to the
penultimate (and optionally intermediate) features of a trained network; the
Mahalanobis distance to the nearest class mean is the score, optionally
sharpened by a small FGSM-style input perturbation and combined across layers
by logistic regression.

:Idea: Class-conditional Gaussians in feature space; distance to the nearest
    one is the OOD score.
:Representation: A scalar OOD score alongside the base
    :ref:`first-order <uq-first-order>` prediction.
:Cost: No retraining. One fitting pass over the training features; the
    multi-layer combiner needs in- and out-of-distribution data.
:Reference: :cite:`leeSimpleUnifiedFramework2018`

.. _m-natural-posterior-network:

:func:`natural_posterior_network <probly.method.natural_posterior_network.natural_posterior_network>`
-----------------------------------------------------------------------------------------------------

The generalization of :ref:`posterior_network <m-posterior-network>` to
exponential-family targets, and the version to prefer in practice: a *single*
shared flow over the latent space provides ``log p(z)``, a linear classifier
provides the class log-probabilities, and the Dirichlet parameters follow the
Bayesian update ``alpha = alpha_prior + n(x) * chi(x)``, with ``n(x)`` a
budget-scaled pseudo-count. Sharing one flow is what makes it scale to many
classes.

:Idea: A Bayesian posterior update per input, with a density-derived
    pseudo-count as the evidence.
:Representation: :ref:`Parameterized second order <uq-second-order>` ---
    a ``DirichletDistribution``.
:Cost: One forward pass. Encoder plus one shared normalizing flow; changed
    loss.
:Reference: :cite:`charpentierNaturalPosteriorNetwork2022`

.. _m-prior-network:

:func:`prior_network <probly.method.prior_network.prior_network>`
-----------------------------------------------------------------

The simplest way to get a Dirichlet out of an existing classifier: exponentiate
the logits and read them as concentration parameters. In its original form the
network is trained to emit a flat Dirichlet on out-of-distribution data and a
sharp one in-distribution, which requires OOD examples during training; without
them it is best read as a reparameterization of the logits.

:Idea: Exponential activation on the logits turns them into Dirichlet
    concentrations.
:Representation: :ref:`Parameterized second order <uq-second-order>` ---
    a ``DirichletDistribution``.
:Cost: One forward pass, no architectural change. The intended training scheme
    needs out-of-distribution data.
:Reference: :cite:`malininPredictiveUncertaintyEstimation2018`

.. _m-evidential-regression:

:func:`evidential_regression <probly.method.evidential_regression>`
-------------------------------------------------------------------

The regression counterpart of evidential classification. The final linear layer
is replaced by a head emitting the four parameters of a Normal-Inverse-Gamma
distribution, which is a distribution over the mean *and* the variance of the
target. One forward pass therefore yields both an aleatoric estimate (the
expected variance) and an epistemic one (the variance of the mean).

:Idea: Predict a Normal-Inverse-Gamma prior over the target's mean and
    variance.
:Representation: :ref:`Parameterized second order <uq-second-order>` over a
    real-valued target.
:Cost: One forward pass. Changed head and an evidential regression loss.
:Reference: :cite:`aminiDeepEvidential2020`

.. _m-het-net:

:func:`het_net <probly.method.het_net>`
---------------------------------------

Aleatoric-only, and included here because it is the honest way to model label
noise: instead of a single logit vector the network predicts a full
input-dependent covariance over logits, parameterized in low rank so it stays
affordable. Useful where labels genuinely disagree; it says nothing about
whether the model has seen data like this before.

:Idea: Input-dependent, low-rank correlated noise over the logits.
:Representation: A Gaussian over logits; aleatoric uncertainty only.
:Cost: One forward pass plus ``num_factors`` extra output channels; sampled
    softmax during training.
:Reference: :cite:`collierCorrelatedInputDependent2021`

Choosing Among Them
-------------------

If the model is already trained and you cannot retrain it, the options are
:ref:`dropout <m-dropout>`, :ref:`laplace <m-laplace>` and
:ref:`mahalanobis <m-mahalanobis>`. If you can afford ``N`` training runs,
:ref:`ensemble <m-ensemble>` is the baseline everything else is measured
against, and :ref:`batchensemble <m-batchensemble>` or
:ref:`subensemble <m-subensemble>` recover most of it for a fraction of the
memory. If inference latency is the binding constraint, use a parameterized
method --- :ref:`natural_posterior_network <m-natural-posterior-network>` for
classification, :ref:`evidential_regression <m-evidential-regression>` for
regression. If the target is out-of-distribution detection specifically, prefer
the distance-aware group: :ref:`ddu <m-ddu>`, :ref:`sngp <m-sngp>`,
:ref:`duq <m-duq>`.

Whichever you pick, :ref:`uq-evaluating` is what decides whether it worked.

Full API
--------

.. autosummary::
    :nosignatures:

    dropout
    dropconnect
    bayesian
    laplace
    ensemble
    batchensemble
    subensemble
    dare
    duq
    ddu
    ~deup.deup
    sngp
    evidential_classification
    ~posterior_network.posterior_network
    mahalanobis
    ~natural_posterior_network.natural_posterior_network
    ~prior_network.prior_network
    evidential_regression
    het_net
