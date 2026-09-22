.. _methods-second-order:

==========================
Second-Order Distributions
==========================

.. currentmodule:: probly.method

The methods on this page turn a point predictor into one that outputs a
:ref:`distribution over distributions <uq-second-order>`. This additional
order is what makes the :ref:`aleatoric/epistemic decomposition
<uq-decomposition>` computable, at least in its most common form: the spread
*between* the predicted first-order distributions is attributed to epistemic
uncertainty, and the average spread *within* them to aleatoric uncertainty.

The methods differ in many respects, but one distinction cuts deeper than the
others, since it determines where the computational cost is paid. *Sampled*
methods represent the second-order distribution implicitly, by a finite set of
first-order distributions obtained from ``T`` stochastic forward passes, ``N``
ensemble members, or ``S`` posterior weight samples. The resolution of this
representation is paid for at every prediction.

*Parameterized* methods, on the other hand, output the second-order
distribution in closed form, typically as a Dirichlet distribution over the
simplex, from a single forward pass. The cost does not disappear, however; it
moves to training, where the architecture, the loss, or both have to change.

A third and smaller group is *distance-aware*. Instead of placing a posterior
over parameters, these methods measure where an input falls relative to the
training data in feature space. They tend to separate in-distribution from
out-of-distribution inputs well, but the distance itself carries no
information about aleatoric uncertainty, which has to come from elsewhere, if
at all. A few methods fall outside all three groups, most notably
:ref:`deup <m-deup>`, which learns to predict the model's error, and
:ref:`het_net <m-het-net>`, which models aleatoric uncertainty only.

.. _m-dropout:

:func:`dropout <probly.transformation.dropout>`
-----------------------------------------------

Dropout layers are kept active at inference, and each stochastic forward pass
is treated as a draw from an approximate posterior. For a network that was
trained with dropout, no retraining is required at all, which makes this the
cheapest retrofit available. The transform inserts a dropout layer in front of
every linear layer; the torch backend additionally offers ``shared_mask``,
which draws a single mask per forward pass, shared across the batch, instead
of an independent mask per sample.

:Idea: Monte Carlo sampling of sub-networks by keeping dropout active at test
    time.
:Representation: :ref:`Sampled second order <uq-second-order>`, i.e., a
    ``Sample`` of categorical distributions.
:Advantages: No retraining if the model was trained with dropout; the
    cheapest retrofit available.
:Disadvantages: ``T`` forward passes per prediction; the quality depends on
    where the dropout layers sit, and a model trained without dropout may have
    to be retrained once they are inserted.
:Reference: :cite:`galDropoutBayesian2016`

.. minigallery:: probly.transformation.dropout

.. _m-dropconnect:

:func:`dropconnect <probly.transformation.dropconnect>`
-------------------------------------------------------

The same Monte Carlo argument, applied one level lower: DropConnect sets
individual *weights* rather than entire activations to zero, so that every
forward pass samples a different sparse weight matrix. The space of
sub-networks is thus larger than for dropout, which tends to yield more diverse
samples at the same drop rate.

:Idea: Monte Carlo sampling over randomly dropped weights.
:Representation: :ref:`Sampled second order <uq-second-order>`.
:Advantages: A larger sub-network space than dropout, and hence more diverse
    samples at the same drop rate; as with dropout, retraining is not strictly
    required.
:Disadvantages: ``T`` forward passes per prediction; masking individual
    weights costs more per pass than masking activations.
:Reference: :cite:`mobinyDropConnectEffective2021`

.. minigallery:: probly.transformation.dropconnect

.. _m-bayesian:

:func:`bayesian <probly.transformation.bayesian>`
-------------------------------------------------

A genuine variational Bayesian neural network: every weight is replaced by a
Gaussian with learned mean and standard deviation, fitted by Bayes by Backprop
against a prior. Unlike with dropout, the posterior is explicit and is itself
the object of training, which is also why it cannot simply be added to a model
that has already been trained.

:Idea: Mean-field Gaussian variational posterior over the weights.
:Representation: :ref:`Sampled second order <uq-second-order>`, obtained by
    sampling weights.
:Advantages: An explicit posterior that is trained for, rather than an
    approximation added afterwards.
:Disadvantages: Requires training with the ELBO and roughly doubles the
    parameter count; cannot be applied to a trained model; ``S`` forward
    passes per prediction.
:Reference: :cite:`blundellWeightUncertainty2015`

.. minigallery:: probly.transformation.bayesian

.. _m-laplace:

:mod:`laplace <probly.method.laplace>`
--------------------------------------

A post-hoc Bayesian treatment: the network is trained as usual, and a Gaussian
is then placed around the resulting MAP weights, with the inverse Hessian of
the loss, or in practice an approximation of it, as its covariance. ``probly``
integrates the ``laplace-torch`` package and exposes its
``predictive_samples`` through the standard representer interface; only
classification is currently supported.

:Idea: Second-order Taylor expansion of the loss around the MAP estimate.
:Representation: :ref:`Sampled second order <uq-second-order>`.
:Advantages: No change to training; a post-hoc treatment of an already
    trained network.
:Disadvantages: A Hessian approximation after training, then ``S`` forward
    passes per prediction; classification only.
:Reference: Wraps the external ``laplace-torch`` package.

.. minigallery:: probly.method.laplace

.. _m-ensemble:

:func:`ensemble <probly.transformation.ensemble>`
-------------------------------------------------

``N`` copies of the same architecture are trained from different
initializations, and their predictions are treated as samples. Deep ensembles
have proven hard to beat, and what explains their success appears to be the
diversity of the loss basins in which the members settle rather than any
Bayesian argument.

:Idea: Independent retraining; the disagreement between members is the
    epistemic signal.
:Representation: :ref:`Sampled second order <uq-second-order>` with ``N``
    members.
:Advantages: Consistently among the strongest baselines, without relying on
    Bayesian assumptions.
:Disadvantages: ``N`` times the training cost, ``N`` times the memory, and
    ``N`` forward passes per prediction.
:Reference: :cite:`lakshminarayananSimpleScalable2017`

.. minigallery:: probly.method.ensemble
.. minigallery:: probly.transformation.ensemble

.. _m-batchensemble:

:func:`batchensemble <probly.transformation.batchensemble>`
-----------------------------------------------------------

An ensemble that fits into roughly the memory of a single model. Each member
owns only a pair of rank-one factors ``(r, s)`` that modulate a shared "slow"
weight matrix, so ``N`` members cost ``N`` additional pairs of vectors per
layer rather than ``N`` additional matrices. The members are trained jointly,
in a single pass over inputs that are tiled ``N`` times.

:Idea: Rank-one, per-member perturbations of a shared weight matrix.
:Representation: :ref:`Sampled second order <uq-second-order>` with ``N``
    members.
:Advantages: Close to a single model in memory, with a single training run,
    yet recovers much of the benefit of a full ensemble.
:Disadvantages: Inputs are tiled ``N`` times per batch, and diversity is lower
    than for fully independent members.
:Reference: :cite:`wenBatchEnsemble2020`

.. minigallery:: probly.transformation.batchensemble

.. _m-subensemble:

:func:`subensemble <probly.transformation.subensemble>`
-------------------------------------------------------

The expensive backbone is shared, and only the head is ensembled. The split is
either taken from an existing model, in which case the last ``head_layer``
layers form the head, or specified explicitly. Since all members see the same
features, diversity is lower than for a full ensemble, but the cost is close
to that of a single model.

:Idea: One shared feature extractor and ``N`` independently initialized heads.
:Representation: :ref:`Sampled second order <uq-second-order>` with ``N``
    heads.
:Advantages: Close to a single model in cost: one backbone and one training
    run.
:Disadvantages: Lower diversity than a full ensemble, since all members share
    the same features.
:Reference: :cite:`valdenegro-toroDeepSub2019`

.. minigallery:: probly.transformation.subensemble

.. _m-dare:

:func:`dare <probly.method.dare>`
---------------------------------

An ensemble whose members are actively pushed apart. The anti-regularization
term rewards large weights instead of penalizing them, but is only active as
long as the training loss stays below a threshold, so that the fit to the
training data is maintained. Members with large weights tend to disagree away
from the training data, which is precisely where conventional ensembles tend
to collapse into agreement; as a result, the epistemic estimates under
distribution shift improve.

:Idea: Deep anti-regularized ensembles: increase member diversity by
    rewarding, rather than penalizing, large weights.
:Representation: :ref:`Sampled second order <uq-second-order>` with ``N``
    members.
:Advantages: Wider and more informative epistemic estimates under
    distribution shift, where ordinary ensembles tend to collapse into
    agreement.
:Disadvantages: ``N`` times the training cost, plus the anti-regularization
    term and the loss threshold it requires.
:Reference: :cite:`demathelinDeepAntiRegularized2023`

.. minigallery:: probly.method.dare

.. _m-duq:

:func:`duq <probly.method.duq>`
-------------------------------

Replaces the linear classification head by a set of learned per-class
centroids in feature space and scores an input by its RBF-kernel distance to
them. A single deterministic forward pass thus yields both the prediction and
the uncertainty. During training, the centroids are updated by an exponential
moving average, and a gradient penalty prevents feature collapse, i.e.,
out-of-distribution inputs being mapped close to the centroids.

:Idea: Distance to learned class centroids in feature space, in place of a
    softmax head.
:Representation: RBF kernel scores, presented as a
    :ref:`categorical distribution <uq-first-order>`; total uncertainty only.
:Advantages: A single deterministic forward pass yields both the prediction
    and the uncertainty.
:Disadvantages: Requires training with a gradient penalty and a modified
    head; does not separate aleatoric from epistemic uncertainty.
:Reference: :cite:`vanAmersfoortUncertaintyEstimation2020`

.. minigallery:: probly.method.duq

.. _m-ddu:

:func:`ddu <probly.method.ddu>`
-------------------------------

Also distance-aware, but with a division of labor: spectral normalization
keeps the feature extractor sensitive to changes in the input, so that
distances in feature space remain meaningful, and a Gaussian mixture density
fitted post-hoc on these features provides the epistemic score. The softmax
entropy is retained as a measure of aleatoric uncertainty.

:Idea: Spectral normalization for a well-behaved feature space, plus a
    Gaussian density estimate over it.
:Representation: The feature-space density as epistemic score, and the
    softmax as :ref:`first-order <uq-first-order>` predictive distribution.
:Advantages: One forward pass, with the density fitted after training; strong
    out-of-distribution detection.
:Disadvantages: Spectral normalization must be applied during training, so the
    method is not a pure retrofit.
:Reference: :cite:`mukhotiDeepDeterministicUncertainty2023`

.. minigallery:: probly.method.ddu

.. _m-deup:

:func:`deup <probly.method.deup.deup>`
--------------------------------------

Rather than deriving uncertainty from a posterior, DEUP *predicts the model's
own error*. A second head is trained on held-out data to regress the
per-sample loss of the frozen main model, and at inference time this predicted
loss serves as the epistemic score. Training therefore proceeds in two phases:
first the classifier, then the error head, on data the classifier has not
seen.

:Idea: Learn a direct regressor of the main model's generalization error.
:Representation: A scalar error score, which the decomposition assigns
    entirely to the epistemic term.
:Advantages: One forward pass at inference, and no assumptions about a
    posterior.
:Disadvantages: Requires an additional head and a second training phase on a
    held-out split.
:Reference: :cite:`lahlouDirectEpistemic2023`

.. minigallery:: probly.method.deup.deup

.. _m-sngp:

:func:`sngp <probly.method.sngp>`
---------------------------------

Combines the two ingredients that make a single deterministic network
distance-aware: spectral normalization of the hidden layers, which makes the
representation approximately distance-preserving, and a Gaussian process
output layer, approximated by random Fourier features with a
Laplace-approximated posterior, whose uncertainty grows away from the training
data.

:Idea: Spectrally normalized backbone with a Gaussian process output layer
    approximated by random Fourier features.
:Representation: A Gaussian over the logits, which yields a
    :ref:`second-order <uq-second-order>` predictive distribution in closed
    form.
:Advantages: Distance-aware uncertainty with a closed-form predictive
    distribution from a single forward pass.
:Disadvantages: A modified head, spectral normalization, and an additional
    covariance update pass at the end of training.
:Reference: :cite:`liuSimplePrincipled2020`

.. minigallery:: probly.method.sngp

.. _m-evidential-classification:

:func:`evidential_classification <probly.method.evidential_classification>`
---------------------------------------------------------------------------

Here, the second order is carried by evidence rather than by sampling: the
network directly outputs the concentration parameters of a Dirichlet
distribution, which are interpreted as accumulated *evidence* for each class.
The entire second-order distribution thus results from a single forward pass,
and the epistemic signal lies in the total amount of evidence rather than in
the spread across samples; low total evidence means, roughly, "I have not
seen anything like this". Note, however, that nothing in the training
objective forces the evidence to decrease away from the training data.

:Idea: Predict Dirichlet concentrations as per-class evidence.
:Representation: :ref:`Parameterized second order <uq-second-order>`, i.e., a
    ``DirichletDistribution``.
:Advantages: The entire second-order distribution from a single forward pass;
    low total evidence flags unfamiliar inputs.
:Disadvantages: Requires an evidential loss and a positive activation; nothing
    forces the evidence to decrease away from the data.
:Reference: :cite:`sensoyEvidentialDeep2018`

.. minigallery:: probly.method.evidential_classification

.. _m-posterior-network:

:func:`posterior_network <probly.transformation.posterior_network>`
-------------------------------------------------------------------

Addresses the main weakness of purely evidential losses, namely that nothing
forces the evidence to decrease away from the data, by making the
concentration parameters proportional to an estimated *density*. This density
is provided by a normalizing flow over a low-dimensional latent space, so that
regions with little training data receive little evidence by construction.

:Idea: Density-based Dirichlet concentrations via a normalizing flow in latent
    space.
:Representation: :ref:`Parameterized second order <uq-second-order>`, i.e., a
    ``DirichletDistribution``.
:Advantages: Evidence decreases by construction where training data is
    sparse, which removes the main weakness of evidential losses; one forward
    pass.
:Disadvantages: Requires an encoder, one normalizing flow per class, and a
    Bayesian training loss; the per-class flows scale poorly in the number of
    classes.
:Reference: :cite:`charpentierPosteriorNetwork2020`

.. minigallery:: probly.transformation.posterior_network

.. _m-mahalanobis:

:func:`mahalanobis <probly.method.mahalanobis>`
-----------------------------------------------

Strictly speaking, a post-hoc out-of-distribution detector rather than a
method for predictive uncertainty. Class-conditional Gaussians with a shared
covariance matrix are fitted to the penultimate, and optionally intermediate,
features of a trained network. The score is the Mahalanobis distance to the
nearest class mean, optionally sharpened by a small FGSM-style perturbation of
the input and combined across layers by logistic regression.

:Idea: Class-conditional Gaussians in feature space; the distance to the
    nearest one is the out-of-distribution score.
:Representation: A scalar out-of-distribution score alongside the base
    :ref:`first-order <uq-first-order>` prediction.
:Advantages: No retraining; a single fitting pass over the training features.
:Disadvantages: An out-of-distribution detector rather than a
    predictive-uncertainty method; the multi-layer combination requires both
    in- and out-of-distribution data.
:Reference: :cite:`leeSimpleUnifiedFramework2018`

.. minigallery:: probly.method.mahalanobis

.. _m-natural-posterior-network:

:func:`natural_posterior_network <probly.transformation.natural_posterior_network.natural_posterior_network>`
--------------------------------------------------------------------------------------------------------------

Generalizes :ref:`posterior_network <m-posterior-network>` to targets from the
exponential family and is the natural first choice among the density-based
methods. A *single* shared flow over the latent space provides ``log p(z)``, a
linear classifier provides the class log-probabilities, and the Dirichlet
parameters follow the Bayesian update ``alpha = alpha_prior + n(x) * chi(x)``,
where ``n(x)`` is a pseudo-count derived from the density and scaled by a
certainty budget. Sharing one flow across classes is what allows the method to
scale to many classes.

:Idea: A Bayesian posterior update per input, with a density-derived
    pseudo-count as the evidence.
:Representation: :ref:`Parameterized second order <uq-second-order>`, i.e., a
    ``DirichletDistribution``.
:Advantages: A single shared flow scales to many classes, unlike the per-class
    flows it replaces; one forward pass.
:Disadvantages: Requires an encoder, a shared normalizing flow, and a modified
    loss.
:Reference: :cite:`charpentierNaturalPosteriorNetwork2022`

.. minigallery:: probly.method.natural_posterior_network.natural_posterior_network

.. _m-prior-network:

:func:`prior_network <probly.method.prior_network.prior_network>`
-----------------------------------------------------------------

The simplest way of obtaining a Dirichlet distribution from an existing
classifier: the logits are exponentiated and interpreted as concentration
parameters. In its original form, the network is trained to output a flat
Dirichlet on out-of-distribution data and a sharp one in-distribution, which
requires out-of-distribution examples during training. Without them, the
method amounts to little more than a reparameterization of the logits.

:Idea: An exponential activation on the logits turns them into Dirichlet
    concentrations.
:Representation: :ref:`Parameterized second order <uq-second-order>`, i.e., a
    ``DirichletDistribution``.
:Advantages: One forward pass and no architectural change; the simplest route
    to a Dirichlet distribution.
:Disadvantages: The intended training scheme requires out-of-distribution
    data; without it, the method is merely a reparameterization of the logits.
:Reference: :cite:`malininPredictiveUncertaintyEstimation2018`

.. minigallery:: probly.method.prior_network.prior_network

.. _m-evidential-regression:

:func:`evidential_regression <probly.method.evidential_regression>`
-------------------------------------------------------------------

The regression counterpart of evidential classification. The final linear
layer is replaced by a head that outputs the four parameters of a
Normal-Inverse-Gamma distribution, i.e., a distribution over both the mean
*and* the variance of the target. A single forward pass therefore yields an
aleatoric estimate, the expected variance, as well as an epistemic one, the
variance of the mean.

:Idea: Predict a Normal-Inverse-Gamma prior over the mean and variance of the
    target.
:Representation: :ref:`Parameterized second order <uq-second-order>` over a
    real-valued target.
:Advantages: A single forward pass yields both an aleatoric (expected
    variance) and an epistemic (variance of the mean) estimate.
:Disadvantages: A modified head and an evidential regression loss.
:Reference: :cite:`aminiDeepEvidential2020`

.. minigallery:: probly.method.evidential_regression

.. _m-het-net:

:func:`het_net <probly.method.het_net>`
---------------------------------------

Models aleatoric uncertainty only, and is included because no other method on
this page captures correlated label noise. Instead of a single logit vector,
the network predicts a full input-dependent covariance over the logits,
parameterized in low rank to keep it affordable. This is useful where labels
are genuinely ambiguous, but it says nothing about whether the model has seen
similar data before.

:Idea: Input-dependent, low-rank correlated noise over the logits.
:Representation: A Gaussian over the logits; aleatoric uncertainty only.
:Advantages: Models correlated, input-dependent label noise at the cost of one
    forward pass and a few additional output channels.
:Disadvantages: Says nothing about epistemic uncertainty; relies on a sampled
    softmax during training.
:Reference: :cite:`collierCorrelatedInputDependent2021`

.. minigallery:: probly.method.het_net

Choosing Among Them
-------------------

The first question is usually whether the model can be retrained at all. If it
cannot, the options reduce to :ref:`laplace <m-laplace>`,
:ref:`mahalanobis <m-mahalanobis>`, and, provided the network was trained with
dropout, :ref:`dropout <m-dropout>`. If ``N`` training runs are affordable,
:ref:`ensemble <m-ensemble>` is the baseline against which everything else
should be measured, and :ref:`batchensemble <m-batchensemble>` or
:ref:`subensemble <m-subensemble>` recover much of its benefit for a fraction
of the memory. If inference latency is the binding constraint, a
parameterized method is the natural choice, such as
:ref:`natural_posterior_network <m-natural-posterior-network>` for
classification or :ref:`evidential_regression <m-evidential-regression>` for
regression. If the goal is specifically out-of-distribution detection, the
distance-aware methods :ref:`ddu <m-ddu>`, :ref:`sngp <m-sngp>`, and
:ref:`duq <m-duq>` are preferable.

Whichever method is chosen, whether it actually works can only be decided
empirically, with the tools described in :ref:`uq-evaluating`.

Full API
--------

.. autosummary::
    :nosignatures:

    ~probly.transformation.dropout
    ~probly.transformation.dropconnect
    ~probly.transformation.bayesian
    laplace
    ~probly.transformation.ensemble
    ~probly.transformation.batchensemble
    ~probly.transformation.subensemble
    dare
    duq
    ddu
    ~deup.deup
    sngp
    evidential_classification
    ~probly.transformation.posterior_network
    mahalanobis
    ~probly.transformation.natural_posterior_network.natural_posterior_network
    ~prior_network.prior_network
    evidential_regression
    het_net
