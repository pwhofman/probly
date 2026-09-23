.. _uq-quantifying:

========================
Quantifying Uncertainty
========================

From an Object to a Number
--------------------------

- a :ref:`representation <uq-representing>` is an object, such as a vector, a collection of vectors, or a set
- decisions, however, need an order: which input is more uncertain than which
- a measure provides that order, as a function from the object to a single scalar :cite:`hullermeierAleatoricEpistemic2021`
- such a scalar is a summary, and every summary discards information
- the question is therefore never "how uncertain is it?" but "uncertain in which sense?"
- in this sense, the measure is a modeling choice on a par with the choice of method
- since the same representation admits several measures, quantification is a stage of its own

.. _uq-measures:

Measures
--------

- each measure answers one specific question about the representation
- the families below correspond to these questions, and each names what the library implements for it

- **spread of one distribution**: entropy in the categorical case, variance in the real-valued one :cite:`hullermeierAleatoricEpistemic2021`
- it reads a single first-order distribution and says how strongly the outcomes compete
- it says nothing about whether the distribution deserves trust
- for a collection, the same question is asked of its mean prediction, that is, the entropy or variance of the averaged distribution
- for ordered classes there are ordinal counterparts, which take the class order into account instead of treating the labels as unrelated

- **disagreement within a collection**: mutual information, variance across members, maximum pairwise disagreement, minimal expected total variation :cite:`depewegDecompositionUncertainty2018, saleSecondOrder2024`
- it reads a second-order object and says how far apart the plausible distributions are
- it is zero whenever the members agree, even if they all agree on a flat distribution
- its cheaper variants read only the top class: the expected gap between a member's top probability and the probability it assigns to the winner of the mean, or the complement of the top probability before and after averaging

- **unassigned mass**: vacuity and Dempster-Shafer uncertainty for a Dirichlet :cite:`sensoyEvidentialDeep2018, malininPredictiveUncertaintyEstimation2018`
- it reads evidence that was never gathered rather than outcomes that compete
- it is defined only where the representation carries an evidence mass, which a sampled collection does not

- **width of a set**: size of a prediction set, length of an interval, generalized Hartley measure of a credal set :cite:`abellanNonSpecificity2000, angelopoulosGentleIntroduction2021`
- it reads how much had to be kept in play
- on a :ref:`credal set <uq-credal>`, measures come in lower/upper pairs, so the answer is an interval, and the gap between its ends is the epistemic reading

- **spread of a sample**: the sample variance of repeated predictions, which is the regression counterpart of member disagreement
- **spread in representation space**: a von Neumann entropy over a kernel of the embeddings, which asks how much of the feature space the prediction draws on rather than which outcomes compete

- one further family is generated rather than listed: given a proper scoring rule, the expected loss under the best possible report is a generalized entropy :cite:`gneitingStrictlyProperScoring2007`
- log loss yields the Shannon entropy, Brier loss the Gini entropy, and spherical and zero-one losses yield further ones, so the rule is the actual choice and the measure follows from it

- note that entropy and set size are not competing answers to one question, but answers to different ones

- a measure read off a :ref:`sampled second-order representation <uq-second-order>` is an estimate, not the quantity itself
- since the members are a finite draw, the number carries sample-size bias and moves with the member count
- the count should therefore be reported next to the number, as runs at different counts are not comparable

.. _uq-measure-matrix:

Which Measures Apply to Which Representation
---------------------------------------------

- a measure is only defined on representations rich enough to feed it
- a point prediction supports no measure, since there is nothing to summarize
- a single distribution supports total spread only, as aleatoric and epistemic cannot be separated on it
- a sampled second-order representation supports total, aleatoric and epistemic measures (:ref:`uq-decomposition`), estimated from finitely many members
- a parameterized second-order representation supports the same three, plus the evidence-based measures its family defines
- a credal set supports lower and upper bounds, the gap between them as its epistemic reading, and a non-specificity measure such as the generalized Hartley measure for its width
- a prediction set supports its size, with coverage as a guarantee rather than as a measure

- picking a method fixes the representation, and the representation fixes the menu
- consequently, a number is only comparable to another number computed by the same measure on the same kind of representation

Choosing a Measure
------------------

- four questions narrow the menu down to one, best asked in this order

- **what is being asked?** how strongly the outcomes compete calls for a spread measure, how much the model disagrees with itself for a disagreement measure, how much evidence was never gathered for an evidence measure, and how much had to be kept in play for a width measure
- **what does the target look like?** unordered classes take entropy, real-valued targets take variance, and ordered classes take the ordinal counterparts, which treat mass spread over neighboring classes as less uncertain than mass spread over distant ones
- **which representation is at hand?** the matrix above is the hard constraint, and a measure requested of a representation that cannot feed it should fail rather than return a value
- **must the answer be a single number?** a credal set answers in lower/upper pairs by design, and collapsing the pair to its midpoint discards exactly what the representation was chosen for

- two further rules of thumb apply
- if the number is to drive a decomposition, pick the scoring rule first and take the measure it generates, so that quantification and :ref:`evaluation <uq-evaluating>` grade with the same rule
- if the number only has to *rank* inputs, the cheap top-class measures often rank about as well as the full-distribution ones at a fraction of the cost, and :ref:`selective prediction <uq-evaluating>` is the way to check this

- the implemented menu, with exact signatures and the backends each measure supports, is documented in :mod:`probly.quantification`

.. _uq-decomposition:

Decomposing the Total
---------------------

- the two failures in :ref:`uq-why` look identical as a single number, and a decomposition keeps them apart
- its parts are the :ref:`aleatoric <uq-aleatoric>` and :ref:`epistemic <uq-epistemic>` uncertainty of :ref:`uq-sources`, now as numbers rather than as a distinction
- a decomposition is not a single formula but a pattern: total, aleatoric and epistemic, with the parts adding up :cite:`hullermeierAleatoricEpistemic2021, depewegDecompositionUncertainty2018`
- the additive form is what makes the ratio between the parts readable, and it requires a collection to average over

- in the entropy version, for a second-order representation:
- **total**: the entropy of the mean prediction, that is, all the uncertainty there is
- **aleatoric**: the mean of the members' entropies, which remains even when all members agree
- **epistemic**: the mutual information, which vanishes once the members agree
- note that the entropy of the mean is not the mean of the entropies, and the gap between the two is exactly the epistemic part

- with variance in place of entropy, the same pattern handles regression
- the credal version reads off the set instead of averaging: total is the upper entropy, aleatoric the lower entropy, and epistemic the gap between them :cite:`abellanDisaggregatedTotal2006`
- more generally, every generalized entropy from :ref:`uq-measures` yields a decomposition of its own: log loss gives back the Shannon version, Brier loss the Gini version, and other proper rules further ones :cite:`gneitingStrictlyProperScoring2007`
- propriety is essential here, since it is what keeps the epistemic part non-negative, and under an improper rule the split may become meaningless

- the reading is operational rather than metaphysical :cite:`lahlouDirectEpistemic2023`
- in other words, it is the :ref:`reducibility test <uq-sources>` computed rather than argued, and the epistemic part is the one that more data can remove
- high aleatoric, low epistemic: the model is confident that the input is ambiguous, and more data will not help
- low aleatoric, high epistemic: the members are individually confident but mutually contradictory, so collecting data is worthwhile
- both high: the input is ambiguous, and the model has no basis for it either
- the responses from :ref:`uq-why` follow the same split, with a high epistemic part being the case for collecting data or routing to a human

- the additive entropy split is the standard one, yet it is not settled: conditional entropy and mutual information violate several natural axioms for such a decomposition :cite:`wimmerQuantifyingAleatoricEpistemic2023`

Common Pitfalls
---------------

- the total has to be computed from the mean distribution, not by averaging the members' scalars: the latter yields the aleatoric part and silently drops the epistemic one
- a split that the representation does not define should fail loudly rather than return a number that looks like the others
- aleatoric and epistemic numbers from two different setups are not comparable, even on the same inputs, because the split is relative to the feature set and the model class (:ref:`uq-sources`) :cite:`hullermeierAleatoricEpistemic2021`
- different measures rank the same inputs differently, so the measure used should always be reported
- measures on different scales are not comparable across representations or class counts, and normalizing entropy by its maximum, the log of the class count, makes runs comparable but changes what "high" means
- a scalar cannot be validated on its own, only against outcomes, see :ref:`evaluating uncertainty <uq-evaluating>`
