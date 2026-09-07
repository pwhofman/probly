.. _uq-quantifying:

========================
Quantifying Uncertainty
========================

From an Object to a Number
--------------------------

- a :ref:`representation <uq-representing>` is an object: a vector, a collection of vectors, a set
- decisions need an order: which input is more uncertain than which
- a measure is a function from that object to one scalar
- the scalar is a summary, and every summary throws information away
- so the question is never "how uncertain is it?" but "uncertain in which sense?"
- the measure is a modeling choice, on the same level as the choice of method
- which is why it is a separate stage: the same representation admits several measures

.. _uq-measures:

Measures
--------

- a measure answers one specific question about the representation

- **spread of one distribution**: entropy for the categorical case, variance for the real-valued one
- reads a single first-order distribution, says how much the outcomes compete
- says nothing about whether the distribution deserves trust

- **disagreement inside a collection**: mutual information, variance across members, maximum pairwise disagreement, minimal expected total variation
- reads a second-order object, says how far the plausible distributions are apart
- zero when every member agrees, even if all agree on a flat distribution

- **mass not assigned**: vacuity and Dempster-Shafer uncertainty for a Dirichlet
- reads evidence that was never gathered rather than outcomes that compete

- **width of a set**: size of a prediction set, length of an interval, generalized Hartley measure of a credal set
- reads how many outcomes had to be kept to stay honest

- entropy and set size are not competing answers to one question, they answer different ones
- a measure is only defined on representations rich enough to feed it

.. _uq-decomposition:

Decomposing the Total
---------------------

- the two failures in :ref:`uq-why` look identical in one number, a decomposition keeps them apart
- a decomposition is not one formula but a pattern: total, aleatoric, epistemic, with the parts adding up
- the additive form is what makes the ratio between the parts readable, and it needs a collection to average over

- the entropy version, for a second-order representation:
- **total**: entropy of the mean prediction, all the uncertainty there is
- **aleatoric**: mean of the members' entropies, what stays even when every member agrees
- **epistemic**: mutual information, the part that disappears once the members agree

- the same pattern with variance instead of entropy handles regression
- the credal version reads off the set instead of averaging: total is the upper entropy, aleatoric the lower entropy, epistemic the gap between them
- an entropy is the special case of a more general construction: pick a proper scoring rule, and its expected loss under the best possible report is a generalized entropy
- log loss gives back the Shannon-entropy decomposition, Brier loss the Gini one, and other proper rules give further ones
- propriety is not decoration here: it is what keeps the epistemic part non-negative, an improper rule can make the split meaningless

- the reading is operational, not metaphysical
- high aleatoric, low epistemic: the model is sure the input is ambiguous, more data will not help
- low aleatoric, high epistemic: the members are individually confident and mutually contradictory, collect data
- both high: an ambiguous input the model also has no basis for
- the third response option from :ref:`uq-why` maps onto the same split

.. _uq-measure-matrix:

Which Measures Apply to Which Representation
---------------------------------------------

- a point prediction supports no measure, there is nothing to summarize
- a single distribution supports total spread only, aleatoric and epistemic cannot be separated
- a sampled second-order representation supports all three parts, estimated from finitely many members
- a parameterized second-order representation supports all three, plus the evidence-based measures its family defines
- a credal set supports lower and upper bounds, and the gap between them as its epistemic reading
- a prediction set supports size, and coverage as its guarantee rather than as a measure

- picking a method fixes the representation, and the representation fixes the menu
- so a number is only comparable to another number computed from the same measure on the same kind of representation

Where Decomposition Is Undefined
--------------------------------

- a single softmax vector has no second-order object behind it, so its entropy is total, not aleatoric
- calling it aleatoric imports an assumption that the model is correct
- some methods target one part by construction: a heteroscedastic head models label noise, so it yields an aleatoric quantity and no epistemic one
- an undefined split should fail loudly rather than return a number that looks like the others
- members trained the same way on the same data agree for reasons that have nothing to do with the input
- distribution shift moves both parts at once, and the split is defined relative to the model class, not the world
- the split is only as trustworthy as the collection it is computed over

Common Pitfalls
---------------

- entropy of the mean is not the mean of the entropies, the gap between them is the epistemic part
- averaging the members' scalars first destroys the very quantity you wanted
- ranking inputs by different measures gives different rankings, so report which one was used
- measures on different scales are not comparable across representations or class counts
- normalizing entropy by the number of classes makes runs comparable, but changes what "high" means
- a scalar cannot be validated on its own, only against outcomes, see :ref:`evaluating uncertainty <uq-evaluating>`
