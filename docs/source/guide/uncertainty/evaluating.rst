.. _uq-evaluating:

=======================
Evaluating Uncertainty
=======================

Why Evaluation Is Harder Than Accuracy
--------------------------------------

- a quantifier always returns a number, including when the number is meaningless
- accuracy compares a prediction to a label, both live in the same space
- an uncertainty estimate has no label, no single outcome says "0.7 was the right confidence"
- the target is a property of a distribution, and you observe one draw from it
- so evaluation is always over a set of inputs, never a single one
- there is no ground-truth uncertainty to regress against, only outcomes that should be consistent with the claim
- and the claim is only as meaningful as the question the :ref:`measure <uq-quantifying>` asked

Two Questions, Not One
----------------------

- **intrinsic**: is the predicted distribution right? proper scoring rules, calibration error, coverage
- **downstream**: is the derived score useful? out-of-distribution detection, selective prediction, active learning
- the two come apart in both directions
- a well-calibrated model can rank inputs badly, so the abstention it drives buys nothing
- a score that separates in- from out-of-distribution perfectly can be miscalibrated as a probability
- pick by what the estimate is for, and never report one as evidence for the other

Proper Scoring Rules
--------------------

- a scoring rule grades a whole predicted distribution against the observed outcome
- **proper**: the expected score is optimized by reporting your true belief
- **strictly proper**: no other report ties it, honesty is the unique optimum
- log loss and Brier loss are the usual choices, spherical loss another
- zero-one loss is the accuracy-like case, and it is exactly the one that is not strictly proper, several reports tie
- accuracy itself is not proper, a model can improve it by overstating confidence
- a proper rule mixes being right with being calibrated in one number, which is why it is a good default and a poor diagnosis
- decomposing it into calibration plus sharpness recovers what the aggregate hid
- the same rules generate the decompositions in :ref:`uq-decomposition`, so a rule chosen for evaluation and one chosen for quantification should be the same rule

Calibration
-----------

- the promise: among inputs given confidence 0.7, about 70 percent are correct
- checked by binning predictions and comparing average confidence to observed frequency in each bin
- a reliability diagram to look at, an expected calibration error to summarize
- that summary depends on the binning, is biased by the bin count, and hides direction, so read the diagram too
- top-label calibration and class-wise calibration are different promises, the second is the stricter one
- calibration is necessary, not sufficient: predicting the base rate for every input is perfectly calibrated and useless
- so pair it with sharpness, be as sharp as calibration permits
- marginal calibration does not imply calibration within a subgroup
- post-hoc recalibration fixes the mapping, not the ranking underneath it

Coverage and Set Size
---------------------

- for set-valued predictions the intrinsic question is coverage: how often the true outcome is inside
- coverage alone is trivially satisfied by returning everything, so it is only readable next to size
- size is the efficiency side: number of labels kept, width of an interval, extent of a credal set
- conformal methods fix coverage by construction, which turns the comparison into one of size at matched coverage
- marginal coverage is an average over inputs, it does not promise the guarantee holds per class or per group

Selective Prediction
--------------------

- this is the decision from :ref:`uq-why`: abstain, or route to a human
- sort inputs by the uncertainty score, reject the most uncertain first
- coverage: the fraction kept, risk: the loss among those kept
- a good ranking makes risk fall as coverage falls, plot the curve, summarize by the area under it, where lower is better
- rank-based, so it grades ordering and ignores the scale, unlike calibration
- directly answers the operational question: what error rate do I get if I hand off 10 percent
- the curve is the honest artifact, the single area hides where the gain sits

Out-of-Distribution Separation
------------------------------

- the promise: inputs unlike the training data get higher epistemic uncertainty
- graded as a detection problem, in-distribution against a held-out foreign set
- AUROC for the overall ranking, average precision when the foreign set is small, false positive rate at a fixed true positive rate for the operating point you would actually run
- the number depends entirely on which foreign set was chosen, near- and far-OOD are different problems
- strong far-OOD numbers say little about the near-OOD case that actually costs you
- it grades the ranking under shift, not the calibration in-distribution
- any score can be plugged in, including plain softmax confidence, so a method has to beat that baseline to have earned its cost

Active Learning
---------------

- the promise: uncertainty picks which examples are worth labeling next
- acquire by the score, retrain, and compare the learning curve against random acquisition
- the honest baseline is random, and it is stronger than it looks
- this is the criterion that tests the epistemic part specifically, the reducible one is what labeling is supposed to remove
- summarized by the area under the learning curve, but the shape early on is what matters

Choosing a Criterion
--------------------

- match the criterion to what the estimate is for
- comparing full predictive distributions: a proper scoring rule
- the number is read as a probability: calibration, with the diagram and not only the summary
- a coverage guarantee is required: compare set size at matched coverage
- the number gates an abstention or a handoff: selective prediction, ranking is all that matters
- the point is detecting inputs unlike training data: OOD separation, stated with its foreign set
- the point is choosing what to label: active learning, against random acquisition
- report more than one, they disagree by design, and a method that wins on all of them is rare
- an uncertainty estimate is a model output like any other, it can be confidently wrong, and only these checks say whether it is
