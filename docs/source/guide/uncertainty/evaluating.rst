.. _uq-evaluating:

=======================
Evaluating Uncertainty
=======================

Why Evaluation Is Harder Than Accuracy
--------------------------------------

- a quantifier always returns a number, including when that number is meaningless
- accuracy compares a prediction to a label, and both live in the same space
- an uncertainty estimate has no such label, since no single outcome says that "0.7 was the right confidence"
- its target is a property of a distribution, of which only one draw is ever observed
- evaluation is therefore always over a set of inputs, never over a single one
- in other words, there is no ground-truth uncertainty to regress against, only outcomes with which the claim should be consistent
- and the claim is only as meaningful as the question the :ref:`measure <uq-quantifying>` asked

Two Questions, Not One
----------------------

- **intrinsic**: is the prediction itself right? graded by proper scoring rules, calibration error, and coverage
- **downstream**: is the derived score useful? graded by out-of-distribution detection, selective prediction, and active learning
- the two come apart in both directions
- a well-calibrated model may rank inputs badly, in which case the abstention it drives buys nothing
- conversely, a score that separates in- from out-of-distribution inputs perfectly may be miscalibrated as a probability
- the criterion should thus be chosen by what the estimate is for, and one should never be reported as evidence for the other

- the intrinsic checks need a rung against which outcomes can be counted, that is, a :ref:`first-order distribution <uq-first-order>` or a :ref:`set of outcomes <uq-sets>`
- a :ref:`second-order representation <uq-second-order>` is graded through its mean prediction, while the second-order claim itself has no observed frequency to be checked against
- the rung that was evaluated is therefore not always the rung that was produced, and a result has to state which one it refers to

Proper Scoring Rules
--------------------

- a scoring rule grades a whole predicted distribution against the observed outcome
- it is **proper** if the expected score is optimized by reporting one's true belief :cite:`gneitingStrictlyProperScoring2007`
- it is **strictly proper** if no other report ties, so that honesty is the unique optimum
- log loss and Brier loss are the usual choices, spherical loss another
- zero-one loss is the accuracy-like case, and it is proper but not strictly so, since every report with the same ``argmax`` ties
- accuracy shares this blindness: it ignores confidence altogether, and can therefore neither reward honest probabilities nor penalize overstated ones
- a proper rule mixes being right with being calibrated in one number, which makes it a good default but a poor diagnosis
- decomposing it into calibration and sharpness recovers what the aggregate hides
- the same rules generate the decompositions in :ref:`uq-decomposition`, so the rule chosen for evaluation and the one chosen for quantification should be the same :cite:`hullermeierAleatoricEpistemic2021`

Calibration
-----------

- the promise is that among inputs given confidence 0.7, about 70 percent are predicted correctly
- it is checked by binning the predictions and comparing, within each bin, the average confidence to the observed frequency
- the result is a reliability diagram to inspect and an expected calibration error to summarize it :cite:`guoOnCalibration2017`
- that summary depends on the binning, is biased by the bin count, and hides the direction of the error, so the diagram should be read as well :cite:`kumarVerifiedCalibration2019`
- top-label and class-wise calibration are different promises, and the latter is the stricter one :cite:`kullBeyondTemperatureScaling2019`
- calibration is necessary, not sufficient: predicting the base rate for every input is perfectly calibrated and entirely useless
- it should therefore be paired with sharpness, following the principle of being as sharp as calibration permits
- note also that marginal calibration does not imply calibration within a subgroup
- post-hoc recalibration fixes the mapping but not the ranking underneath it, at least for monotone maps, whereas matrix and Dirichlet scaling have enough freedom to reorder :cite:`guoOnCalibration2017, kullBeyondTemperatureScaling2019`

Coverage and Set Size
---------------------

- for the :ref:`set-valued rung <uq-sets>`, the intrinsic question is coverage, that is, how often the true outcome lies inside the set
- coverage alone is trivially achieved by returning everything, so it is only meaningful next to size
- size is the efficiency side: the number of labels kept, the width of an interval, the extent of a credal set
- conformal methods fix coverage by construction, which turns the comparison into one of size at matched coverage :cite:`angelopoulosGentleIntroduction2021`
- marginal coverage is an average over inputs and does not promise that the guarantee holds per class or per group
- the guarantee also rests on exchangeability of calibration and test data, so :ref:`distribution shift <uq-sources>` is exactly what voids it

Selective Prediction
--------------------

- this is the decision from :ref:`uq-why`: to abstain, or to route the input to a human
- inputs are sorted by their uncertainty score, and the most uncertain ones are rejected first :cite:`geifmanSelectiveClassification2017`
- **coverage** here denotes the fraction of inputs kept, which is not the conformal notion above, and **risk** the loss among those kept
- a good ranking makes risk fall as coverage falls, which the risk-coverage curve shows and the area under it summarizes, lower being better
- being rank-based, the check grades the ordering and ignores the scale, unlike calibration
- it directly answers the operational question of what error rate remains if, say, 10 percent of inputs are handed off
- the curve is the more informative artifact, since the single area hides where the gain occurs

Out-of-Distribution Separation
------------------------------

- the promise is that inputs unlike the training data receive higher :ref:`epistemic uncertainty <uq-epistemic>`
- it is graded as a detection problem, with in-distribution inputs against a held-out foreign set
- AUROC measures the overall ranking
- average precision is preferable when the foreign set is small, since AUROC is insensitive to the base rate and AP is not
- the false positive rate at a fixed true positive rate describes the operating point one would actually run :cite:`leeSimpleUnifiedFramework2018`
- the result depends entirely on which foreign set was chosen, and near- and far-OOD are different problems :cite:`snoekCanYouTrust2019, hendrycksBenchmarkingNeural2019`
- in the :ref:`vocabulary of shift <uq-sources>`, the foreign set amounts to covariate shift, near-OOD to a mild one and far-OOD to a regime with no training density at all
- concept drift cannot be tested this way, since it moves the input-outcome relation rather than the inputs, and thus shows up as a rising error rate rather than as a separable score
- consequently, strong far-OOD numbers say little about the near-OOD case, which tends to be the costly one in practice
- the check grades the ranking under shift, not the calibration in distribution
- any score can be plugged in, including plain softmax confidence, so a method has to beat that baseline to justify its cost :cite:`hendrycksBaselineDetecting2017`

Active Learning
---------------

- the promise is that uncertainty identifies the examples worth labeling next
- one acquires by the score, retrains, and compares the learning curve against random acquisition :cite:`galDropoutBayesian2016`
- random acquisition is the baseline to beat, and it tends to be stronger than it appears :cite:`munjalRobustActiveLearning2022`
- this is the criterion that tests the epistemic part specifically, since the reducible part is what labeling is supposed to remove, and the usual acquisition function is the mutual information from :ref:`uq-decomposition`
- in other words, it is the :ref:`reducibility test <uq-sources>` run as an experiment: if the flagged inputs really were the reducible ones, labeling them moves the curve faster than random acquisition does
- the area under the learning curve summarizes the result, but the early shape of the curve matters most

Choosing a Criterion
--------------------

- the criterion should match what the estimate is for
- if full predictive distributions are compared: a proper scoring rule
- if the number is read as a probability: calibration, with the diagram and not only the summary
- if a coverage guarantee is required: set size at matched coverage
- if the number gates an abstention or a handoff: selective prediction, where only the ranking matters
- if the point is to detect inputs unlike the training data: OOD separation, stated together with its foreign set
- if the point is to choose what to label: active learning, against random acquisition
- the :ref:`representation <uq-representing>` limits the intrinsic part of this menu in the same way it limits the measures: a point prediction admits only accuracy, a set only coverage and size, and a second-order object only what its mean prediction exposes
- more than one criterion should be reported, since they disagree by design, and a method that wins on all of them is rare
- ultimately, an uncertainty estimate is a model output like any other: it can be confidently wrong, and only these checks reveal whether it is
