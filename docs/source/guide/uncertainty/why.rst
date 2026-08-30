.. _uq-why:

============================
Why One Number Is Not Enough
============================

The 0.51 Problem
----------------

- classification task, three classes "cat", "dog" and "fox", output 0.51 for "dog"
- output valid, but discarded by ``argmax`` and never examined
- what if you did examine that output?
- two readings: **as a ranking** (dog barely beats fox) and **claim about the world** (of images like this 51% are dogs)
- before we reach uncertainty, the number you have is unreliable at the job you assumed it was doing

Two Failures, One Number
------------------------

- fixed output, varied input
- case a: blurry photo or one where humans disagree, 0.51 is correct here, more data does not move it
- case b: an animal that was never trained on, 0.51 is about forcing a choice on an input with no basis
- identical output, one says stop collecting data, the other says collect
- third response option alongside collect more data: route to a human
- :ref:`sources of uncertainty <uq-sources>` enables split into **epistemic** and **aleatoric** uncertainty

A Richer Output Doesn't Fix It
------------------------------

- report the whole vector, not the max
- (0.34, 0.33, 0.33) vs (0.51, 0.49, 0.00) same max-probability, different situations
- vector shape says which outcomes compete. real information

- (0.34, 0.33, 0.33), two readings
- reading 1: well-trained model, genuinely ambiguous input. correct, confident statement
- reading 2: no basis for an answer, architecture lands at uniform
- so reporting the vector did not separate them. more numbers of the same kind did not offer a fix

- distribution assigns mass to outcomes
- both readings assign the same mass to the same outcomes
- difference is not about the outcomes, it is how far the distribution itself should be trusted
- no three numbers summing to one can carry that
- a distribution has no slot for a claim about itself

- ask many models or one model many times, look at disagreement
- or report a set of plausible distributions instead of one
- object is no longer a distribution, but a collection of them :ref:`representing uncertainty <uq-representing>`
- with a collection, "how uncertain?" needs a choice on what to measure :ref:`quantifying uncertainty <uq-quantifying>`

Being Wrong Versus Not Knowing You Might Be
-------------------------------------------

- uncertainty does not make the model more accurate
- failures become anticipatable
- flagged error costs an abstention, while unflagged error costs the full downstream consequence
- uncertainty estimates are model outputs and can be wrong, :ref:`evaluating uncertainty <uq-evaluating>`
