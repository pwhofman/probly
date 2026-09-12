.. _uq-sources:

=========================
Where Uncertainty Arises
=========================

.. _uq-aleatoric:

Aleatoric Uncertainty
---------------------

- back to case a: the blurry photo, or the one where humans disagree
- given these pixels the outcome is genuinely not determined, the world holds a 51/49 mix
- the best possible model on these features still outputs 0.51, the error rate is part of the deal
- name: **aleatoric** uncertainty, from *alea*, the dice, also called data uncertainty :cite:`hullermeierAleatoricEpistemic2021`
- lives in the data-generating process: class overlap, label noise, measurement noise
- annotator disagreement makes it measurable, many humans label the same image and the split of their votes is the aleatoric part :cite:`petersonHumanUncertainty2019`
- more data sharpens the estimate of the odds, it does not remove the odds

.. _uq-epistemic:

Epistemic Uncertainty
---------------------

- back to case b: an animal the model never trained on
- the world is not mixed here, the model is, the 0.51 came from the model and not from the image
- uncertainty about the model itself: too little data in this region, or a model class that cannot fit the true relation
- name: **epistemic** uncertainty, from *episteme*, knowledge, also called model uncertainty :cite:`kendallWhatUncertainties2017`
- reducible: the right data, labeled examples near this input, moves it
- this is the claim from :ref:`uq-why` that a single distribution has no slot for
- it shows up as disagreement between plausible models, which is why it needs the richer objects of :ref:`representing uncertainty <uq-representing>`

The Reducibility Test
---------------------

- one question separates the two: **would more data move this number?**
- case a: no, more blurry photos re-estimate the same 51/49, aleatoric
- case b: yes, labeled examples of the new animal would move it, epistemic
- "more data" is doing work here: more samples of the same kind, targeted labels near this input, and new features are three different interventions
- the test picks the response: epistemic justifies collecting or routing to a human, aleatoric says stop collecting, accept the odds or improve the features
- the split is exactly the information the identical 0.51s could not carry
- it is also computable: a total uncertainty decomposes into an aleatoric and an epistemic part, :ref:`decomposing the total <uq-decomposition>`

Why the Split Is Relative
-------------------------

- the split is not a property of the world, it is a property of the modeling setup :cite:`hullermeierAleatoricEpistemic2021`
- relative to the feature set: blurry dog vs. fox is aleatoric for a pixel classifier, add a sharper sensor or a second view and part of the noise becomes signal you do not have yet, epistemic
- relative to the model class: what a linear model must write off as noise, a richer model can resolve
- relative to context: a coin flip is aleatoric to the bettor, epistemic to the physicist measuring the throw
- "irreducible" always means irreducible given this feature set and this model class
- consequence: aleatoric and epistemic numbers from different setups are not comparable

Distribution Shift
------------------

- everything so far assumed training and deployment draw from the same distribution
- deployment breaks that: a new sensor, a new season, a new population
- case b at scale: not one strange animal but a whole regime without training density
- shift is the canonical mass producer of epistemic uncertainty at deployment time
- two flavors worth naming: **covariate shift**, the inputs move, and **concept drift**, the input-outcome relation moves, so even the aleatoric odds change
- whether an uncertainty estimate holds up under shift is an empirical question, and often the answer is no :cite:`snoekCanYouTrust2019`
- out-of-distribution inputs are the far end of shift, whether a score separates them from familiar ones is a test in :ref:`evaluating uncertainty <uq-evaluating>`
