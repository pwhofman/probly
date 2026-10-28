.. _uq-sources:

=========================
Where Uncertainty Arises
=========================

.. _uq-aleatoric:

Aleatoric Uncertainty
---------------------

- return to case a: the blurry photo, or the one on which humans disagree
- given these pixels, the outcome is genuinely not determined, and the world itself holds a 51/49 mix
- consequently, even the best possible model on these features outputs 0.51, and the resulting error rate cannot be trained away
- roughly speaking, this is **aleatoric** uncertainty, from *alea*, the dice, also called data uncertainty :cite:`hullermeierAleatoricEpistemic2021`
- it resides in the data-generating process: class overlap, label noise, measurement noise
- annotator disagreement makes it observable: when many humans label the same image, the split of their votes estimates the aleatoric part :cite:`petersonHumanUncertainty2019`
- more data sharpens the estimate of the odds, but it does not remove the odds

.. _uq-epistemic:

Epistemic Uncertainty
---------------------

- return to case b: an animal the model was never trained on
- here the world is not mixed, the model is, so the 0.51 originates in the model and not in the image
- roughly speaking, this is uncertainty about the model itself, caused by too little data in this region or by a model class that cannot represent the true relation
- it is called **epistemic** uncertainty, from *episteme*, knowledge, also called model uncertainty :cite:`kendallWhatUncertainties2017`
- it is reducible in principle: the right data, such as labeled examples near this input, moves it
- this is precisely the claim for which, as :ref:`uq-why` argued, a single distribution has no slot
- it shows up as disagreement between plausible models, which is why it requires the richer objects of :ref:`representing uncertainty <uq-representing>`

The Reducibility Test
---------------------

- a single question separates the two: **would more data move this number?**
- in case a it would not, since more blurry photos only re-estimate the same 51/49, so the uncertainty is aleatoric
- in case b it would, since labeled examples of the new animal move the prediction, so the uncertainty is epistemic
- note that "more data" is ambiguous: more samples of the same kind, targeted labels near this input, and new features are three different interventions
- the answer determines the response: epistemic uncertainty justifies collecting data or routing to a human, whereas aleatoric uncertainty says stop collecting and either accept the odds or improve the features
- in other words, the split is exactly the information that the two identical 0.51s could not carry
- it is also computable, in the sense that total uncertainty can be decomposed into an aleatoric and an epistemic part, see :ref:`decomposing the total <uq-decomposition>`

Why the Split Is Relative
-------------------------

- the last option, improving the features, already hints that the split is not a property of the world but of the modeling setup :cite:`hullermeierAleatoricEpistemic2021`
- it is relative to the feature set: blurry dog versus fox is aleatoric for a pixel classifier, yet with a sharper sensor or a second view, part of the noise becomes signal that is merely missing, that is, epistemic
- it is relative to the model class: what a linear model must write off as noise, a richer model may resolve
- it is relative to the context: a coin flip is aleatoric to the bettor, but epistemic to the physicist measuring the throw
- "irreducible" therefore always means irreducible given this feature set and this model class
- as a result, aleatoric and epistemic numbers obtained from different setups are not comparable

Distribution Shift
------------------

- everything so far assumed that training and deployment data are drawn from the same distribution
- deployment tends to break that assumption, through a new sensor, a new season, or a new population
- shift is case b at scale: not one unfamiliar animal, but a whole regime without training density
- it is thus arguably the main source of epistemic uncertainty at deployment time
- two kinds are worth distinguishing: under **covariate shift** the inputs move, under **concept drift** the input-outcome relation moves, so that even the aleatoric odds change
- whether an uncertainty estimate holds up under shift is an empirical question, and the answer is often negative :cite:`snoekCanYouTrust2019`
- out-of-distribution inputs lie at the far end of shift, and whether a score separates them from familiar ones is one of the checks in :ref:`evaluating uncertainty <uq-evaluating>`
