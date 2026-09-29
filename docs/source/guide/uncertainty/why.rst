.. _uq-why:

============================
Why One Number Is Not Enough
============================

The 0.51 Problem
----------------

- consider a classifier over three classes, "cat", "dog" and "fox", that outputs 0.51 for "dog"
- the output is perfectly valid, yet ``argmax`` discards it and nobody ever examines it
- suppose one does examine it: the number admits two readings
- **as a ranking**, it says that "dog" narrowly beats "fox"
- **as a claim about the world**, it says that among images like this one, 51 percent show a dog
- only the second reading makes the number useful for deciding, and nothing in training forces it to hold
- in other words, even before uncertainty enters the picture, the number may be unreliable at the job one assumed it was doing

Two Failures, One Number
------------------------

- fix the output at 0.51 and vary the input
- **case a**: a blurry photo, or one on which human annotators disagree; here 0.51 is correct, and more data does not move it
- **case b**: an animal the model was never trained on; here 0.51 reflects a choice forced on an input the model has no basis for
- the two outputs are identical, yet they call for opposite responses: case a says stop collecting data, case b says collect more, or route the input to a human
- telling them apart requires distinguishing two :ref:`sources of uncertainty <uq-sources>`, **aleatoric** and **epistemic**

A Richer Output Doesn't Fix It
------------------------------

- a natural remedy is to report the whole vector instead of its maximum
- indeed, (0.34, 0.33, 0.33) and (0.51, 0.49, 0.00) share the same ``argmax`` but describe different situations
- the shape of the vector says which outcomes compete, which is genuine information

- yet (0.34, 0.33, 0.33) itself still admits two readings
- **reading 1**: a well-trained model faces a genuinely ambiguous input, and the vector is a correct, confident statement about it
- **reading 2**: the model has no basis for an answer, and the architecture simply lands near uniform
- reporting the vector does not separate the two, so more numbers of the same kind are not the fix

- a distribution assigns mass to outcomes, and both readings assign the same mass to the same outcomes
- the difference lies not in the outcomes but in how far the distribution itself deserves trust
- no three numbers summing to one can carry that, since a distribution has no slot for a claim about itself

- one way out is to ask many models, or one model many times, and look at how much they disagree
- another is to report a set of plausible distributions instead of a single one
- either way, the object is no longer a distribution but a collection of them, see :ref:`representing uncertainty <uq-representing>`
- once the prediction is a collection, "how uncertain?" requires a choice of what to measure, see :ref:`quantifying uncertainty <uq-quantifying>`

Being Wrong Versus Not Knowing You Might Be
-------------------------------------------

- uncertainty does not make a model more accurate, it makes its failures anticipatable
- the difference matters because a flagged error costs an abstention, whereas an unflagged one costs its full downstream consequence
- note, however, that uncertainty estimates are model outputs themselves and can be wrong, which is why they have to be :ref:`evaluated <uq-evaluating>`
