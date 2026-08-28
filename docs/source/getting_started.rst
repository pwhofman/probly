.. _getting_started:

Getting Started
================

``probly`` is an open source library giving easy access to uncertainty representation and quantification by allowing
the transformation of any PyTorch, Flax/JAX or scikit-learn model into an uncertainty aware one with a single line of code.
It offers an extensive collection of methods, representations as well as their corresponding quantification measures,
and by simply choosing one of the many methods, your provided model can become uncertainty aware.

``probly`` aims to to unify what already exists in one easy to use implementation without hard dependencies to only one
framework or particular approach to modeling uncertainty.

This section serves as a fundamental overview of the importance of uncertainty as well as the basic functions of ``probly``
using an easy to follow example. To follow along start with the :ref:`installation` instruction. For a more detailed explanation
on ``probly``, its component as foundational theories, or additional reading material please refer to the section
:ref:`next_steps`.

**Uncertainty**

As machine learning is increasingly being used in real-world application, especially in safety-critical fields it becomes
imperative to represent and quantify uncertainty, as the question poses "How can we trust the predictions of machine learning
systems?"

At its core uncertainty describes the lack of confidence or precision in a model's prediction, mainly stemming from imperfect or
incomplete information, for example through noise in observation.
Using probability provides a way of handling the randomness of predictive modeling by sorting the prediction of an instance into
a probability distribution.
Being able to quantify a model is necessary as it gives a measure on how reliable not only the model, but each prediction is.
Uncertainty can be divided into two categories for an even better interpretation of its reliance: Epistemic and Aleatoric uncertainty.
Epistemic uncertainty refers to lack of knowledge and therefore can be reduced, whereas aleatoric uncertainty stems from the inherent randomness,
which cannot be reduced.
In practice it is worthwhile trying to reduce the epistemic uncertainty by improving or changing model.
For a more detailed explanation on uncertainty in machine learning and its relevance in practice refer to ~link~.

How Probly Works
-----------------

``probly`` offers all the relevant tools in one place to transform any model into an uncertainty aware one in a four-stage process:

.. image:: _static/readme/from_paper/paper_workflow_light.png

The example follows the ``probly`` pipeline and applies it to an exemplary model, in this case the Two-Moons Dataset, and applies
the dropout method to make it uncertainty aware.

To make your model uncertainty aware, using a method like ``dropout`` you need the following imports:

.. code-block:: python

    import torch
    from torch import nn
    from sklearn.datasets import make_moons

    from probly.method import dropout                            # choose your preferred method
    from probly.representer import representer                   # either use the default representer or chose a custom matching one
    from probly.quantification import quantify
    from probly.evaluation.selective_prediction import selective_prediction  # if you want to evaluate the data, use the given evaluation

For an overview on methods and representer check out the respective chapters in the user-guide.

As well as your model to transform, we are defining a simple Classifier as follows:

.. code-block:: python

    class MLPClassifier(nn.Module):
        def __init__(
            self, in_features: int = 2, hidden_features: int = 64, out_features: int = 2
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, out_features),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

To use the two-moons dataset using PyTorch, we must prepare it as follows:

.. code-block:: python

    X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()



Now we can begin the ``probly`` pipeline:

1. Transformation
~~~~~~~~~~~~~~~~~~

The first step is choosing any of the methods to transform the model into an uncertainty aware one.
Instead of changing the model however, it wraps it as a ``Predictor`` (link) and therefore does not change
the underlying model. This step allows the measuring of uncertainty as well as the splitting into
Aleatoric and Epistemic Uncertainty.
For this example we use ``dropout``:

.. code-block:: python

    base_model = MLPClassifier()

    dropout_model = dropout(
        base_model,
        p=0.5,
        predictor_type="logit_classifier",
        shared_mask=True,
    )

Depending on the method you might need to adapt the parameters or proceed with the default option.
The user-guide (link) or alternatively the API reference :ref:`api_ref` offer information about
all the transformations.

Train the wrapped model just like you would train the original one, dropout stays active at inference
time, which is what enables repeated forward passes to produce a distribution over predictions:

.. code-block:: python

    opt = torch.optim.Adam(dropout_model.parameters(), lr=1e-3)

    dropout_model.train()
    for _ in range(300):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(dropout_model(X_tensor), y_tensor)
        loss.backward()
        opt.step()

    dropout_model.eval()

2. Representation
~~~~~~~~~~~~~~~~~~

Having the correct Representation is key for later measuring and evaluating the uncertainty. ``probly`` offers
both first and second order distributions as well as credal sets. To choose the representation either select
the generic ``representer`` or any of the more targeted representers (link).

.. code-block:: python

    rep = representer(dropout_model, num_samples=100)
    representation = rep.represent(X_tensor)

3. Quantification
~~~~~~~~~~~~~~~~~~
To understand and decide on a model's robustness, there needs to be a metric. Quantification delivers such a measure
either as an actual measure or a decomposition depending on the transformation. This choice is decided downstream through the
``quantify`` method:

.. code-block:: python

    quantification = quantify(representation)
    uncertainty = quantification.total.detach().numpy()
    if uncertainty.ndim > 1:
        uncertainty = uncertainty.sum(axis=-1)

4. Evaluation
~~~~~~~~~~~~~~~
Finally ``probly`` offers a unified evaluation structure. Here we perform selective prediction: sorting instances
by their uncertainty and checking whether the most uncertain ones are indeed the ones the model gets wrong.

.. code-block:: python

    with torch.no_grad():
        predictions = dropout_model(X_tensor).argmax(-1)
    losses = (predictions != y_tensor).float().numpy()

    aurc, bin_losses = selective_prediction(uncertainty, losses)
    print(f"AURC: {aurc:.4f}")


Output
~~~~~~~
As Output you can expect something like this:

.. code-block:: text

    AURC: 0.0061

Plotting allows easy understanding of uncertainty, a variation of the dropout example can be seen here:

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Keep dropout active at inference and average several stochastic forward passes.">

.. only:: html

  .. image:: /auto_examples/transformation/images/thumb/sphx_glr_plot_dropout_thumb.png
    :alt:

  :doc:`/auto_examples/transformation/plot_dropout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MC Dropout on Two Moons</div>
    </div>

.. raw:: html

    </div>

.. _next_steps:

Next Steps
-----------

Many steps happen downstream behind the curtain, allowing easy application of ``probly``!
However to get a deeper understanding of ``probly``, its workings and how to use it for your own applications, check out
our :ref:`user_guide`.

For the exhaustive overview of methods check out :ref:`api_ref` and for a practical insight look at :ref:`examples`.
Found a method that is missing please refer to our guide :ref:`adding_a_method`.

Finally have fun using ``probly``!
