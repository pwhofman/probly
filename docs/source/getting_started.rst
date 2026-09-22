.. _getting_started:

Getting Started
================

``probly`` turns any PyTorch, Flax/JAX or scikit-learn model into an uncertainty-aware one in a
single line of code, and offers methods, representations and quantification measures in one place,
without committing you to a framework or to a particular view of what uncertainty is.

Machine learning is increasingly deployed in safety-critical settings, where a prediction alone is
not enough: one also wants to know how far it can be trusted. ``probly`` makes that quantifiable,
and separates the two sources that matter in practice, aleatoric uncertainty arising from noise in
the data and epistemic uncertainty arising from a lack of knowledge, which can be reduced with more
data or a better model.

To follow the example below, start with the :ref:`installation` instructions; for the theory behind
it and further reading, see :ref:`next_steps`.

How Probly Works
-----------------

A model becomes uncertainty-aware in four stages:

.. image:: _static/readme/from_paper/paper_workflow_light.png
    :class: only-light

.. image:: _static/readme/from_paper/paper_workflow_dark.png
    :class: only-dark

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
