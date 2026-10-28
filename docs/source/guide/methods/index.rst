.. _methods:

===================
Uncertainty Methods
===================

As argued in :ref:`uq-representing`, choosing an uncertainty method is to a
large extent choosing a representation. The methods ``probly`` provides are
therefore grouped by the representation they produce rather than by the
mechanism that produces it.

Roughly speaking, the four families answer four different questions about a
prediction. A **second-order distribution** asks *how much would the
predictive distribution change had the model been trained differently?* A
**credal set** asks *which distributions remain compatible with what the model
has learned?*, and declines to rank them. **Conformal prediction** asks *which
outcomes must be retained so that the truth is covered with a prescribed
probability?*, and answers with a set of labels or an interval instead of a
distribution. **Calibration**, finally, asks *can the predicted probabilities
be taken at face value?*, and corrects their scale where they cannot.

Within each family, the methods range from post-hoc wrappers around an
already-trained network to methods that change the architecture, the loss, or
both. In practice, this is often the more decisive distinction, since it
determines whether a method is applicable to a model that cannot be retrained.
Every entry states the idea behind the method, the representation it returns,
its main advantages and disadvantages, and the paper it originates from, and
links to a worked example in the gallery.

.. toctree::
    :maxdepth: 2

    second_order
    credal
    conformal
    calibration
