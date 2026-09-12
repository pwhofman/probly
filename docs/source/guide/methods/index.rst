.. _methods:

===================
Uncertainty Methods
===================

:ref:`uq-representing` argued that choosing a method is choosing a
representation. This part makes that concrete: it walks the methods ``probly``
ships, grouped by the representation they produce.

The four families answer four different questions.  A **second-order
distribution** asks *how much would my predictive distribution move if I had
trained differently?* A **credal set** refuses to commit to one distribution at
all and reports a set of admissible ones. **Conformal prediction** gives up on
describing the distribution and instead returns a set of labels with a coverage
guarantee. **Calibration** keeps the first-order distribution and only fixes
how its probabilities are scaled.

Each entry names the paper it comes from, the representation it hands back, and
what it costs you at training and at inference time. None of them are free: the
cheapest ones are post-hoc wrappers you can put on an already-trained network,
the most expensive ones change the architecture and the loss.

.. note::

    The four pages below deliberately use four different presentation styles so
    they can be compared side by side. The style will be unified once one is
    chosen.

.. toctree::
    :maxdepth: 2

    second_order
    credal
    conformal
    calibration
