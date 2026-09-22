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

Each entry names the idea behind the method, the representation it hands back,
its advantages and disadvantages, and the paper it comes from, and links to the
worked example in the gallery. The trade-offs run from post-hoc wrappers you can
put on an already-trained network to methods that change the architecture and
the loss.

.. toctree::
    :maxdepth: 2

    second_order
    credal
    conformal
    calibration
