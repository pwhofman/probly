.. _demos:

Demos
=====

What ``probly`` makes of real models and data. Every demo was computed once with ``probly``, and its
page shows the result, interactive where that helps. The scripts that computed it are in the
repository, next to its page, so that you can run it again or adapt it to your own models and data.

Demos are not :doc:`examples <../examples>`: the examples run with every build of this documentation
and show how to use the API, step by step. A demo is a computation that takes too long or needs too
large a model to run with the docs, and its page shows what comes out of it.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Hurricane Milton: chaos, or what the models don't know?
      :link: milton/index
      :link-type: doc
      :img-top: /_static/demos/milton.png
      :img-alt: The Milton demo: the runs of WeatherNext's forecast of Hurricane Milton over the Gulf of
                Mexico, with their spread split into the part that changes from run to run (red-pink)
                and the part where the models disagree (blue).

      WeatherNext's forecast of Hurricane Milton, four days ahead: ``probly`` splits the spread of its
      runs into the atmosphere's chaos (aleatoric) and the disagreement of its models (epistemic).

.. toctree::
   :hidden:

   milton/index
