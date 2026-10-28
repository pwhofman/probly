Hurricane Milton: Where the Uncertainty Comes From
==================================================

How sure is a weather model of a hurricane's path, and why? This demo shows Google DeepMind's
WeatherNext forecasting Hurricane Milton from 7 October 2024, 00 UTC, three days before it came
ashore in Florida, and splits the spread of its runs, with ``probly``, into what varies from run to
run of one model and what the models do not know.

.. probly-demo:: milton
   :fallback: demos/milton.png
   :alt: The runs of WeatherNext's forecast of Hurricane Milton over the Gulf of Mexico, with their
         spread split into the part that varies from run to run (aleatoric, blue) and the part where
         the models disagree (epistemic, red).

What You See
------------

**The forecast.** WeatherNext 2 :cite:`aletSkillfulJoint2025` and WeatherNext Cyclones
:cite:`aletOperationalTropical2026` forecast the weather of the whole globe in steps of 6 hours. Each
is four models, trained independently, and every run of a model draws random noise of its own. The
demo shows WeatherNext Cyclones at 0.25° (*Full*), with 8 runs of each of its four models, and its
small version at 1° (*Mini*), whose two public checkpoints stand in for the four models, with 8 runs
each. All checkpoints were trained on data from before 2024, so none of them has seen Milton.

**The views.**

- *One run* shows a single run as if it were the forecast: one track, one landfall, and no hint of
  how sure it is.
- *All runs* shows every run's track, the chance of tropical-storm-force wind (34 kt, the lowest
  threshold of the US National Hurricane Center's wind probabilities), and the outline of every
  run's tropical-storm-force wind. A strip along Florida's Gulf coast shows where the runs come
  ashore, each run's dot filled once it has.
- *Decomposition* splits the spread of the runs: blue where the runs of one model scatter
  (aleatoric), red where the models disagree (epistemic). Together, they make up the total spread.

**Time and places.** Play the four days or drag through them, and pick a place, or click the map, to
see it over time. *Show what happened* draws Milton's observed track, from IBTrACS
:cite:`knappInternationalBest2010`.

How It Was Computed
-------------------

WeatherNext as a Second-Order Predictor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

WeatherNext models its two kinds of uncertainty with two mechanisms :cite:`aletSkillfulJoint2025`.
Within a model, every run draws a small noise vector at every step, a perturbation of the model's
function: the runs of one model scatter as the atmosphere might, which is aleatoric uncertainty.
Across models, four models trained independently, a deep ensemble
:cite:`lakshminarayananSimpleScalable2017`, disagree where the training data does not pin them down,
which is epistemic uncertainty. Each model's runs are a distribution, and the models are a sample of
such distributions: a second-order distribution, whose variance splits by the law of total variance,

.. math::

   \underbrace{\operatorname{Var}(Y)}_{\text{total}}
   = \underbrace{\mathbb{E}_m\left[\operatorname{Var}(Y \mid m)\right]}_{\text{aleatoric}}
   + \underbrace{\operatorname{Var}_m\left(\mathbb{E}[Y \mid m]\right)}_{\text{epistemic}},

where :math:`m` is a model and :math:`Y` the 10 m wind speed at a place and time.

``probly``'s WeatherNext integration wraps a WeatherNext model as a predictor that draws a fresh run
with every call, and ``representer`` collects its runs, one sample per variable. The split is
``probly``'s ``SecondOrderVarianceDecomposition``, with every model's runs as a Gaussian of their
mean and variance:

.. code-block:: python

   import numpy as np

   from probly.integrations.weathernext import WeatherNextPredictor
   from probly.quantification.decomposition.variance import SecondOrderVarianceDecomposition
   from probly.representation.distribution.numpy_gaussian import (
       NumpyGaussianDistribution,
       NumpyGaussianDistributionSample,
   )
   from probly.representer import representer

   # every model as a predictor; ensemble_fn rolls out the runs of the seeds it is given
   # (WeatherNext's own code, in compute/forecast.py)
   models = [WeatherNextPredictor(ensemble_fn=ensemble_of(params)) for params in checkpoints]
   runs = [representer(model, num_samples=8).represent(inputs, targets_template, forcings) for model in models]

   # the 10 m wind of every run: (models, runs, time, lat, lon)
   wind = np.stack(
       [np.hypot(r["10m_u_component_of_wind"].array, r["10m_v_component_of_wind"].array) for r in runs]
   )

   # every model's runs as a Gaussian, the models as a sample of them
   gaussians = NumpyGaussianDistribution(mean=wind.mean(axis=1), var=wind.var(axis=1))
   split = SecondOrderVarianceDecomposition(NumpyGaussianDistributionSample(array=gaussians, sample_axis=0))
   split.aleatoric  # the mean variance within a model: the runs scatter as the atmosphere might
   split.epistemic  # the variance of the models' means: the models disagree

The demo shows the square root of the total variance as the spread, and the epistemic share of the
variance as the color between blue and red. Every run's track is the lowest sea-level pressure
near its last position, and a run comes ashore where its track first crosses Florida's coast.

Run It Again
------------

The scripts that computed the demo's data are in
`docs/source/demos/milton/compute <https://github.com/pwhofman/probly/tree/main/docs/source/demos/milton/compute>`__.
Each downloads WeatherNext's public initial conditions and checkpoints, rolls out the runs through
``probly``'s WeatherNext integration, and writes the demo's data; its docstring says what to install
next to ``probly``:

.. code-block:: bash

   python docs/source/demos/milton/compute/run_mini.py  # Mini: about 3 hours on four CPU cores
   python docs/source/demos/milton/compute/run_full.py  # full: a GPU with about 80 GB of memory

The Mini checkpoints run on a CPU, in about 12 minutes per run of 5 days on four cores, and in
minutes on a GPU; the full checkpoints need a GPU with about 80 GB of memory, or a TPU. The
coastlines are from `Natural Earth <https://www.naturalearthdata.com>`__.
