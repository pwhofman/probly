Hurricane Milton: Where the Uncertainty Comes From
==================================================

How sure is a weather model of a hurricane's path, and why? This demo shows Google DeepMind's
WeatherNext forecasting Hurricane Milton from 7 October 2024, 00 UTC, three days before it came
ashore in Florida, and splits the uncertainty of its runs, with ``probly``, into what varies from
run to run of one model and what the models do not know.

.. probly-demo:: milton
   :fallback: demos/milton.png
   :alt: The runs of WeatherNext's forecast of Hurricane Milton over the Gulf of Mexico, with their
         uncertainty split into the part that varies from run to run (aleatoric, blue) and the part
         where the models disagree (epistemic, red).

What You See
------------

**The forecast.** WeatherNext 2 :cite:`aletSkillfulJoint2025` and WeatherNext Cyclones
:cite:`aletOperationalTropical2026` forecast the weather of the whole globe in steps of 6 hours. Each
is four models, trained independently, and every run of a model draws random noise of its own. The
demo shows WeatherNext Cyclones at 0.25°, with 8 runs of each of its four models, all trained on
data from before 2024, so none of them has seen Milton.

**The views.**

- *One run* shows a single run as if it were the forecast: one track, one landfall, and no hint of
  how sure it is.
- *All runs* shows every run's track, and a strip along Florida's Gulf coast shows where the runs
  come ashore, each run's dot filled once it has; *Decomposition* shows it too.
- *Decomposition* splits the uncertainty of the runs: blue where the runs of one model scatter
  (aleatoric), red where the models disagree (epistemic). Together, they make up the total
  uncertainty.
- *Better models* shows what would change if the four models agreed: the epistemic part would go,
  as every model's runs move onto the models' common average, which keeps how they scatter. The
  aleatoric part stays, and a forecast has to plan for its range. Models that agree are not
  necessarily right, though: a bias they share stays.

**Time and places.** Play the four days or drag through them, and pick a place, or click the map, to
see it over time. *Show wind* draws where the run, or any run, has tropical-storm-force wind (34 kt):
in *One run* shaded by how strong it is, and in *All runs* by the share of runs that have it at that
time. *Show Milton's track* draws its observed track, from IBTrACS :cite:`knappInternationalBest2010`.

**The wind is the model's, not the storm's.** The wind is the 10 m wind of the model's grid cells
(0.25°, shown at 0.5°) every 6 hours. It is far below the storm's observed sustained wind near the
center, so the share of runs with 34-kt wind is not the National Hurricane Center's wind speed
probability, which is for the strongest wind over a whole period.

How It Was Computed
-------------------

WeatherNext as a Second-Order Predictor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

WeatherNext models its two kinds of uncertainty with two mechanisms :cite:`aletSkillfulJoint2025`.
Within a model, every run draws a small noise vector at every step, a perturbation of the model's
function: the runs of one model scatter as the model has learned the atmosphere might, which is
aleatoric uncertainty.
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

   # every model as a predictor; ensemble_fn rolls out the runs of the seeds it is given, with
   # WeatherNext's own code and a model's forward pass (apply) and weights (compute/forecast.py)
   models = [WeatherNextPredictor(ensemble_fn=ensemble_of(apply, params)) for params in checkpoints]
   runs = [representer(model, num_samples=8).represent(inputs, targets_template, forcings) for model in models]

   # the 10 m wind of every run: (models, runs, time, lat, lon)
   wind = np.stack(
       [np.hypot(r["10m_u_component_of_wind"].array, r["10m_v_component_of_wind"].array) for r in runs]
   )

   # every model's runs as a Gaussian, the models as a sample of them; at the start, all runs are the
   # same analysis, and a Gaussian takes no zero variance, so the variance gets a tiny floor
   gaussians = NumpyGaussianDistribution(mean=wind.mean(axis=1), var=np.maximum(wind.var(axis=1), 1e-9))
   split = SecondOrderVarianceDecomposition(NumpyGaussianDistributionSample(array=gaussians, sample_axis=0))
   split.aleatoric  # the mean variance within a model: the runs of a model scatter
   split.epistemic  # the variance of the models' means: the models disagree

The demo shows the square root of the total variance, the runs' standard deviation, as the total
uncertainty, and the epistemic share of the variance as the color between blue and red. Every run's
track is the lowest sea-level pressure near its last position, and a run, or Milton, comes ashore
where its track first crosses Florida's coast. Between the 6-hourly steps, the page blends the
fields of the steps before and after, rather than moving the storm.

What the Split Can Say
^^^^^^^^^^^^^^^^^^^^^^

**Some epistemic share comes by chance.** Eight runs pin a model's average down only roughly, so the
models' averages differ even if the models were the same. With :math:`M` models of :math:`n` runs
each, the epistemic share of the variance is then about :math:`(M-1)/(M(n-1)+M-1)`, here 3/31, about
10%. Read epistemic shares near 10% as models that agree; for Milton, the epistemic share is only a
little higher, so the four models mostly agree, and most of the uncertainty is aleatoric. More runs
per model would lower that floor.

**The same seeds.** Every model draws the runs of the seeds 0 to 7, the same noise. Runs of different
models with the same seed do not correlate more than runs with different seeds, so this does not
make the models agree more than they do.

**Four models are few, and alike.** The aleatoric part is each model's own estimate of the spread,
from one shared analysis of the weather at the start. The four models share an architecture and
training data, so their disagreement is a lower bound on what they do not know: a bias they share,
such as the wind that is too weak near the center, does not show as epistemic uncertainty.

Run It Again
------------

The scripts that computed the demo's data are in :source:`docs/source/demos/milton/compute`.
``run_full.py`` downloads WeatherNext's public initial conditions and checkpoints, rolls out the runs
through ``probly``'s WeatherNext integration, and writes the demo's data; its docstring says what to
install next to ``probly``:

.. code-block:: bash

   python docs/source/demos/milton/compute/run_full.py --check  # two steps of one run, to test the setup
   python docs/source/demos/milton/compute/run_full.py

The checkpoints need a GPU with about 80 GB of memory, or a TPU. The coastlines are from
`Natural Earth <https://www.naturalearthdata.com>`__.
