# river-uncertainty - Phase 1 experiment

A small sandbox living next to the probly package (in `experiments/`, **not**
under `src/probly/`). The goal is to see how naturally a river online
ensemble - `river.forest.ARFClassifier`, the Adaptive Random Forest - plugs
into probly's second-order uncertainty representations.

It is a standalone uv project with its own `pyproject.toml`; it depends on
probly via an editable local path pointing at the repo root (`../..`).

## TL;DR

1. The natural probly representation for an ensemble of per-tree categorical
   distributions is `ArrayCategoricalDistributionSample` with `sample_axis=0`
   (each row is one tree's categorical distribution).
2. With that representation in hand, probly's existing
   `SecondOrderEntropyDecomposition` and `SecondOrderZeroOneDecomposition`
   give us aleatoric / epistemic / total uncertainty "for free" via
   `quantify`.
3. Per-tree weights (ARF's internal accuracy metric per learner) can be
   carried alongside the representation so we can reconstruct ARF's weighted
   BMA. We do **not** apply the weights to the stacked array because we want
   probly quantifiers to see an unbiased empirical distribution, not a
   weight-collapsed one.

## Layout

```
experiments/river_uncertainty/
├── pyproject.toml                   # uv project; pulls probly as editable
├── src/river_uncertainty/
│   ├── __init__.py
│   ├── stream.py                    # make_synthetic_stream(kind, n, seed)
│   ├── representation.py            # ARFEnsembleRepresentation + bridge
│   ├── experiment.py                # prequential loop + PrequentialTrace
│   └── plotting.py                  # rolling_mean helper
├── experiments/
│   ├── 01_stream_response.py        # Level 1
│   ├── 02_ensemble_size_ablation.py # Level 2
│   ├── 03_uncertainty_drift_detector.py  # Level 3
│   └── run_all.py                   # convenience: run all three in sequence
├── tests/
│   └── test_representation.py
└── results/                         # plots, npz dumps, csv tables
```

## Quick start

```bash
cd experiments/river_uncertainty
uv sync -p 3.13
uv run pytest                                     # 4 tests, should all pass
uv run python experiments/run_all.py              # reproduce all three figures
# ...or run them individually:
uv run python experiments/01_stream_response.py
uv run python experiments/02_ensemble_size_ablation.py
uv run python experiments/03_uncertainty_drift_detector.py
```

Each level script has an "easy-to-tweak settings" block at the top (constants
like `N_MODELS`, `SEEDS`, `STREAM_KIND`, etc.). The committed defaults
reproduce the findings below. Plots, npz dumps, and CSVs land in `results/`.

## The bridge, in one picture

```
river ARFClassifier
   │
   ├─► [tree 0].predict_proba_one(x) = {cls: p}
   ├─► [tree 1].predict_proba_one(x) = {cls: p}
   │    ...
   └─► [tree n-1].predict_proba_one(x) = {cls: p}

            ▼   align to common class order, normalise rows

   numpy array of shape (n_trees, n_classes)

            ▼

   ArrayCategoricalDistribution (shape (n_trees, n_classes))

            ▼   wrap as a Sample along axis 0

   ArrayCategoricalDistributionSample     ◄── this IS an empirical
                                                second-order distribution

            ▼   probly.quantification.quantify(...)

   SecondOrderEntropyDecomposition
         .total     = H[E_{theta~Q}[theta]]   (entropy of the BMA)
         .aleatoric = E[H[theta]]             (conditional entropy)
         .epistemic = MI(Y; theta)            (mutual information)
```

The same sample object flows through `SecondOrderZeroOneDecomposition` for
the alternative (bounded) zero-one decomposition.

## Three levels of investigation

### Level 1 - Stream-response survey

*File: `experiments/01_stream_response.py` → `results/level1_stream_response.png`*

Train an identical ARF (`n_models=15`, seed `42`) on three contrasting
streams and watch how the decomposition reacts:

| stream                    | regime                      | what to expect               |
|---------------------------|-----------------------------|------------------------------|
| `agrawal`                 | stationary, 9-feature noisy | baseline shape of components |
| `stagger_drift` (0→2)     | abrupt, noise-free          | textbook epistemic spike     |
| `sea_drift` (0→3, width 30) | abrupt, noisy              | both alea + epi rise         |

Selected numbers (mean over a window just before and just after the drift):

```
stream                  acc    total     alea      epi
agrawal (tail)         0.978   0.506   0.253    0.253
stagger_drift (pre)    0.950   0.065   0.027    0.038
stagger_drift (post)   0.800   0.429   0.252    0.177   <- ~5x epi spike
sea_drift (pre)        0.980   0.197   0.085    0.112
sea_drift (post)       0.870   0.242   0.107    0.135
```

Observations:

- **STAGGER** is the cleanest: epistemic entropy jumps ~5× on the drift,
  relaxes to near zero once ARF has swapped trees.
- **SEA** is harder - both aleatoric and epistemic move, accuracy drops 11%.
  The aleatoric rise is the empirical-sample estimator flagging that trees
  are "confidently wrong" on many samples.
- **Agrawal** (stationary) shows that epistemic does *not* decay to zero.
  ARF uses `max_features="sqrt"` feature subsampling, so trees make
  systematically different trade-offs and keep disagreeing. That is a
  feature of the quantifier, not a bug.

### Level 2 - Ensemble-size ablation

*File: `experiments/02_ensemble_size_ablation.py` → `results/level2_ensemble_size_ablation.{png,csv}`*

Sweep `n_models ∈ {3, 5, 10, 20, 40}` over `SEEDS = (0..4)` on STAGGER drift.
For each combination we measure window-mean accuracy and epistemic entropy
pre- and post-drift.

Headline numbers (mean across seeds):

| n_models | acc pre | acc post | epi pre | epi post | spike |
|----------|---------|----------|---------|----------|-------|
| 3        | 0.981   | 0.796    | 0.033   | 0.127    | 0.094 |
| 5        | 0.981   | 0.812    | 0.021   | 0.175    | 0.154 |
| 10       | 0.981   | 0.806    | 0.025   | 0.177    | 0.152 |
| 20       | 0.981   | 0.822    | 0.031   | 0.168    | 0.138 |
| 40       | 0.981   | 0.824    | 0.040   | 0.192    | 0.151 |

Take-aways:

- **Accuracy is essentially flat** past `n_models=5` - ARF is already close
  to the ceiling on STAGGER.
- The **epistemic spike** plateaus quickly. With only 3 trees the spike is
  ~40% smaller than with 10+, and its seed-to-seed variance is huge: one
  seed actually goes *negative* (a single tree happened to dominate the
  post-drift vote). With `n_models >= 10` the spike is consistently in the
  0.14-0.18 nats range.
- The **cost of a small ensemble is variance, not bias** - the mean spike
  is similar, but 3 trees gives a useless estimator of the spike.

Practical rule of thumb: use `n_models >= 10` if you want the epistemic
uncertainty to be informative at the sample level.

### Level 3 - Uncertainty-based drift detection

*File: `experiments/03_uncertainty_drift_detector.py` → `results/level3_uncertainty_drift_detector.{png,npz}`*

A thresholding drift detector that **only sees the ensemble state**, not the
labels: estimate a baseline ``mu ± sigma`` of smoothed epistemic entropy on
the window `[500, 1500)`, then flag a drift the first time the rolling
epistemic entropy rises above ``mu + 4·sigma`` for 5 consecutive steps (all
tunable at the top of the script).

We compare the detection latency with ARF's own internal ADWIN-based drift
detector (which *does* see the labels via the `metric` update).

Result (STAGGER, seed 42, true drift at step 2000):

```
Baseline epistemic stats (window (500, 1500)): mu=0.0410, sigma=0.0112
Threshold mu + 4.0*sigma = 0.0858
Uncertainty detector fired at step: 2004  (latency  +4 samples)
First ARF drift after truth      : 2016  (latency +16 samples)
```

The uncertainty-only detector fires **12 samples earlier** than ARF's
internal detector on this seed. This is not a general claim about
superiority - ARF's detectors are per-tree and reset subtree growth, they
are doing more than just fire - but it strongly suggests that the second-order
representation is a real, usable drift signal.

Things to try from here (all one-line edits in the script):

- Change `STREAM_KIND` to `sea_drift` (harder, noisier): the detector still
  fires quickly but with more variance; tune `K_SIGMA`.
- Lower `K_SIGMA` (e.g. 3.0): catches drift earlier on STAGGER, at the cost
  of more false positives on Agrawal. Try it - the stationary-Agrawal epistemic
  signal has no outliers, so you get a clean negative example.
- Increase `MIN_CONSECUTIVE`: trades latency for robustness; useful on
  `sea_drift`.

## Things I learned about probly while doing this

- `quantify` is a `lazydispatch` dispatch on the representation's type, so
  wiring a new source of categorical distributions is literally "produce an
  `ArrayCategoricalDistributionSample` and hand it to `quantify`".
- `ArrayCategoricalDistribution` accepts unnormalised probabilities and
  normalises internally. Handy: we can drop a `(n_trees, n_classes)` array
  straight in without pre-normalising.
- Both entropy and zero-one decompositions inherit `AdditiveDecomposition`,
  so for a single API call we get cached access to `.total`, `.aleatoric`,
  `.epistemic` without re-computing shared intermediates.

## Phase 2 ideas (not implemented)

If we want to make this a first-class probly feature, a natural API would be:

```python
# Hypothetical
from probly.representation.river import RiverEnsembleDistributionSample

rep = RiverEnsembleDistributionSample.from_classifier(arf, x)
# rep is an ArrayCategoricalDistributionSample subclass that also stores
# .classes and .weights; all probly quantifiers continue to work unchanged.

decomp = probly.quantification.quantify(rep)
```

Open questions:

- **Weighted second-order distributions.** Probly currently assumes the
  empirical sample is uniform. ARF, SRP, Leveraging Bagging and friends all
  produce *weighted* ensembles. Adding a `weights` field to
  `Sample`/`ArraySample` (or a new `WeightedSample` representation) would let
  probly cover the whole river ensemble family without bespoke wrappers.
- **Regressors.** For `ARFRegressor` the natural probly representation is a
  Gaussian mixture (one Gaussian per tree using each tree's residual
  variance) or a Dirac-mixture `DistributionSample[ArrayGaussian]`.
- **Concept-drift detection via uncertainty.** The experiments suggest
  epistemic spikes are often faster than accuracy drops and, in at least one
  setting (Level 3), they precede ARF's own drift detections. Worth a proper
  study.
- **Online conformal.** Probly already has `conformal_prediction`; plugging
  an online conformalizer on top of the ensemble BMA is a third Phase 2 thread.
