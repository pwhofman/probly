# Plan: Integrate River ARF into Probly's Ensemble Dispatch Chain

> Status: **Not started** — designed 2026-04-21, ready for implementation.

## Goal

Move the river ARF bridge from the standalone experiment (`experiments/river_uncertainty/`) into probly's existing ensemble dispatch infrastructure (`src/probly/method/ensemble/`), so that `representer(arf).represent(x)` produces an `ArrayCategoricalDistributionSample` through the standard dispatch chain — just like sklearn, torch, and flax ensembles.

## Context for a New Session

### Where to look first

| What | Path |
|------|------|
| **Reference implementation** (what we're moving) | `experiments/river_uncertainty/src/river_uncertainty/representation.py` |
| **Existing ensemble backends** (pattern to follow) | `src/probly/method/ensemble/{sklearn,torch,flax}.py` |
| **Ensemble dispatch core** | `src/probly/method/ensemble/_common.py` |
| **Lazy types for optional backends** | `src/probly/lazy_types.py` |
| **Predictor protocols + predict dispatch** | `src/probly/predictor/_common.py` |
| **Sample creation dispatch** | `src/probly/representation/sample/_common.py` (line 142: `create_sample`) |
| **ArraySample.from_iterable** | `src/probly/representation/sample/array.py` (line 62) |
| **ArrayCategoricalDistribution + Sample** | `src/probly/representation/distribution/array_categorical.py` |
| **SecondOrderEntropyDecomposition** | `src/probly/quantification/decomposition/entropy/_common.py` |
| **Experiment README** | `experiments/river_uncertainty/README.md` |

### Key architecture concepts

- **probly uses `lazydispatch`** (custom singledispatch variant) for type-based dispatch throughout: `predict`, `predict_raw`, `representer`, `create_sample`, `quantify` are all lazydispatch functions.
- **Optional backends use `delayed_register`**: a callback fires on first encounter of a lazy type (string path), importing the backend module which registers actual handlers. See `method/ensemble/__init__.py` for examples.
- **`lazy_types.py`** maps string type paths (e.g. `"sklearn.base.BaseEstimator"`) for delayed registration.
- **Representations are frozen dataclasses**. `ArrayCategoricalDistributionSample` extends `ArraySample[ArrayCategoricalDistribution]` and must have an `ArrayCategoricalDistribution` as its `.array` (not a plain ndarray) for `DistributionSample.__instancehook__` to match.
- **`DistributionSample.__instancehook__`** checks `isinstance(instance.samples, cls.sample_space)` — so the sample's inner array must remain a `Distribution`, not be unwrapped to numpy.

### The existing ensemble dispatch chain

```
representer(ensemble_predictor)                # IterablePredictor → IterableSampler
  sampler.represent(X)
    predict(ensemble, X)                       # dispatches to predict_raw
      predict_raw(EnsemblePredictor, X)        # [predict(member, X) for member]
        → list[ArrayCategoricalDistribution]
    create_sample(list[ACD], sample_axis=-1)   # dispatches on first element
      → ArraySample.from_iterable(...)         # !! converts dists to ndarray, loses type
      → ArraySample[np.ndarray]                # not a DistributionSample!
```

### River ARF specifics

- River's `ARFClassifier` is at `river.forest.adaptive_random_forest.ARFClassifier`
- Per-tree predictions: `tree.predict_proba_one(x) → dict[class, prob]` (not `predict_proba(X) → ndarray`)
- Online class discovery: different trees may have seen different classes — need alignment to a common class order before stacking
- ARF is ALREADY an ensemble (it creates its own trees) — we don't need `ensemble_generator`, just `predict_raw` registration

---

## The Two Problems and Solutions

### Problem 1: `create_sample` loses distribution type (GENERAL FIX)

`create_sample(list[ArrayCategoricalDistribution])` routes to `ArraySample.from_iterable` which calls `to_numpy_array_like()` on each element, unwrapping to plain `np.ndarray`. The result `ArraySample[np.ndarray]` doesn't match `DistributionSample`, so `quantify()` can't dispatch to `SecondOrderEntropyDecomposition`.

**Fix**: Register `create_sample` specifically for `ArrayCategoricalDistribution`:

**File**: `src/probly/representation/distribution/array_categorical.py`

```python
from probly.representation.sample._common import create_sample, SampleAxis

@create_sample.register(ArrayCategoricalDistribution)
def _create_array_categorical_distribution_sample(
    samples: Iterable[ArrayCategoricalDistribution],
    sample_axis: SampleAxis = "auto",
    **_kwargs,
) -> ArrayCategoricalDistributionSample:
    prob_arrays = [s.unnormalized_probabilities for s in samples]
    if sample_axis == "auto":
        sample_axis = -1
    stacked = np.stack(prob_arrays, axis=sample_axis)
    dist = ArrayCategoricalDistribution(unnormalized_probabilities=stacked)
    if sample_axis < 0:
        sample_axis = stacked.ndim + sample_axis
    return ArrayCategoricalDistributionSample(array=dist, sample_axis=sample_axis)
```

This benefits ALL ensemble backends (sklearn, torch, etc.), not just river.

### Problem 2: ARF needs predict_raw + type registration (RIVER-SPECIFIC)

**File**: `src/probly/lazy_types.py` — add:
```python
RIVER_ARF = "river.forest.adaptive_random_forest.ARFClassifier"
```

**File**: `src/probly/method/ensemble/river.py` — NEW, following `sklearn.py` pattern:

```python
"""River ARF ensemble integration."""
from river.forest import ARFClassifier
import numpy as np

from probly.predictor import predict_raw
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution
from ._common import EnsembleCategoricalDistributionPredictor

# Register ARF as ensemble categorical predictor → representer gives IterableSampler
EnsembleCategoricalDistributionPredictor.register(ARFClassifier)

@predict_raw.register(ARFClassifier)
def predict_arf_ensemble(arf: ARFClassifier, x: dict[str, float]):
    """Extract aligned per-tree categorical distributions from an ARF."""
    per_tree_dicts = [m.predict_proba_one(x) for m in arf.models]

    # Infer common class order
    seen: set = set()
    for pt in per_tree_dicts:
        seen.update(pt)
    seen.update(arf.predict_proba_one(x))
    classes = sorted(seen, key=lambda c: (not isinstance(c, (int, float)), str(c)))
    if not classes:
        classes = [0]

    class_idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)

    result = []
    for pt in per_tree_dicts:
        probs = np.zeros(k, dtype=np.float64)
        total = sum(pt.values())
        if total <= 0:
            probs[:] = 1.0 / k
        else:
            for cls, p in pt.items():
                probs[class_idx[cls]] = p / total
        result.append(ArrayCategoricalDistribution(unnormalized_probabilities=probs))
    return result
```

**File**: `src/probly/method/ensemble/__init__.py` — add delayed_register:

```python
from probly.lazy_types import RIVER_ARF
from probly.predictor import predict_raw

@predict_raw.delayed_register(RIVER_ARF)
def _(_: type) -> None:
    from . import river as river
```

**File**: `pyproject.toml` — add dependency group:

```toml
river = ["river>=0.21"]
```

And include in `all_ml`.

---

## Target User Flow After Implementation

```python
from river.forest import ARFClassifier
from probly.representer import representer
from probly.quantification import quantify

arf = ARFClassifier(n_models=15, seed=42)
for x, y in training_stream:
    arf.learn_one(x, y)

sample = representer(arf).represent(x)   # ArrayCategoricalDistributionSample
decomp = quantify(sample)                # SecondOrderEntropyDecomposition
print(decomp.aleatoric, decomp.epistemic)
```

## Implementation Steps Summary

1. **Fix `create_sample`** for `ArrayCategoricalDistribution` → `ArrayCategoricalDistributionSample` (general fix in `array_categorical.py`)
2. **Add `RIVER_ARF`** to `lazy_types.py`
3. **Create `method/ensemble/river.py`** — register `EnsembleCategoricalDistributionPredictor` + `predict_raw`
4. **Wire `delayed_register`** in `method/ensemble/__init__.py`
5. **Add `river` dependency group** to `pyproject.toml`
6. **Add tests** in `tests/probly/method/ensemble/test_river.py` (follow `test_sklearn.py` pattern) and test the `create_sample` fix
7. **Update experiment** to import from probly; rerun to verify same results

## What This Does NOT Cover (follow-up work)

- Deep ensemble / MC-dropout bridges (same pattern, different `predict_raw`)
- Weighted second-order distributions (broader probly feature)
- `ensemble_generator` for ARF (not needed — ARF creates its own trees)
- `OnlineClassifier` / `DropoutMLP` (remain in experiments)

## Verification

```bash
uv run pytest tests/probly/method/ensemble/test_river.py         # new tests
uv run pytest tests/probly/                                       # no regressions
cd experiments/river_uncertainty && uv run pytest                  # experiment tests
cd experiments/river_uncertainty && uv run python experiments/run_all.py  # reproduce results
uv run prek run --all-files                                       # linting
```
