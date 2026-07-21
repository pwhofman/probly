# Stacking Experiment (paper §5.4)

A playground for composing probly's uncertainty-quantification primitives
on top of two interchangeable datasets. Each composition is one
self-contained script under `scripts/`; the package only exposes shared,
dataset-agnostic helpers.

## Quick start

```bash
cd experiments/stacking
uv sync -p 3.13

# (optional, one-time, requires GPU + network) build CIFAR-10-H embedding cache
uv run python scripts/cache_cifar10h_embeddings.py --encoder siglip2

# run the headline composition on either dataset
uv run python scripts/stack_dare_temp_conformal.py --dataset two_moons
uv run python scripts/stack_dare_temp_conformal.py --dataset cifar10h --encoder siglip2

# tests
uv run pytest -v
```

## Dataset contract

Every dataset is a frozen `Dataset` dataclass (`src/stacking/data.py`):

```python
@dataclass(frozen=True)
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    in_features: int
    num_classes: int
    meta: dict[str, Any]
```

Splits are constructed inside the loader; scripts never split. Scripts
read `in_features` and `num_classes` to size their model. The `meta` dict
carries dataset-specific extras (encoder name, soft labels, vote counts,
class names) that scripts that know they are on CIFAR-10-H may consult by
key.

Two loaders ship in v1, both via the `load_dataset(name, **kwargs)`
dispatcher:

| name | what | source |
|---|---|---|
| `two_moons` | sklearn 2-D synthetic; 2-class | `make_moons` + two stratified `train_test_split`s |
| `cifar10h` | cached image embeddings; 10-class | `probly.datasets.torch.CIFAR10H` test set (with soft labels) for calib+test; CIFAR-10 train as the train side |

## Encoders (for `cifar10h`)

Two encoders are supported via the lazy `ENCODERS` registry in
`src/stacking/embed.py`:

- `siglip2` — `google/siglip2-base-patch16-256`, pooled image embedding
- `dinov2_with_registers` — `facebook/dinov2-with-registers-base`, CLS-token embedding

Cache files are written to `cache/<dataset>_<encoder>_<split>.npz` and
the cache builder is idempotent. Pass `--force` to re-encode.

## The "swap recipe"

Adding a new composition = copy `scripts/stack_dare_temp_conformal.py`
to `scripts/stack_<new_name>.py` and edit the middle.

- Swap the **base layer**: replace `dare(base, num_members=...)` with
  `mc_dropout(base)`, `subensemble(base, ...)`, `sngp(base, ...)`,
  `bayesian(base, ...)`, `evidential(base, ...)`, etc.
- Swap the **calibration layer**: replace `temperature_scaling(...)`
  with `platt_scaling(...)`, `vector_scaling(..., num_classes=...)`,
  or `isotonic_regression(...)`.
- Swap the **conformal layer**: replace `conformal_raps(...)` with
  `conformal_lac(...)`, `conformal_aps(...)`, or `conformal_saps(...)`.
- **Drop a layer entirely**: skip the calibration step and feed pooled
  logits directly into the conformal layer.

Each new composition imports only what it needs from probly.

## What is deliberately not here

By design, the playground does **not** ship:

- A second composition script — those are added when you actually want
  to play with a new combination.
- A `bases/` directory or a "stack builder" — composition logic lives in
  scripts; the package only ships data, model, and embedding helpers.
- A training utility (`train.py`). Each composition writes its own loop;
  if two scripts duplicate one, the duplication is the point.
- Plotting code. v1 prints metrics to stdout. Plotting is added per
  script when a specific figure is wanted.
- Multi-seed sweeps / parquet pipelines. `experiments/river_uncertainty/`
  fills that pattern; this experiment is deliberately a different shape.

See `docs/superpowers/specs/2026-05-03-stacking-experiment-playground-design.md`
for the full design rationale.
