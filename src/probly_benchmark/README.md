# probly_benchmark

## Running Benchmark Experiments

TODO

## Active Learning Experiments

Entry point: `probly_benchmark.active_learning`

### Single run

```bash
python -m probly_benchmark.active_learning \
    method=ensemble \
    al_strategy=uncertainty \
    al_dataset=cifar10
```

Override defaults as needed:

```bash
python -m probly_benchmark.active_learning \
    method=dropout al_strategy=margin al_dataset=fashion_mnist \
    seed=42 n_iterations=5 epochs=20 device=cpu wandb.enabled=false
```

### SLURM sweep (all methods x datasets x seeds)

```bash
# UQ methods
python -m probly_benchmark.active_learning --multirun \
    method=ensemble,dropout,credal_ensembling,credal_relative_likelihood,efficient_credal_prediction,evidential_classification,ddu,posterior_network \
    al_strategy=uncertainty \
    al_dataset=openml_6,openml_155,openml_156,fashion_mnist,cifar10 \
    seed=0,1,2,3,4,5,6,7,8,9 \
    hydra/launcher=submitit_slurm

# Override quantifier (e.g. mutual information for ensembles)
python -m probly_benchmark.active_learning --multirun \
    method=ensemble \
    al_strategy=uncertainty \
    al_dataset=cifar10 \
    quantifier=mutual_information \
    seed=0,1,2,3,4,5,6,7,8,9 \
    hydra/launcher=submitit_slurm
```

### Local smoke test (no WandB, CPU only)

```bash
uv run python -m probly_benchmark.active_learning \
    method=ensemble \
    al_strategy=uncertainty \
    al_dataset=fashion_mnist \
    wandb.enabled=false \
    initial_size=100 \
    query_size=100 \
    n_iterations=2 \
    epochs=1 \
    device=cpu
```

### Config structure

| Config group   | Options |
|----------------|---------|
| `method`       | `ensemble`, `dropout`, `credal_ensembling`, `credal_relative_likelihood`, `efficient_credal_prediction`, `evidential_classification`, `ddu`, `posterior_network` |
| `al_strategy`  | `uncertainty`, `margin`, `badge`, `random` |
| `al_dataset`   | `cifar10`, `fashion_mnist`, `openml_6`, `openml_155`, `openml_156` |
| `quantifier`   | Override: `entropy_of_expected_value`, `mutual_information`, `entropy`, `upper_entropy`, `ddu_entropy` (default per method) |

Results are logged to WandB (project: `probly-active-learning`). Disable with `wandb.enabled=false`.
