# Experiments

Practical experiments showcasing `probly` in ML applications. Each experiment
is a self-contained project with its own dependencies.

All results are written to `data/results/experiments/<name>/` at the repo root.

## Experiments

| Name | Description |
|------|-------------|
| [gemma](gemma/) | Semantic entropy & calibration with Gemma 4 |
| [river_uncertainty](river_uncertainty/) | Online learning uncertainty with River |

## Running an experiment

```bash
cd experiments/<name>
uv sync
uv run python experiments/...
```

See each experiment's `README.md` for detailed instructions.
