# Graph Posterior Network Demo

Small node-classification demos for probly's graph posterior network methods.

It trains three models on a synthetic stochastic-block graph with intentionally ambiguous bridge nodes:

- GPN: density evidence is propagated over the graph.
- LOP-GPN: local feature Dirichlets are pooled through graph propagation weights.
- CUQ-GNN: hidden node features are graph-refined before density evidence is computed.

The script writes selective-prediction curves and graph visualizations showing predictions and uncertainty.

## Quick Start

```bash
uv sync -p 3.13
uv run python scripts/run_gpn_demo.py
```

Amazon Photos run:

```bash
uv run python scripts/run_gpn_demo.py --experiment amazon-photo
```

Run both demos:

```bash
uv run python scripts/run_gpn_demo.py --experiment all
```

Use CUDA for model training and inference:

```bash
uv run python scripts/run_gpn_demo.py --experiment amazon-photo --device cuda
```

CUDA runs automatically use float16 mixed precision for training and inference.

Force retraining instead of loading cached model weights:

```bash
uv run python scripts/run_gpn_demo.py --experiment amazon-photo --retrain
```

Fast smoke run:

```bash
uv run python scripts/run_gpn_demo.py --epochs 5 --nodes-per-class 12
```

Outputs are written to `results/`:

- `selective_prediction.pdf`: accuracy after rejecting increasingly uncertain test nodes.
- `graph_uncertainty.pdf`: graph layout with predicted class color, epistemic uncertainty size, and classification errors shown in red.
- `metrics.json`: accuracy and selective-prediction area for each method.
- `amazon_photo_selective_prediction.pdf`: Amazon Photos selective-prediction curves.
- `amazon_photo_graph_uncertainty.pdf`: Amazon Photos ForceAtlas2 graph visualization.
- `amazon_photo_metrics.json`: Amazon Photos accuracy and selective-prediction area.

Amazon Photos uses `torch_geometric.datasets.Amazon(name="Photo")`. The first run downloads the data, trains all three models, and computes a deterministic ForceAtlas2 layout. Later runs reuse cached model weights from `checkpoints/` and cached positions from `cache/`. Increase `--amazon-epochs` or `--forceatlas2-iterations` for longer runs.
