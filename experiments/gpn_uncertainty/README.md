# Graph Posterior Network Demo

Small, self-contained node-classification demo for probly's graph posterior network methods.

It trains three models on a synthetic stochastic-block graph with intentionally ambiguous bridge nodes:

- GPN: density evidence is propagated over the graph.
- LOP-GPN: local feature Dirichlets are pooled through graph propagation weights.
- CUQ-GNN: hidden node features are graph-refined before density evidence is computed.

The script writes selective-prediction curves and a graph visualization showing predictions and uncertainty.

## Quick Start

```bash
uv sync -p 3.13
uv run python scripts/run_gpn_demo.py
```

Fast smoke run:

```bash
uv run python scripts/run_gpn_demo.py --epochs 5 --nodes-per-class 12
```

Outputs are written to `results/`:

- `selective_prediction.pdf`: loss after rejecting increasingly uncertain test nodes.
- `graph_uncertainty.pdf`: graph layout with predicted class color, epistemic uncertainty size, and classification errors shown in red.
- `metrics.json`: accuracy and selective-prediction area for each method.
