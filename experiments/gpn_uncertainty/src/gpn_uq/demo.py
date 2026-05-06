"""Synthetic graph node-classification demo for GPN variants."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from probly.evaluation.tasks import selective_prediction
from probly.predictor import predict
from probly.quantification import quantify
from probly.train.evidential.torch import mixture_uce_loss, postnet_loss
from probly.method.graph_posterior_network import (
    cuq_graph_neural_network,
    graph_posterior_network,
    lop_graph_posterior_network,
)

plt.rcParams["font.family"] = "DejaVu Sans"

ModelName = Literal["GPN", "LOP-GPN", "CUQ-GNN"]


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation output for one trained model."""

    accuracy: float
    selective_area: float
    coverage: np.ndarray
    selective_loss: np.ndarray
    predictions: np.ndarray
    total_uncertainty: np.ndarray
    epistemic_uncertainty: np.ndarray


def set_seed(seed: int) -> None:
    """Set numpy and torch random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_graph(nodes_per_class: int = 24, seed: int = 0) -> Data:
    """Create a small three-community graph with uncertain bridge nodes.

    Args:
        nodes_per_class: Number of nodes per class/community.
        seed: Random seed.

    Returns:
        A PyG ``Data`` object with features, labels, masks, and 2D positions.
    """
    rng = np.random.default_rng(seed)
    num_classes = 3
    centers = np.array([[0.0, 0.0], [3.0, 0.3], [1.5, 2.7]])
    feature_centers = np.eye(num_classes, dtype=np.float32)
    positions: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[int] = []

    for class_idx in range(num_classes):
        for node_idx in range(nodes_per_class):
            bridge_fraction = node_idx / max(nodes_per_class - 1, 1)
            neighbor_class = (class_idx + 1) % num_classes
            if node_idx >= nodes_per_class - 4:
                mix = 0.45 + 0.1 * rng.random()
                position = (1 - mix) * centers[class_idx] + mix * centers[neighbor_class]
                feature = (1 - mix) * feature_centers[class_idx] + mix * feature_centers[neighbor_class]
            else:
                position = centers[class_idx] + rng.normal(scale=0.35 + 0.15 * bridge_fraction, size=2)
                feature = feature_centers[class_idx] + rng.normal(scale=0.18, size=num_classes)
            positions.append(position.astype(np.float32))
            features.append(feature.astype(np.float32))
            labels.append(class_idx)

    pos = np.vstack(positions)
    x = np.vstack(features)
    y = np.asarray(labels, dtype=np.int64)
    distances = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    edges: set[tuple[int, int]] = set()
    for i in range(len(y)):
        neighbors = np.argsort(distances[i])[1:7]
        for j in neighbors:
            same_class_bonus = 0.35 if y[i] == y[j] else 0.0
            bridge_bonus = 0.18 if i % nodes_per_class >= nodes_per_class - 4 or j % nodes_per_class >= nodes_per_class - 4 else 0.0
            if rng.random() < 0.35 + same_class_bonus + bridge_bonus:
                edges.add((i, int(j)))
                edges.add((int(j), i))
    for source_class in range(num_classes):
        for target_class in range(source_class + 1, num_classes):
            source_idx = np.where(y == source_class)[0]
            target_idx = np.where(y == target_class)[0]
            pair_distances = distances[np.ix_(source_idx, target_idx)]
            for flat_index in np.argsort(pair_distances, axis=None)[:2]:
                source_offset, target_offset = np.unravel_index(flat_index, pair_distances.shape)
                source = int(source_idx[source_offset])
                target = int(target_idx[target_offset])
                edges.add((source, target))
                edges.add((target, source))
    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()

    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    for class_idx in range(num_classes):
        idx = np.where(y == class_idx)[0]
        train_count = min(5, max(2, nodes_per_class // 3))
        val_count = min(3, max(1, nodes_per_class // 4))
        train_mask[idx[:train_count]] = True
        val_mask[idx[train_count : train_count + val_count]] = True
    test_mask = ~(train_mask | val_mask)

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
        pos=torch.tensor(pos, dtype=torch.float),
    )


def build_model(name: ModelName, feature_dim: int, num_classes: int) -> nn.Module:
    """Build one GPN variant.

    Args:
        name: Model variant name.
        feature_dim: Input feature dimension.
        num_classes: Number of target classes.

    Returns:
        A trainable GPN variant.
    """
    encoder_dim = 24
    encoder = nn.Sequential(nn.Linear(feature_dim, encoder_dim), nn.ReLU(), nn.Linear(encoder_dim, encoder_dim), nn.ReLU())
    common: dict[str, int | float] = {
        "encoder_dim": encoder_dim,
        "num_flows": 4,
        "propagation_steps": 8,
        "teleport_probability": 0.12,
    }
    if name == "GPN":
        return graph_posterior_network(encoder, 8, num_classes, **common)
    if name == "LOP-GPN":
        return lop_graph_posterior_network(encoder, 8, num_classes, **common)
    if name == "CUQ-GNN":
        return cuq_graph_neural_network(encoder, 8, num_classes, convolution_name="appnp", **common)
    raise ValueError(name)


def train_model(model: nn.Module, data: Data, name: ModelName, epochs: int, lr: float) -> None:
    """Train a GPN model on the graph train mask."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        if name == "LOP-GPN":
            alpha_features, mixture_weights = model.forward(data)
            loss = mixture_uce_loss(alpha_features, mixture_weights[data.train_mask], data.y[data.train_mask], "mean")
        else:
            alpha = model(data)
            loss = postnet_loss(alpha[data.train_mask], data.y[data.train_mask], entropy_weight=1e-5, reduction="mean")
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate_model(model: nn.Module, data: Data, n_bins: int) -> EvaluationResult:
    """Evaluate accuracy, uncertainty decomposition, and selective prediction."""
    model.eval()
    distribution = predict(model, data)
    probabilities = distribution.mean.probabilities
    predictions = probabilities.argmax(dim=-1)
    decomposition = quantify(distribution)
    total = decomposition.total.detach().cpu().numpy()
    epistemic = decomposition.epistemic.detach().cpu().numpy()
    mask = data.test_mask.cpu().numpy()
    labels = data.y.cpu().numpy()
    pred_np = predictions.cpu().numpy()
    losses = (pred_np[mask] != labels[mask]).astype(float)
    selective_area, selective_loss = selective_prediction(total[mask], losses, n_bins=min(n_bins, int(mask.sum())))
    coverage = np.linspace(1.0, 1.0 / len(selective_loss), len(selective_loss))
    accuracy = float((pred_np[mask] == labels[mask]).mean())
    return EvaluationResult(accuracy, selective_area, coverage, selective_loss, pred_np, total, epistemic)


def plot_selective(results: dict[ModelName, EvaluationResult], output_path: Path) -> None:
    """Plot selective-prediction curves."""
    _, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, result in results.items():
        ax.plot(result.coverage, result.selective_loss, marker="o", linewidth=2, label=f"{name} (AULC={result.selective_area:.3f})")
    ax.set_xlabel("Coverage after rejecting most uncertain nodes")
    ax.set_ylabel("Zero-one loss on retained nodes")
    ax.set_title("Selective Prediction on Ambiguous Graph Nodes")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_graph(data: Data, results: dict[ModelName, EvaluationResult], output_path: Path) -> None:
    """Plot graph predictions and uncertainty for all model variants."""
    pos = data.pos.cpu().numpy()
    edges = data.edge_index.cpu().numpy().T
    labels = data.y.cpu().numpy()
    fig, axes = plt.subplots(1, len(results), figsize=(14.5, 4.75), sharex=True, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")
    colors = np.array(["#3A86FF", "#FFB000", "#5AD08A"])
    edge_color = "#DDE3EA"
    error_color = "#FF2D6D"
    train_color = "#8D96A6"
    train_mask = data.train_mask.cpu().numpy()
    for ax, (name, result) in zip(axes, results.items(), strict=True):
        ax.set_facecolor("#FBFCFF")
        for src, dst in edges[::2]:
            ax.plot(
                [pos[src, 0], pos[dst, 0]],
                [pos[src, 1], pos[dst, 1]],
                color=edge_color,
                linewidth=0.45,
                alpha=0.72,
                zorder=0,
            )
        uncertainty = result.epistemic_uncertainty
        sizes = 18 + 150 * (uncertainty - uncertainty.min()) / max(float(np.ptp(uncertainty)), 1e-6)
        wrong = result.predictions != labels
        correct = ~wrong & ~train_mask
        train = train_mask
        wrong_test = wrong & ~train_mask
        ax.scatter(
            pos[correct, 0],
            pos[correct, 1],
            s=sizes[correct],
            c=colors[result.predictions[correct]],
            alpha=0.86,
            edgecolors="#FFFFFF",
            linewidths=0.65,
        )
        ax.scatter(
            pos[wrong_test, 0],
            pos[wrong_test, 1],
            s=sizes[wrong_test],
            c=error_color,
            alpha=0.92,
            edgecolors="none",
            linewidths=0.0,
        )
        ax.scatter(
            pos[train, 0],
            pos[train, 1],
            s=sizes[train],
            c=train_color,
            alpha=0.9,
            edgecolors="#FFFFFF",
            linewidths=0.65,
        )
        ax.set_title(f"{name}\naccuracy = {result.accuracy:.2f}", fontsize=10.5, fontweight="bold", color="#171A1F", pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_color("#D7DCE3")
            spine.set_linewidth(0.9)
    fig.suptitle(
        "Prediction and Epistemic Uncertainty on a Synthetic Graph",
        fontsize=11.5,
        fontweight="bold",
        color="#171A1F",
        y=0.975,
    )
    fig.text(
        0.5,
        0.035,
        "color = prediction  |  size = epistemic uncertainty  |  red = classification error  |  gray = train data",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#5A6472",
    )
    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.105, top=0.85, wspace=0.14)
    plt.savefig(output_path)
    plt.close()


def run_demo(output_dir: Path, seed: int = 7, nodes_per_class: int = 120, epochs: int = 120, lr: float = 0.01) -> None:
    """Train all variants and write demo plots and metrics."""
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = make_synthetic_graph(nodes_per_class=nodes_per_class, seed=seed)
    num_classes = int(data.y.max().item()) + 1
    results: dict[ModelName, EvaluationResult] = {}
    for name in ("GPN", "LOP-GPN", "CUQ-GNN"):
        model = build_model(name, data.x.shape[-1], num_classes)
        train_model(model, data, name, epochs, lr)
        results[name] = evaluate_model(model, data, n_bins=12)
    plot_selective(results, output_dir / "selective_prediction.pdf")
    plot_graph(data, results, output_dir / "graph_uncertainty.pdf")
    metrics = {
        name: {"accuracy": result.accuracy, "selective_area": result.selective_area} for name, result in results.items()
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
