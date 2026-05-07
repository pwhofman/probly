"""Graph node-classification demos for GPN variants."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Literal, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon
import networkx as nx

from probly.evaluation.tasks import selective_prediction
from probly.predictor import predict
from probly.quantification import quantify
from probly.train.evidential.torch import mixture_uce_loss, postnet_loss
from probly.method.graph_posterior_network import (
    cuq_graph_neural_network,
    graph_posterior_network,
    lop_graph_posterior_network,
)

FIGURE_FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
for font_path in FIGURE_FONT_DIR.glob("FiraSans-*.ttf"):
    font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "Fira Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

ModelName = Literal["GPN", "LOP-GPN", "CUQ-GNN"]
ExperimentName = Literal["synthetic", "amazon-photo", "all"]
LayoutName = Literal["forceatlas2", "precomputed"]
MODEL_NAMES: tuple[ModelName, ...] = ("GPN", "LOP-GPN", "CUQ-GNN")
DEFAULT_AMAZON_LAYOUT_PATH = Path(__file__).resolve().parents[2] / "assets" / "amazon_photo_gephi_layout.npy"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation output for one trained model."""

    accuracy: float | None
    selective_area: float | None
    coverage: np.ndarray
    selective_loss: np.ndarray
    predictions: np.ndarray
    total_uncertainty: np.ndarray
    epistemic_uncertainty: np.ndarray
    available: bool = True
    dirichlet_alphas: np.ndarray | None = None
    mixture_weights: np.ndarray | None = None


def set_seed(seed: int) -> None:
    """Set numpy and torch random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_class_colors(num_classes: int) -> np.ndarray:
    """Return a bright deterministic color palette for class predictions."""
    base_colors = [
        "#1F77D0",
        "#00A6D6",
        "#34B3A0",
        "#4CB944",
        "#8BC34A",
        "#C5C934",
        "#F2C230",
        "#5DADE2",
        "#2E86AB",
        "#A3D977",
    ]
    if num_classes <= len(base_colors):
        return np.array(base_colors[:num_classes])
    cmap = plt.get_cmap("tab20", num_classes)
    return np.array([cmap(class_idx) for class_idx in range(num_classes)], dtype=object)


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


def add_deterministic_masks(data: Data, seed: int, train_per_class: int = 20, val_per_class: int = 30) -> Data:
    """Add deterministic stratified train, validation, and test masks to a graph."""
    rng = np.random.default_rng(seed)
    labels = data.y.cpu().numpy()
    train_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)
    for class_idx in np.unique(labels):
        class_nodes = np.where(labels == class_idx)[0]
        shuffled = rng.permutation(class_nodes)
        train_count = min(train_per_class, len(shuffled))
        val_count = min(val_per_class, max(0, len(shuffled) - train_count))
        train_mask[shuffled[:train_count]] = True
        val_mask[shuffled[train_count : train_count + val_count]] = True
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = ~(data.train_mask | data.val_mask)
    return data


def load_amazon_photo(data_dir: Path, seed: int) -> Data:
    """Load Amazon Photos and attach deterministic node-classification masks."""
    LOGGER.info("Loading Amazon Photos dataset from %s", data_dir / "amazon")
    dataset = Amazon(root=str(data_dir / "amazon"), name="Photo")
    data = dataset[0]
    LOGGER.info("Loaded Amazon Photos: %d nodes, %d directed edges", data.num_nodes, data.edge_index.shape[1])
    return add_deterministic_masks(data, seed=seed)


def normalize_positions(pos: np.ndarray) -> np.ndarray:
    """Center and scale graph positions for stable plotting."""
    centered = pos - pos.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(centered, axis=1))
    return centered / max(float(scale), 1e-6)


def forceatlas2_positions(data: Data, cache_path: Path, seed: int, max_iter: int) -> np.ndarray:
    """Compute or load deterministic ForceAtlas2 node positions."""
    if cache_path.exists():
        LOGGER.info("Loading cached ForceAtlas2 positions from %s", cache_path)
        return np.load(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Computing ForceAtlas2 positions: seed=%d, iterations=%d", seed, max_iter)
    start = time.perf_counter()
    edge_index = data.edge_index.cpu().numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from((int(src), int(dst)) for src, dst in edge_index.T if src != dst)
    layout = nx.forceatlas2_layout(
        graph,
        max_iter=max_iter,
        scaling_ratio=2.0,
        gravity=1.0,
        seed=seed,
    )
    pos = np.array([layout[node] for node in range(data.num_nodes)], dtype=np.float32)
    pos = normalize_positions(pos).astype(np.float32)
    np.save(cache_path, pos)
    LOGGER.info("Saved ForceAtlas2 positions to %s in %.1fs", cache_path, time.perf_counter() - start)
    return pos


def precomputed_positions(data: Data, layout_path: Path) -> np.ndarray:
    """Load precomputed node positions for a graph.

    Args:
        data: Graph whose node count must match the layout.
        layout_path: Path to a ``.npy`` file containing ``(num_nodes, 2)`` positions.

    Returns:
        Centered and scaled positions as ``float32``.
    """
    if not layout_path.exists():
        raise FileNotFoundError(f"Precomputed layout file does not exist: {layout_path}")
    LOGGER.info("Loading precomputed positions from %s", layout_path)
    pos = np.load(layout_path)
    expected_shape = (data.num_nodes, 2)
    if pos.shape != expected_shape:
        raise ValueError(f"Precomputed layout at {layout_path} has shape {pos.shape}; expected {expected_shape}")
    return normalize_positions(pos).astype(np.float32)


def build_model(name: ModelName, feature_dim: int, num_classes: int, encoder_dim: int = 24, latent_dim: int = 8) -> nn.Module:
    """Build one GPN variant.

    Args:
        name: Model variant name.
        feature_dim: Input feature dimension.
        num_classes: Number of target classes.
        encoder_dim: Hidden dimension used by the encoder.
        latent_dim: Latent dimension used by the GPN density model.

    Returns:
        A trainable GPN variant.
    """
    encoder = nn.Sequential(nn.Linear(feature_dim, encoder_dim), nn.ReLU(), nn.Linear(encoder_dim, encoder_dim), nn.ReLU())
    common: dict[str, int | float] = {
        "encoder_dim": encoder_dim,
        "num_flows": 4,
        "propagation_steps": 8,
        "teleport_probability": 0.12,
    }
    if name == "GPN":
        return graph_posterior_network(encoder, latent_dim, num_classes, **common)
    if name == "LOP-GPN":
        return lop_graph_posterior_network(encoder, latent_dim, num_classes, **common)
    if name == "CUQ-GNN":
        return cuq_graph_neural_network(encoder, latent_dim, num_classes, convolution_name="appnp", **common)
    raise ValueError(name)


def train_model(model: nn.Module, data: Data, name: ModelName, epochs: int, lr: float, use_mixed_precision: bool) -> None:
    """Train a GPN model on the graph train mask."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
    log_every = max(1, epochs // 10)
    start = time.perf_counter()
    LOGGER.info(
        "Training %s for %d epochs on %d labeled nodes%s",
        name,
        epochs,
        int(data.train_mask.sum().item()),
        " with CUDA float16 mixed precision" if use_mixed_precision else "",
    )
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_mixed_precision):
            if name == "LOP-GPN":
                alpha_features, mixture_weights = model.forward(data)
                loss = mixture_uce_loss(alpha_features, mixture_weights[data.train_mask], data.y[data.train_mask], "mean")
            else:
                alpha = model(data)
                loss = postnet_loss(alpha[data.train_mask], data.y[data.train_mask], entropy_weight=1e-5, reduction="mean")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            LOGGER.info("%s epoch %d/%d: loss=%.4f", name, epoch, epochs, float(loss.detach().cpu().item()))
    LOGGER.info("Finished training %s in %.1fs", name, time.perf_counter() - start)


def checkpoint_path(checkpoint_dir: Path, dataset_name: str, model_name: ModelName) -> Path:
    """Return the checkpoint path for one dataset/model pair."""
    safe_model_name = model_name.lower().replace("-", "_")
    return checkpoint_dir / f"{dataset_name}_{safe_model_name}.pt"


def train_or_load_model(
    name: ModelName,
    data: Data,
    num_classes: int,
    checkpoint_dir: Path,
    dataset_name: str,
    seed: int,
    epochs: int,
    lr: float,
    device: torch.device,
    retrain: bool,
    use_mixed_precision: bool,
    encoder_dim: int = 24,
    latent_dim: int = 8,
) -> nn.Module:
    """Load a cached model checkpoint or train and save a new model."""
    model = build_model(name, data.x.shape[-1], num_classes, encoder_dim=encoder_dim, latent_dim=latent_dim).to(device)
    path = checkpoint_path(checkpoint_dir, dataset_name, name)
    if path.exists() and not retrain:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        compatible = (
            checkpoint.get("feature_dim") == int(data.x.shape[-1])
            and checkpoint.get("num_classes") == num_classes
            and checkpoint.get("seed") == seed
            and checkpoint.get("epochs") == epochs
            and checkpoint.get("lr") == lr
            and checkpoint.get("mixed_precision", False) == use_mixed_precision
            and checkpoint.get("encoder_dim", 24) == encoder_dim
            and checkpoint.get("latent_dim", 8) == latent_dim
        )
        if compatible:
            LOGGER.info("Loading cached %s checkpoint from %s", name, path)
            model.load_state_dict(checkpoint["model_state_dict"])
            return model
        LOGGER.info("Ignoring incompatible %s checkpoint at %s", name, path)
    elif path.exists():
        LOGGER.info("Retraining %s despite existing checkpoint at %s", name, path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_model(model, data, name, epochs, lr, use_mixed_precision=use_mixed_precision)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_name": name,
            "dataset_name": dataset_name,
            "feature_dim": int(data.x.shape[-1]),
            "num_classes": num_classes,
            "seed": seed,
            "epochs": epochs,
            "lr": lr,
            "mixed_precision": use_mixed_precision,
            "encoder_dim": encoder_dim,
            "latent_dim": latent_dim,
        },
        path,
    )
    LOGGER.info("Saved %s checkpoint to %s", name, path)
    return model


def load_checkpointed_model(
    name: ModelName,
    data: Data,
    num_classes: int,
    checkpoint_dir: Path,
    dataset_name: str,
    device: torch.device,
    encoder_dim: int = 24,
    latent_dim: int = 8,
) -> nn.Module | None:
    """Load an architecture-compatible checkpoint without training.

    Args:
        name: Model variant name.
        data: Graph data used for inference.
        num_classes: Number of target classes.
        checkpoint_dir: Directory containing model checkpoints.
        dataset_name: Dataset prefix used in checkpoint filenames.
        device: Torch device for inference.

    Returns:
        A model with loaded weights, or ``None`` when no compatible checkpoint exists.
    """
    path = checkpoint_path(checkpoint_dir, dataset_name, name)
    if not path.exists():
        LOGGER.info("No %s checkpoint found at %s; using gray placeholder plot", name, path)
        return None
    model = build_model(name, data.x.shape[-1], num_classes, encoder_dim=encoder_dim, latent_dim=latent_dim).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    compatible = (
        checkpoint.get("feature_dim") == int(data.x.shape[-1])
        and checkpoint.get("num_classes") == num_classes
        and checkpoint.get("encoder_dim", 24) == encoder_dim
        and checkpoint.get("latent_dim", 8) == latent_dim
    )
    if not compatible:
        LOGGER.info("Ignoring incompatible %s checkpoint at %s; using gray placeholder plot", name, path)
        return None
    LOGGER.info("Loading %s checkpoint from %s", name, path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def missing_evaluation(data: Data) -> EvaluationResult:
    """Return a placeholder result for an unavailable checkpoint."""
    num_nodes = int(data.num_nodes)
    return EvaluationResult(
        accuracy=None,
        selective_area=None,
        coverage=np.array([], dtype=float),
        selective_loss=np.array([], dtype=float),
        predictions=np.full(num_nodes, -1, dtype=int),
        total_uncertainty=np.zeros(num_nodes, dtype=float),
        epistemic_uncertainty=np.zeros(num_nodes, dtype=float),
        available=False,
        dirichlet_alphas=None,
        mixture_weights=None,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: Data,
    n_bins: int,
    use_mixed_precision: bool,
    include_density_parameters: bool = False,
) -> EvaluationResult:
    """Evaluate accuracy, uncertainty decomposition, and selective prediction."""
    start = time.perf_counter()
    model.eval()
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_mixed_precision):
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
    dirichlet_alphas: np.ndarray | None = None
    mixture_weights: np.ndarray | None = None
    if include_density_parameters:
        representation = model.predict_representation(data)
        if hasattr(representation, "components") and hasattr(representation, "mixture_weights"):
            dirichlet_alphas = representation.components.alphas.detach().cpu().numpy()
            mixture_weights = representation.mixture_weights.detach().cpu().numpy()
        else:
            dirichlet_alphas = representation.alphas.detach().cpu().numpy()
    LOGGER.info("Evaluated model: accuracy=%.3f, AULC=%.3f in %.1fs", accuracy, selective_area, time.perf_counter() - start)
    return EvaluationResult(accuracy, selective_area, coverage, selective_loss, pred_np, total, epistemic, True, dirichlet_alphas, mixture_weights)


def plot_selective(results: dict[ModelName, EvaluationResult], output_path: Path) -> None:
    """Plot selective-prediction curves."""
    LOGGER.info("Writing selective-prediction plot to %s", output_path)
    _, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, result in results.items():
        if not result.available or result.selective_area is None:
            continue
        ax.plot(result.coverage, 1.0 - result.selective_loss, linewidth=2.2, label=f"{name} (AULC={result.selective_area:.3f})")
    ax.set_xlabel("Coverage after rejecting most uncertain nodes")
    ax.set_ylabel("Accuracy on retained nodes")
    ax.set_title("Selective Prediction Accuracy")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    if any(result.available for result in results.values()):
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def simplex_grid(resolution: int = 45) -> np.ndarray:
    """Create barycentric coordinates on the three-class probability simplex."""
    points: list[tuple[float, float, float]] = []
    for blue in range(resolution + 1):
        for yellow in range(resolution + 1 - blue):
            green = resolution - blue - yellow
            points.append((blue / resolution, yellow / resolution, green / resolution))
    return np.asarray(points, dtype=float)


def dirichlet_log_density(points: np.ndarray, alphas: np.ndarray, max_visual_concentration: float = 18.0) -> np.ndarray:
    """Evaluate Dirichlet log density on simplex points."""
    clipped = np.clip(points, 1e-6, 1.0)
    alpha = np.asarray(alphas, dtype=float)
    concentration = float(alpha.sum())
    if concentration > max_visual_concentration:
        mean = alpha / concentration
        alpha = 1.0 + mean * (max_visual_concentration - len(alpha))
    log_norm = float(torch.lgamma(torch.tensor(alpha.sum())) - torch.lgamma(torch.tensor(alpha)).sum())
    return log_norm + ((alpha - 1.0) * np.log(clipped)).sum(axis=1)


def node_simplex_density(result: EvaluationResult, node_idx: int, points: np.ndarray, top_components: int = 96) -> np.ndarray:
    """Evaluate a node's predicted Dirichlet or Dirichlet-mixture density."""
    if result.dirichlet_alphas is None:
        return np.zeros(len(points), dtype=float)
    if result.mixture_weights is None:
        return dirichlet_log_density(points, result.dirichlet_alphas[node_idx])
    weights = result.mixture_weights[node_idx]
    component_count = min(top_components, len(weights))
    component_idx = np.argsort(weights)[-component_count:]
    component_weights = weights[component_idx]
    component_weights = component_weights / max(float(component_weights.sum()), 1e-12)
    component_alphas = result.dirichlet_alphas[node_idx, component_idx]
    log_terms = np.vstack(
        [np.log(max(float(weight), 1e-12)) + dirichlet_log_density(points, alpha) for weight, alpha in zip(component_weights, component_alphas, strict=True)]
    )
    max_log = log_terms.max(axis=0)
    return max_log + np.log(np.exp(log_terms - max_log).sum(axis=0))


def select_bridge_zoom_node(data: Data, results: dict[ModelName, EvaluationResult]) -> int | None:
    """Pick a shared misclassified node on the blue-yellow bridge."""
    labels = data.y.cpu().numpy()
    pos = data.pos.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    available_results = [result for result in results.values() if result.available]
    if not available_results:
        return None
    wrong_for_all = np.logical_and.reduce([result.predictions != labels for result in available_results])
    bridge_candidates = np.flatnonzero(wrong_for_all & ~train_mask & np.isin(labels, [0, 1]))
    if len(bridge_candidates) == 0:
        bridge_candidates = np.flatnonzero(wrong_for_all & ~train_mask)
    if len(bridge_candidates) == 0:
        return None
    blue_center = pos[labels == 0].mean(axis=0)
    yellow_center = pos[labels == 1].mean(axis=0)
    bridge_center = 0.5 * (blue_center + yellow_center)
    scores = np.linalg.norm(pos[bridge_candidates] - bridge_center, axis=1)
    return int(bridge_candidates[np.argmin(scores)])


def draw_simplex_density_inset(
    ax: plt.Axes,
    result: EvaluationResult,
    node_idx: int,
    node_pos: np.ndarray,
    inset_center: np.ndarray,
    class_colors: np.ndarray,
    density_color: str,
) -> None:
    """Draw a barycentric density inset below a highlighted graph node."""
    if not result.available or result.dirichlet_alphas is None:
        return
    points = simplex_grid(resolution=75)
    width = 0.72
    height = 0.56
    center = inset_center + np.array([0.0, -0.28])
    top = center + np.array([0.0, 2.0 * height / 3.0])
    bottom_left = center + np.array([-width / 2.0, -height / 3.0])
    bottom_right = center + np.array([width / 2.0, -height / 3.0])
    xy = points[:, [0]] * bottom_left + points[:, [1]] * bottom_right + points[:, [2]] * top
    log_density = node_simplex_density(result, node_idx, points)
    relative_density = np.exp(log_density - np.max(log_density))
    density = np.where(relative_density > 1e-4, 0.16 + 0.84 * relative_density**0.28, 0.0)
    bottom_midpoint = 0.5 * (bottom_left + bottom_right)
    ax.plot([node_pos[0], bottom_midpoint[0]], [node_pos[1], bottom_midpoint[1]], color="#2B2D33", linewidth=0.8, alpha=0.7, zorder=7)
    density_cmap = LinearSegmentedColormap.from_list("error_density", ["#FFFFFF", "#FFB4C3", density_color])
    ax.tripcolor(xy[:, 0], xy[:, 1], density, shading="gouraud", cmap=density_cmap, vmin=0.0, vmax=1.0, alpha=0.96, zorder=8)
    ax.add_patch(Polygon([bottom_left, bottom_right, top], closed=True, fill=False, edgecolor="#2B2D33", linewidth=0.9, zorder=9))
    ax.scatter([bottom_left[0], bottom_right[0], top[0]], [bottom_left[1], bottom_right[1], top[1]], s=22, c=class_colors[:3], edgecolors="#FFFFFF", linewidths=0.5, zorder=10)


def largest_component_mask(num_nodes: int, edge_pairs: np.ndarray) -> np.ndarray:
    """Return a mask selecting nodes in the largest connected component."""
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from((int(src), int(dst)) for src, dst in edge_pairs)
    component = max(nx.connected_components(graph), key=len)
    mask = np.zeros(num_nodes, dtype=bool)
    mask[list(component)] = True
    return mask


def uncertainty_marker_sizes(uncertainty: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    """Map uncertainty to marker areas without letting outliers dominate dense graphs."""
    visible_uncertainty = uncertainty[visible_mask]
    if int(visible_mask.sum()) > 1000:
        lower = float(np.percentile(visible_uncertainty, 82.0))
        upper = float(np.percentile(visible_uncertainty, 99.2))
        normalized = np.clip((uncertainty - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
        return 3.5 + 24.0 * normalized**1.6
    lower = float(np.percentile(visible_uncertainty, 5.0))
    upper = float(np.percentile(visible_uncertainty, 95.0))
    normalized = np.clip((uncertainty - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
    return 18.0 + 120.0 * np.sqrt(normalized)


def set_square_limits(ax: plt.Axes, pos: np.ndarray, pad_fraction: float = 0.04) -> None:
    """Set equal square limits around the visible graph positions."""
    min_xy = pos.min(axis=0)
    max_xy = pos.max(axis=0)
    center = 0.5 * (min_xy + max_xy)
    half_width = 0.5 * float(np.max(max_xy - min_xy))
    half_width = max(half_width * (1.0 + pad_fraction), 1e-6)
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)


def draw_graph_row(
    axes: Sequence[plt.Axes],
    data: Data,
    results: dict[ModelName, EvaluationResult],
    show_density_zoom: bool = False,
    edge_linewidth: float = 0.35,
    edge_alpha: float = 0.55,
    show_column_titles: bool = True,
    accuracy_inside: bool = False,
) -> None:
    """Draw one dataset row of graph predictions and uncertainty."""
    pos = data.pos.cpu().numpy()
    edges = data.edge_index.cpu().numpy().T
    labels = data.y.cpu().numpy()
    edge_pairs = np.unique(np.sort(edges, axis=1), axis=0)
    visible_mask = largest_component_mask(len(pos), edge_pairs)
    visible_edges = edge_pairs[visible_mask[edge_pairs[:, 0]] & visible_mask[edge_pairs[:, 1]]]
    visible_pos = pos[visible_mask]
    segments = np.stack((pos[visible_edges[:, 0]], pos[visible_edges[:, 1]]), axis=1)
    colors = get_class_colors(int(labels.max()) + 1)
    edge_color = "#DDE3EA"
    error_color = "#FF2D6D"
    train_color = "#8D96A6"
    missing_color = "#B8C0CC"
    train_mask = data.train_mask.cpu().numpy()
    zoom_node = select_bridge_zoom_node(data, results) if show_density_zoom else None
    zoom_inset_center = None
    if zoom_node is not None:
        LOGGER.info("Selected node %d for synthetic Dirichlet density insets", zoom_node)
        zoom_inset_center = np.vstack([pos[labels == class_idx].mean(axis=0) for class_idx in range(3)]).mean(axis=0)
    for ax, (name, result) in zip(axes, results.items(), strict=True):
        ax.set_facecolor("#FBFCFF")
        edge_collection = LineCollection(segments, colors=edge_color, linewidths=edge_linewidth, alpha=edge_alpha, zorder=0, rasterized=True)
        ax.add_collection(edge_collection)
        uncertainty = result.epistemic_uncertainty
        sizes = uncertainty_marker_sizes(uncertainty, visible_mask)
        if result.available:
            wrong = result.predictions != labels
            correct = ~wrong & ~train_mask & visible_mask
            train = train_mask & visible_mask
            wrong_test = wrong & ~train_mask & visible_mask
            ax.scatter(
                pos[correct, 0],
                pos[correct, 1],
                s=sizes[correct],
                c=colors[result.predictions[correct]],
                alpha=0.86,
                edgecolors="#FFFFFF",
                linewidths=0.65,
                rasterized=True,
            )
            ax.scatter(
                pos[wrong_test, 0],
                pos[wrong_test, 1],
                s=sizes[wrong_test],
                c=error_color,
                alpha=0.92,
                edgecolors="none",
                linewidths=0.0,
                rasterized=True,
            )
            ax.scatter(
                pos[train, 0],
                pos[train, 1],
                s=sizes[train],
                c=train_color,
                alpha=0.9,
                edgecolors="#FFFFFF",
                linewidths=0.65,
                rasterized=True,
            )
            accuracy_label = f"{result.accuracy:.2f}" if result.accuracy is not None else "?"
        else:
            ax.scatter(
                visible_pos[:, 0],
                visible_pos[:, 1],
                s=sizes[visible_mask],
                c=missing_color,
                alpha=0.86,
                edgecolors="#FFFFFF",
                linewidths=0.65,
                rasterized=True,
            )
            accuracy_label = "?"
        if zoom_node is not None:
            if zoom_inset_center is not None:
                draw_simplex_density_inset(ax, result, zoom_node, pos[zoom_node], zoom_inset_center, colors, error_color)
        if accuracy_inside:
            if show_column_titles:
                ax.set_title(name, fontsize=12.0, fontweight="bold", color="#171A1F", pad=7)
            ax.text(
                0.035,
                0.965,
                f"accuracy = {accuracy_label}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10.0,
                fontweight="bold",
                color="#171A1F",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FBFCFF", "edgecolor": "none", "alpha": 0.86},
                zorder=20,
            )
        elif show_column_titles:
            ax.set_title(f"{name}\naccuracy = {accuracy_label}", fontsize=11.5, fontweight="bold", color="#171A1F", pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_box_aspect(1.0)
        set_square_limits(ax, visible_pos)
        for spine in ax.spines.values():
            spine.set_color("#D7DCE3")
            spine.set_linewidth(0.9)


def plot_graph(
    data: Data,
    results: dict[ModelName, EvaluationResult],
    output_path: Path,
    figure_dpi: int,
    show_density_zoom: bool = False,
    edge_linewidth: float = 0.35,
    edge_alpha: float = 0.55,
) -> None:
    """Plot graph predictions and uncertainty for all model variants."""
    LOGGER.info("Writing graph uncertainty plot to %s", output_path)
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.75), sharex=True, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")
    draw_graph_row(
        axes,
        data,
        results,
        show_density_zoom=show_density_zoom,
        edge_linewidth=edge_linewidth,
        edge_alpha=edge_alpha,
        accuracy_inside=True,
    )
    fig.subplots_adjust(left=0.01, right=0.998, bottom=0.012, top=0.92, wspace=0.025)
    plt.savefig(output_path, dpi=figure_dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def plot_unified_graph(
    synthetic_data: Data,
    synthetic_results: dict[ModelName, EvaluationResult],
    amazon_data: Data,
    amazon_results: dict[ModelName, EvaluationResult],
    output_path: Path,
    figure_dpi: int,
) -> None:
    """Plot synthetic and Amazon Photos graph results in one 2x3 figure."""
    LOGGER.info("Writing unified graph uncertainty plot to %s", output_path)
    fig, axes = plt.subplots(2, len(MODEL_NAMES), figsize=(12.6, 9.6), sharex="row", sharey="row")
    fig.patch.set_facecolor("#FFFFFF")
    draw_graph_row(
        axes[0],
        synthetic_data,
        synthetic_results,
        show_density_zoom=True,
        edge_linewidth=0.65,
        edge_alpha=0.82,
        show_column_titles=True,
        accuracy_inside=True,
    )
    draw_graph_row(
        axes[1],
        amazon_data,
        amazon_results,
        show_column_titles=False,
        accuracy_inside=True,
    )
    fig.subplots_adjust(left=0.05, right=0.998, bottom=0.012, top=0.965, wspace=0.025, hspace=0.045)
    for row_idx, label in enumerate(("Synthetic Clusters", "Amazon Photos")):
        bbox = axes[row_idx, 0].get_position()
        fig.text(
            bbox.x0 - 0.026,
            0.5 * (bbox.y0 + bbox.y1),
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=12.0,
            fontweight="bold",
            color="#171A1F",
        )
    plt.savefig(output_path, dpi=figure_dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def write_metrics(results: dict[ModelName, EvaluationResult], output_path: Path) -> None:
    """Write accuracy and selective-prediction metrics."""
    metrics = {
        name: {"accuracy": result.accuracy, "selective_area": result.selective_area} for name, result in results.items()
    }
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Wrote metrics to %s", output_path)


def run_synthetic_demo(
    output_dir: Path,
    checkpoint_dir: Path,
    seed: int,
    nodes_per_class: int,
    epochs: int,
    lr: float,
    device: torch.device,
    retrain: bool,
    inference_only: bool,
    use_mixed_precision: bool,
    figure_dpi: int,
) -> tuple[Data, dict[ModelName, EvaluationResult]]:
    """Train synthetic graph variants and write plots and metrics."""
    LOGGER.info("Starting synthetic demo: seed=%d, nodes_per_class=%d, device=%s", seed, nodes_per_class, device)
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = make_synthetic_graph(nodes_per_class=nodes_per_class, seed=seed).to(device)
    LOGGER.info("Created synthetic graph: %d nodes, %d directed edges", data.num_nodes, data.edge_index.shape[1])
    num_classes = int(data.y.max().item()) + 1
    results: dict[ModelName, EvaluationResult] = {}
    for name in MODEL_NAMES:
        if inference_only:
            model = load_checkpointed_model(name, data, num_classes, checkpoint_dir, "synthetic", device)
        else:
            model = train_or_load_model(
                name,
                data,
                num_classes,
                checkpoint_dir,
                dataset_name="synthetic",
                seed=seed,
                epochs=epochs,
                lr=lr,
                device=device,
                retrain=retrain,
                use_mixed_precision=use_mixed_precision,
            )
        results[name] = (
            missing_evaluation(data)
            if model is None
            else evaluate_model(
                model,
                data,
                n_bins=100,
                use_mixed_precision=use_mixed_precision,
                include_density_parameters=True,
            )
        )
    plot_selective(results, output_dir / "selective_prediction.pdf")
    plot_graph(
        data,
        results,
        output_dir / "graph_uncertainty.pdf",
        figure_dpi=figure_dpi,
        show_density_zoom=True,
        edge_linewidth=0.65,
        edge_alpha=0.82,
    )
    plot_graph(
        data,
        {name: results[name] for name in ("GPN", "LOP-GPN")},
        output_dir / "graph_uncertainty_min.pdf",
        figure_dpi=figure_dpi,
        show_density_zoom=True,
        edge_linewidth=0.65,
        edge_alpha=0.82,
    )
    write_metrics(results, output_dir / "metrics.json")
    return data, results


def run_amazon_photo_demo(
    output_dir: Path,
    data_dir: Path,
    checkpoint_dir: Path,
    cache_dir: Path,
    seed: int,
    epochs: int,
    lr: float,
    forceatlas2_iterations: int,
    layout: LayoutName,
    layout_path: Path,
    device: torch.device,
    retrain: bool,
    inference_only: bool,
    use_mixed_precision: bool,
    figure_dpi: int,
) -> tuple[Data, dict[ModelName, EvaluationResult]]:
    """Train or load Amazon Photos variants and write plots and metrics."""
    LOGGER.info(
        "Starting Amazon Photos demo: seed=%d, epochs=%d, layout=%s, forceatlas2_iterations=%d, device=%s",
        seed,
        epochs,
        layout,
        forceatlas2_iterations,
        device,
    )
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_amazon_photo(data_dir=data_dir, seed=seed)
    if layout == "precomputed":
        pos = precomputed_positions(data, layout_path)
    else:
        pos = forceatlas2_positions(
            data,
            cache_dir / f"amazon_photo_forceatlas2_seed{seed}_iter{forceatlas2_iterations}.npy",
            seed=seed,
            max_iter=forceatlas2_iterations,
        )
    data.pos = torch.tensor(pos, dtype=torch.float)
    data = data.to(device)
    num_classes = int(data.y.max().item()) + 1
    encoder_dim = 32
    default_latent_dim = 16
    results: dict[ModelName, EvaluationResult] = {}
    for name in MODEL_NAMES:
        latent_dim = 8 if name == "GPN" else default_latent_dim
        if inference_only:
            model = load_checkpointed_model(
                name,
                data,
                num_classes,
                checkpoint_dir,
                "amazon_photo",
                device,
                encoder_dim=encoder_dim,
                latent_dim=latent_dim,
            )
        else:
            model = train_or_load_model(
                name,
                data,
                num_classes,
                checkpoint_dir,
                dataset_name="amazon_photo",
                seed=seed,
                epochs=epochs,
                lr=lr,
                device=device,
                retrain=retrain,
                use_mixed_precision=use_mixed_precision,
                encoder_dim=encoder_dim,
                latent_dim=latent_dim,
            )
        results[name] = missing_evaluation(data) if model is None else evaluate_model(model, data, n_bins=100, use_mixed_precision=use_mixed_precision)
    plot_selective(results, output_dir / "amazon_photo_selective_prediction.pdf")
    plot_graph(
        data,
        results,
        output_dir / "amazon_photo_graph_uncertainty.pdf",
        figure_dpi=figure_dpi,
    )
    write_metrics(results, output_dir / "amazon_photo_metrics.json")
    return data, results


def run_demo(
    output_dir: Path,
    seed: int = 7,
    nodes_per_class: int = 120,
    epochs: int = 120,
    lr: float = 0.01,
    experiment: ExperimentName = "synthetic",
    data_dir: Path = Path("data"),
    checkpoint_dir: Path = Path("checkpoints"),
    cache_dir: Path = Path("cache"),
    amazon_epochs: int = 20,
    forceatlas2_iterations: int = 10,
    amazon_layout: LayoutName = "precomputed",
    amazon_layout_path: Path = DEFAULT_AMAZON_LAYOUT_PATH,
    figure_dpi: int = 200,
    device: str = "cpu",
    retrain: bool = False,
    inference_only: bool = False,
) -> None:
    """Run one or both GPN graph uncertainty demos."""
    if figure_dpi <= 0:
        msg = "figure_dpi must be positive."
        raise ValueError(msg)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    use_mixed_precision = torch_device.type == "cuda"
    if use_mixed_precision:
        LOGGER.info("CUDA enabled: using float16 mixed precision for training and inference")
    synthetic_output: tuple[Data, dict[ModelName, EvaluationResult]] | None = None
    amazon_output: tuple[Data, dict[ModelName, EvaluationResult]] | None = None
    if experiment in {"synthetic", "all"}:
        synthetic_output = run_synthetic_demo(
            output_dir,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            nodes_per_class=nodes_per_class,
            epochs=epochs,
            lr=lr,
            device=torch_device,
            retrain=retrain,
            inference_only=inference_only,
            use_mixed_precision=use_mixed_precision,
            figure_dpi=figure_dpi,
        )
    if experiment in {"amazon-photo", "all"}:
        amazon_output = run_amazon_photo_demo(
            output_dir,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            cache_dir=cache_dir,
            seed=seed,
            epochs=amazon_epochs,
            lr=lr,
            forceatlas2_iterations=forceatlas2_iterations,
            layout=amazon_layout,
            layout_path=amazon_layout_path,
            device=torch_device,
            retrain=retrain,
            inference_only=inference_only,
            use_mixed_precision=use_mixed_precision,
            figure_dpi=figure_dpi,
        )
    if experiment == "all" and synthetic_output is not None and amazon_output is not None:
        synthetic_data, synthetic_results = synthetic_output
        amazon_data, amazon_results = amazon_output
        plot_unified_graph(
            synthetic_data,
            synthetic_results,
            amazon_data,
            amazon_results,
            output_dir / "unified_graph_uncertainty.pdf",
            figure_dpi=figure_dpi,
        )
