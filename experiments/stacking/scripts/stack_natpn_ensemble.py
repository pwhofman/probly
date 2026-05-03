"""Composition #2: ensemble of Natural Posterior Networks.

Builds N independent NaturalPosteriorNetwork members on top of cached
embeddings (each with its own latent projection + radial flow +
classifier head), trains every member with the standard PostNet
Bayesian loss, and aggregates per-sample Dirichlet means across members
for the final predictive probability.

NatPN does not output raw logits, so the calibration / conformal stack
used by ``stack_dare_temp_conformal.py`` is intentionally not applied
here -- this script just measures the headline ensemble metrics
(accuracy + ECE).

CLI:

    --dataset {two_moons,cifar10h}   required
    --encoder <name>                 default: siglip2 (ignored on two_moons)
    --num-members <int>              default: 10
    --epochs <int>                   default: 100
    --batch-size <int>               default: 512
    --latent-dim <int>               default: 64
    --hidden <int>                   default: 256
    --num-flows <int>                default: 8
    --entropy-weight <float>         default: 1e-5
    --seed <int>                     default: 0
    --device <str>                   default: auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from probly.method.natural_posterior_network import natural_posterior_network
from probly.train.evidential.torch import postnet_loss

from stacking.data import load_dataset
from stacking.utils import get_device, set_seed

ECE_BINS = 15


def _write_results(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON dict to ``path``, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"\nwrote results -> {path}")


class _NatPNEncoder(nn.Module):
    """Tiny MLP encoder mapping ``(B, in_features) -> (B, hidden)``.

    Exposes ``out_features`` so probly's ``get_output_dim`` resolves the
    latent projection input size without a forward pass.
    """

    def __init__(self, in_features: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out_features = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_member(
    in_features: int,
    num_classes: int,
    *,
    hidden: int,
    latent_dim: int,
    num_flows: int,
) -> nn.Module:
    """Construct one NatPN member from a fresh encoder."""
    encoder = _NatPNEncoder(in_features, hidden)
    return natural_posterior_network(
        encoder,
        latent_dim=latent_dim,
        num_classes=num_classes,
        num_flows=num_flows,
        predictor_type="probabilistic_classifier",
    )


def _train_member(
    member: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    entropy_weight: float,
    device: torch.device,
) -> None:
    """Train one NatPN member with mini-batch postnet_loss + Adam(lr=1e-3)."""
    member.train()
    member.to(device)
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X[idx]
            yb = y[idx]
            opt.zero_grad()
            alpha = member(xb)
            loss = postnet_loss(alpha, yb, entropy_weight=entropy_weight, reduction="mean")
            loss.backward()
            opt.step()


@torch.no_grad()
def _ensemble_probs(
    members: list[nn.Module], X: torch.Tensor, device: torch.device, batch_size: int
) -> torch.Tensor:
    """Mean of per-member Dirichlet-mean probabilities across members.

    Returns probabilities of shape ``(B, num_classes)``.
    """
    per_member: list[torch.Tensor] = []
    for member in members:
        member.eval()
        member.to(device)
        chunks: list[torch.Tensor] = []
        for start in range(0, X.shape[0], batch_size):
            xb = X[start : start + batch_size].to(device)
            alpha = member(xb)
            chunks.append((alpha / alpha.sum(dim=-1, keepdim=True)).detach())
        per_member.append(torch.cat(chunks, dim=0))
    return torch.stack(per_member, dim=0).mean(dim=0)


def _accuracy(probs: torch.Tensor, y: torch.Tensor) -> float:
    """Top-1 accuracy from class probabilities."""
    return float((probs.argmax(dim=-1) == y).float().mean().item())


def _ece(probs: torch.Tensor, y: torch.Tensor, *, n_bins: int) -> float:
    """Expected calibration error with uniform binning over max-prob.

    Args:
        probs: (B, C) class probabilities.
        y: (B,) integer labels.
        n_bins: Number of equal-width bins over ``[0, 1]``.
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == y).float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        in_bin = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if i == 0:
            in_bin = in_bin | (confidences <= edges[1])
        prop = in_bin.float().mean()
        if prop.item() > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece = ece + prop * (avg_conf - avg_acc).abs()
    return float(ece.item())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["two_moons", "cifar10h"], required=True)
    parser.add_argument("--encoder", default="siglip2", help="Ignored for two_moons.")
    parser.add_argument("--num-members", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-flows", type=int, default=8)
    parser.add_argument("--entropy-weight", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="If given, write a JSON dump of the run's hyperparams + metrics to this path.",
    )
    return parser.parse_args()


def _to_tensors(arr: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=dtype, device=device)


def main() -> None:
    """Run an ensemble of NatPN members on the chosen dataset."""
    args = _parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    ds = load_dataset(args.dataset, encoder=args.encoder, seed=args.seed)
    print(f"Dataset: {ds.meta.get('name')}  in_features={ds.in_features}  num_classes={ds.num_classes}")
    print(f"  train={ds.X_train.shape[0]}  calib={ds.X_calib.shape[0]}  test={ds.X_test.shape[0]}")

    X_train = _to_tensors(ds.X_train, dtype=torch.float32, device=device)
    y_train = _to_tensors(ds.y_train, dtype=torch.long, device=device)
    X_test = _to_tensors(ds.X_test, dtype=torch.float32, device=device)
    y_test = _to_tensors(ds.y_test, dtype=torch.long, device=device)

    members: list[nn.Module] = []
    for i in range(args.num_members):
        torch.manual_seed(args.seed * 1000 + i)
        member = _build_member(
            in_features=ds.in_features,
            num_classes=ds.num_classes,
            hidden=args.hidden,
            latent_dim=args.latent_dim,
            num_flows=args.num_flows,
        )
        print(f"  training member {i + 1}/{args.num_members} ...")
        _train_member(
            member,
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            entropy_weight=args.entropy_weight,
            device=device,
        )
        members.append(member)

    test_probs = _ensemble_probs(members, X_test, device=device, batch_size=args.batch_size)
    acc = _accuracy(test_probs, y_test)
    ece = _ece(test_probs, y_test, n_bins=ECE_BINS)
    print(f"\nNatPN ensemble (N={args.num_members}): test_acc={acc:.4f}  ECE={ece:.4f}")

    if args.results_json is not None:
        _write_results(
            args.results_json,
            {
                "composition": "natpn_ensemble",
                "dataset": args.dataset,
                "encoder": args.encoder if args.dataset == "cifar10h" else None,
                "in_features": ds.in_features,
                "num_classes": ds.num_classes,
                "splits": {
                    "train": int(ds.X_train.shape[0]),
                    "calib": int(ds.X_calib.shape[0]),
                    "test": int(ds.X_test.shape[0]),
                },
                "hyperparams": {
                    "num_members": args.num_members,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "hidden": args.hidden,
                    "latent_dim": args.latent_dim,
                    "num_flows": args.num_flows,
                    "entropy_weight": args.entropy_weight,
                    "seed": args.seed,
                    "ece_bins": ECE_BINS,
                },
                "metrics": {
                    "test_acc": acc,
                    "ece": ece,
                },
            },
        )


if __name__ == "__main__":
    main()
