"""Composition #1: DARE -> temperature scaling -> conformal RAPS.

Self-contained narrative: defines its own MLP class (or imports the
factory), trains each DARE member with cross-entropy, pools member
logits on the calib + test splits, then composes the calibration and
conformal layers as wrappers around ``torch_identity_logit_model``.

Runs on either dataset via ``--dataset``. CLI:

    --dataset {two_moons,cifar10h}   required
    --encoder <name>                 default: siglip2 (ignored on two_moons)
    --num-members <int>              default: 5
    --epochs <int>                   default: 200
    --alpha <float>                  default: 0.1
    --seed <int>                     default: 0
    --device <str>                   default: auto

Prints test accuracy of the uncalibrated mean-logit ensemble, ECE before
vs. after temperature scaling, and the conformal coverage / average set
size of the conformal-RAPS layer at the requested alpha. No figures.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
from torch import nn

from probly.calibrator import calibrate
from probly.method.calibration import temperature_scaling, torch_identity_logit_model
from probly.method.conformal import conformal_raps
from probly.method.dare import dare
from probly.metrics import average_set_size, empirical_coverage_classification
from probly.predictor import predict, predict_raw

from stacking.data import load_dataset
from stacking.models import build_mlp
from stacking.utils import get_device, set_seed

ECE_BINS = 15


def _train_member(
    member: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    device: torch.device,
) -> None:
    """Train one ensemble member with full-batch CE + Adam (lr=1e-2).

    Full-batch suffices for small classification heads on cached
    embeddings or 2-D toy data. Mini-batches can be added per-script
    later if a future composition needs them; this script keeps the
    loop visible top-to-bottom.
    """
    member.train()
    member.to(device)
    opt = torch.optim.Adam(member.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = member(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()


@torch.no_grad()
def _pool_logits(ensemble: nn.ModuleList, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Mean of member logits on x. Shape (B, num_classes)."""
    stacked: list[torch.Tensor] = []
    for member in ensemble:
        member.eval()
        member.to(device)
        stacked.append(member(x.to(device)))
    return torch.stack(stacked, dim=0).mean(dim=0).detach()


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Top-1 accuracy from raw logits."""
    return float((logits.argmax(dim=-1) == y).float().mean().item())


def _ece(logits: torch.Tensor, y: torch.Tensor, *, n_bins: int) -> float:
    """Expected calibration error with uniform binning over max-prob.

    Args:
        logits: (B, C) float tensor.
        y: (B,) integer tensor.
        n_bins: Number of equal-width bins over [0, 1].
    """
    probs = torch.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == y).float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)
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
    parser.add_argument("--num-members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _to_tensors(arr: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=dtype, device=device)


def main() -> None:
    """Run DARE -> temperature -> conformal-RAPS on the chosen dataset."""
    args = _parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    ds = load_dataset(args.dataset, encoder=args.encoder, seed=args.seed)
    print(f"Dataset: {ds.meta.get('name')}  in_features={ds.in_features}  num_classes={ds.num_classes}")
    print(f"  train={ds.X_train.shape[0]}  calib={ds.X_calib.shape[0]}  test={ds.X_test.shape[0]}")

    x_train = _to_tensors(ds.X_train, dtype=torch.float32, device=device)
    y_train = _to_tensors(ds.y_train, dtype=torch.long, device=device)
    x_calib = _to_tensors(ds.X_calib, dtype=torch.float32, device=device)
    y_calib = _to_tensors(ds.y_calib, dtype=torch.long, device=device)
    x_test = _to_tensors(ds.X_test, dtype=torch.float32, device=device)
    y_test = _to_tensors(ds.y_test, dtype=torch.long, device=device)

    # 1. Build the base model and the DARE ensemble of fresh-initialised copies.
    base = build_mlp(in_features=ds.in_features, num_classes=ds.num_classes)
    ensemble: nn.ModuleList = dare(base, num_members=args.num_members)  # ty: ignore[invalid-assignment]

    # 2. Train each member independently with CE + Adam.
    for i, member in enumerate(ensemble):
        print(f"  training member {i + 1}/{args.num_members} ...")
        _train_member(member, x_train, y_train, epochs=args.epochs, device=device)

    # 3. Pool member logits on the calibration and test splits.
    calib_logits = _pool_logits(ensemble, x_calib, device=device)
    test_logits = _pool_logits(ensemble, x_test, device=device)

    # 4. Report uncalibrated test accuracy + ECE on the pooled ensemble.
    acc = _accuracy(test_logits, y_test)
    ece_uncal = _ece(test_logits, y_test, n_bins=ECE_BINS)
    print(f"\nuncalibrated ensemble: test_acc={acc:.4f}  ECE={ece_uncal:.4f}")

    # 5. Calibration layer: temperature scaling fitted on cached calib logits.
    calibrator = temperature_scaling(torch_identity_logit_model())
    calibrate(calibrator, y_calib, calib_logits)
    test_logits_calibrated = predict_raw(calibrator, test_logits)
    if not isinstance(test_logits_calibrated, torch.Tensor):
        msg = f"expected torch.Tensor calibrated logits, got {type(test_logits_calibrated)}"
        raise TypeError(msg)
    ece_cal = _ece(test_logits_calibrated, y_test, n_bins=ECE_BINS)
    print(f"calibrated ensemble:    test_acc={_accuracy(test_logits_calibrated, y_test):.4f}  ECE={ece_cal:.4f}")

    # 6. Conformal layer: RAPS wrapped around the (temperature-scaled) calibrator.
    conformalizer: Any = conformal_raps(calibrator)
    calibrate(conformalizer, args.alpha, y_calib, calib_logits)
    test_sets = predict(conformalizer, test_logits)
    coverage = float(empirical_coverage_classification(test_sets, y_test))
    avg_size = float(average_set_size(test_sets))
    print(
        f"\nconformal RAPS @ alpha={args.alpha}:"
        f"  coverage={coverage:.3f}  avg_set_size={avg_size:.3f}"
    )


if __name__ == "__main__":
    main()
