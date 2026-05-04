"""Experiment ``dare_stack``: DARE -> {none|temperature|isotonic|label_relaxation} -> conformal RAPS.

Trains an N-member DARE ensemble on the chosen dataset, pools member
logits by mean, and runs the conformal layer on top. The
calibration step in the middle is selected via ``--calibration``:

    none              plain CE training, no post-hoc layer
    temperature       CE training, post-hoc temperature scaling
    isotonic          CE training, post-hoc per-class isotonic
    label_relaxation  LabelRelaxationLoss (Lienen & Huellermeier 2021)
                      at training time, no post-hoc layer

The conformal RAPS layer always runs at the requested ``--alpha`` and
is fitted on the (possibly calibrated) calibration logits.

Pass ``--results-json <path>`` to write the run's hyperparams and
metrics to disk under the uniform stack-experiments schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from probly.calibrator import calibrate
from probly.method.calibration import torch_identity_logit_model
from probly.method.conformal import conformal_raps
from probly.method.dare import dare
from probly.metrics import average_set_size, empirical_coverage_classification
from probly.predictor import predict

from stacking.calibration_modes import CALIBRATION_MODES, calibrate_logits, make_loss
from stacking.data import load_dataset
from stacking.models import build_mlp
from stacking.utils import get_device, set_seed

ECE_BINS = 15
EXPERIMENT = "dare_stack"


def _train_member(
    member: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    loss_fn: nn.Module,
    device: torch.device,
) -> None:
    """Train one ensemble member with full-batch ``loss_fn`` + Adam(lr=1e-2)."""
    member.train()
    member.to(device)
    opt = torch.optim.Adam(member.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        logits = member(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()


@torch.no_grad()
def _pool_logits(ensemble: nn.ModuleList, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Mean of member logits on x. Shape ``(B, num_classes)``."""
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
    """Expected calibration error with uniform binning over max-prob."""
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


def _write_results(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON dict to ``path``, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"\nwrote results -> {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["two_moons", "cifar10h"], required=True)
    parser.add_argument("--encoder", default="siglip2", help="Ignored for two_moons.")
    parser.add_argument("--num-members", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage level.")
    parser.add_argument("--calibration", choices=list(CALIBRATION_MODES), default="temperature")
    parser.add_argument(
        "--lr-alpha",
        type=float,
        default=0.1,
        help="LabelRelaxationLoss alpha; only consulted when --calibration label_relaxation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-json", type=Path, default=None)
    return parser.parse_args()


def _to_tensors(arr: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=dtype, device=device)


def main() -> None:
    """Run dare_stack on the chosen dataset / encoder / calibration mode."""
    args = _parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    ds = load_dataset(args.dataset, encoder=args.encoder, seed=args.seed)
    print(
        f"Dataset: {ds.meta.get('name')}  in_features={ds.in_features}  num_classes={ds.num_classes}  "
        f"calibration={args.calibration}  seed={args.seed}"
    )
    print(f"  train={ds.X_train.shape[0]}  calib={ds.X_calib.shape[0]}  test={ds.X_test.shape[0]}")

    x_train = _to_tensors(ds.X_train, dtype=torch.float32, device=device)
    y_train = _to_tensors(ds.y_train, dtype=torch.long, device=device)
    x_calib = _to_tensors(ds.X_calib, dtype=torch.float32, device=device)
    y_calib = _to_tensors(ds.y_calib, dtype=torch.long, device=device)
    x_test = _to_tensors(ds.X_test, dtype=torch.float32, device=device)
    y_test = _to_tensors(ds.y_test, dtype=torch.long, device=device)

    base = build_mlp(in_features=ds.in_features, num_classes=ds.num_classes)
    ensemble: nn.ModuleList = dare(base, num_members=args.num_members)  # ty: ignore[invalid-assignment]

    loss_fn = make_loss(args.calibration, lr_alpha=args.lr_alpha)
    for i, member in enumerate(ensemble):
        print(f"  training member {i + 1}/{args.num_members} ...")
        _train_member(member, x_train, y_train, epochs=args.epochs, loss_fn=loss_fn, device=device)

    calib_logits = _pool_logits(ensemble, x_calib, device=device)
    test_logits = _pool_logits(ensemble, x_test, device=device)

    test_acc = _accuracy(test_logits, y_test)
    ece_uncal = _ece(test_logits, y_test, n_bins=ECE_BINS)
    print(f"\npooled ensemble: test_acc={test_acc:.4f}  ECE_uncal={ece_uncal:.4f}")

    calib_calibrated, test_calibrated = calibrate_logits(
        mode=args.calibration,
        calib_logits=calib_logits,
        test_logits=test_logits,
        y_calib=y_calib,
        num_classes=ds.num_classes,
    )
    ece_cal = _ece(test_calibrated, y_test, n_bins=ECE_BINS)
    print(f"after calibration ({args.calibration}): ECE_cal={ece_cal:.4f}")

    conformalizer: Any = conformal_raps(torch_identity_logit_model())
    calibrate(conformalizer, args.alpha, y_calib, calib_calibrated)
    test_sets = predict(conformalizer, test_calibrated)
    coverage = float(empirical_coverage_classification(test_sets, y_test))
    avg_size = float(average_set_size(test_sets))
    print(
        f"conformal RAPS @ alpha={args.alpha}: coverage={coverage:.3f}  avg_set_size={avg_size:.3f}"
    )

    if args.results_json is not None:
        _write_results(
            args.results_json,
            {
                "experiment": EXPERIMENT,
                "calibration": args.calibration,
                "dataset": args.dataset,
                "encoder": args.encoder if args.dataset == "cifar10h" else None,
                "seed": args.seed,
                "splits": {
                    "train": int(ds.X_train.shape[0]),
                    "calib": int(ds.X_calib.shape[0]),
                    "test": int(ds.X_test.shape[0]),
                },
                "hyperparams": {
                    "num_members": args.num_members,
                    "epochs": args.epochs,
                    "lr_alpha": args.lr_alpha,
                    "ece_bins": ECE_BINS,
                },
                "metrics": {
                    "test_acc": test_acc,
                    "ece_uncalibrated": ece_uncal,
                    "ece_calibrated": ece_cal,
                    "conformal_alpha": args.alpha,
                    "conformal_coverage": coverage,
                    "conformal_avg_set_size": avg_size,
                },
            },
        )


if __name__ == "__main__":
    main()
