"""Composition #3: credal-wrapper ensemble + conformal RAPS, ablating temperature scaling.

Trains an N-member ensemble (probly's ``credal_wrapper`` is the same
training-time object as ``ensemble`` -- N independent reset-init
classifiers; only the inference-time representer differs), pools the
member logits by mean, and then evaluates two stacks side-by-side at the
same conformal level alpha:

    A.  pooled-logits          ->  conformal_raps  ->  set
    B.  pooled-logits  ->  T   ->  conformal_raps  ->  set

where ``T`` is a temperature-scaling layer fitted on the calibration
split. Reports coverage and average set size for both, so the change
attributable to the temperature step is visible directly.

CLI:

    --dataset {two_moons,cifar10h}   required
    --encoder <name>                 default: siglip2 (ignored on two_moons)
    --num-members <int>              default: 10
    --epochs <int>                   default: 200
    --alpha <float>                  default: 0.1
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

from probly.calibrator import calibrate
from probly.method.calibration import temperature_scaling, torch_identity_logit_model
from probly.method.conformal import conformal_raps
from probly.method.credal_wrapper import credal_wrapper
from probly.metrics import average_set_size, empirical_coverage_classification
from probly.predictor import predict, predict_raw

from stacking.data import load_dataset
from stacking.models import build_mlp
from stacking.utils import get_device, set_seed


def _write_results(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON dict to ``path``, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"\nwrote results -> {path}")


def _train_member(
    member: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    device: torch.device,
) -> None:
    """Train one ensemble member with full-batch CE + Adam(lr=1e-2)."""
    member.train()
    member.to(device)
    opt = torch.optim.Adam(member.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = member(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()


@torch.no_grad()
def _pool_logits(ensemble: nn.ModuleList, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Mean of member logits on X. Shape (B, num_classes)."""
    stacked: list[torch.Tensor] = []
    for member in ensemble:
        member.eval()
        member.to(device)
        stacked.append(member(X.to(device)))
    return torch.stack(stacked, dim=0).mean(dim=0).detach()


def _conformal_metrics(
    *,
    calib_logits: torch.Tensor,
    test_logits: torch.Tensor,
    y_calib: torch.Tensor,
    y_test: torch.Tensor,
    alpha: float,
    use_temperature: bool,
) -> tuple[float, float]:
    """Build a logit-side conformal RAPS predictor and return (coverage, avg_set_size).

    The base of the conformal wrapper is either ``torch_identity_logit_model``
    directly (path A, no temperature scaling) or
    ``temperature_scaling(torch_identity_logit_model())`` after fitting on
    the calibration logits (path B). Both paths are then conformalised at
    the same alpha and evaluated on the test split.
    """
    if use_temperature:
        base: Any = temperature_scaling(torch_identity_logit_model())
        calibrate(base, y_calib, calib_logits)
    else:
        base = torch_identity_logit_model()

    conformalizer: Any = conformal_raps(base)
    calibrate(conformalizer, alpha, y_calib, calib_logits)
    test_sets = predict(conformalizer, test_logits)
    coverage = float(empirical_coverage_classification(test_sets, y_test))
    avg_size = float(average_set_size(test_sets))
    return coverage, avg_size


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Top-1 accuracy from raw logits."""
    return float((logits.argmax(dim=-1) == y).float().mean().item())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["two_moons", "cifar10h"], required=True)
    parser.add_argument("--encoder", default="siglip2", help="Ignored for two_moons.")
    parser.add_argument("--num-members", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
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
    """Run credal-wrapper -> conformal RAPS, ablating temperature scaling."""
    args = _parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    ds = load_dataset(args.dataset, encoder=args.encoder, seed=args.seed)
    print(f"Dataset: {ds.meta.get('name')}  in_features={ds.in_features}  num_classes={ds.num_classes}")
    print(f"  train={ds.X_train.shape[0]}  calib={ds.X_calib.shape[0]}  test={ds.X_test.shape[0]}")

    X_train = _to_tensors(ds.X_train, dtype=torch.float32, device=device)
    y_train = _to_tensors(ds.y_train, dtype=torch.long, device=device)
    X_calib = _to_tensors(ds.X_calib, dtype=torch.float32, device=device)
    y_calib = _to_tensors(ds.y_calib, dtype=torch.long, device=device)
    X_test = _to_tensors(ds.X_test, dtype=torch.float32, device=device)
    y_test = _to_tensors(ds.y_test, dtype=torch.long, device=device)

    base = build_mlp(in_features=ds.in_features, num_classes=ds.num_classes)
    ensemble: nn.ModuleList = credal_wrapper(base, num_members=args.num_members)  # ty: ignore[invalid-assignment]

    for i, member in enumerate(ensemble):
        print(f"  training member {i + 1}/{args.num_members} ...")
        _train_member(member, X_train, y_train, epochs=args.epochs, device=device)

    calib_logits = _pool_logits(ensemble, X_calib, device=device)
    test_logits = _pool_logits(ensemble, X_test, device=device)
    pooled_acc = _accuracy(test_logits, y_test)
    print(f"\npooled-ensemble test_acc={pooled_acc:.4f}")

    cov_a, size_a = _conformal_metrics(
        calib_logits=calib_logits,
        test_logits=test_logits,
        y_calib=y_calib,
        y_test=y_test,
        alpha=args.alpha,
        use_temperature=False,
    )
    cov_b, size_b = _conformal_metrics(
        calib_logits=calib_logits,
        test_logits=test_logits,
        y_calib=y_calib,
        y_test=y_test,
        alpha=args.alpha,
        use_temperature=True,
    )

    print(f"\nA. pooled       -> conformal_raps @ alpha={args.alpha}: coverage={cov_a:.3f}  avg_set_size={size_a:.3f}")
    print(f"B. pooled -> T  -> conformal_raps @ alpha={args.alpha}: coverage={cov_b:.3f}  avg_set_size={size_b:.3f}")
    print(f"\ndelta (B - A): coverage={cov_b - cov_a:+.3f}  avg_set_size={size_b - size_a:+.3f}")

    if args.results_json is not None:
        _write_results(
            args.results_json,
            {
                "composition": "credal_wrapper_temp_conformal",
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
                    "alpha": args.alpha,
                    "seed": args.seed,
                },
                "metrics": {
                    "pooled_acc": pooled_acc,
                    "no_temperature": {
                        "coverage": cov_a,
                        "avg_set_size": size_a,
                    },
                    "with_temperature": {
                        "coverage": cov_b,
                        "avg_set_size": size_b,
                    },
                    "delta_with_minus_no": {
                        "coverage": cov_b - cov_a,
                        "avg_set_size": size_b - size_a,
                    },
                },
            },
        )

    # Silence unused-warning for the predict_raw import (kept for parity with sibling scripts).
    _ = predict_raw


if __name__ == "__main__":
    main()
