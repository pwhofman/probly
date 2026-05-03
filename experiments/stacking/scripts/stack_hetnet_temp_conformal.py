"""Composition #4: single HetNet -> temperature scaling -> conformal RAPS.

Wraps a small MLP with probly's heteroscedastic head (``het_net``),
trains end-to-end with the standard HetNet loss (NLL on the average
softmax over MC samples), then averages MC samples at inference to
form a single ``(B, num_classes)`` probability vector. The
log-probability vector is fed through the same calibration / conformal
chain used by the DARE script: ``temperature_scaling`` fitted on the
calibration logits, then ``conformal_raps`` calibrated at level alpha.

CLI:

    --dataset {two_moons,cifar10h}   required
    --encoder <name>                 default: siglip2 (ignored on two_moons)
    --num-factors <int>              default: 10
    --train-samples <int>            default: 10
    --inf-samples <int>              default: 10
    --epochs <int>                   default: 200
    --alpha <float>                  default: 0.1
    --seed <int>                     default: 0
    --device <str>                   default: auto
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from probly.calibrator import calibrate
from probly.method.calibration import temperature_scaling, torch_identity_logit_model
from probly.method.conformal import conformal_raps
from probly.method.het_net import het_net
from probly.metrics import average_set_size, empirical_coverage_classification
from probly.predictor import predict, predict_raw

from stacking.data import load_dataset
from stacking.models import build_mlp
from stacking.utils import get_device, set_seed

ECE_BINS = 15


def _train_hetnet(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    samples: int,
    lr: float,
    grad_clip: float,
    device: torch.device,
) -> None:
    """Train a HetNet end-to-end with NLL on mean-of-MC-softmax.

    Mirrors :func:`probly_benchmark.train_funcs.train_epoch_het_net` minus
    the optimizer/AMP plumbing: average ``S`` softmax samples per step
    (each forward draws a fresh noise sample from the het head), then
    NLL. Adam + global gradient clipping keeps the heteroscedastic head's
    low-rank covariance from blowing up on differently scaled feature
    distributions (raw DINOv2 embeddings, in particular, diverge with
    the larger learning rates that work for SigLIP2).
    """
    model.train()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        avg_probs = torch.stack(
            [F.softmax(model(X), dim=1) for _ in range(samples)],
        ).mean(0)
        loss = F.nll_loss(avg_probs.log(), y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()


@torch.no_grad()
def _hetnet_mean_logits(
    model: nn.Module, X: torch.Tensor, *, samples: int, device: torch.device
) -> torch.Tensor:
    """Average ``samples`` MC predictive probabilities and return them as logits.

    HetNet draws fresh noise per forward call, so we sample ``S`` times,
    average the softmax, then take ``log`` to recover a logit-space tensor
    of shape ``(B, num_classes)`` consumable by ``temperature_scaling``.
    A small floor avoids ``log(0)`` on classes the ensemble never picks.
    """
    model.eval()
    model.to(device)
    chunks: list[torch.Tensor] = []
    for _ in range(samples):
        logits = model(X.to(device))
        chunks.append(F.softmax(logits, dim=-1).detach())
    mean_probs = torch.stack(chunks, dim=0).mean(dim=0).clamp(min=1e-12)
    return mean_probs.log()


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["two_moons", "cifar10h"], required=True)
    parser.add_argument("--encoder", default="siglip2", help="Ignored for two_moons.")
    parser.add_argument("--num-factors", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=10)
    parser.add_argument("--inf-samples", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _to_tensors(arr: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=dtype, device=device)


def main() -> None:
    """Run a single HetNet -> temp -> conformal RAPS on the chosen dataset."""
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

    # 1. Build a base MLP and wrap its last logit layer with the HetNet head.
    base = build_mlp(in_features=ds.in_features, num_classes=ds.num_classes)
    model: nn.Module = het_net(base, num_factors=args.num_factors)  # ty: ignore[invalid-assignment]

    # 2. Train end-to-end with the HetNet MC-softmax loss.
    print("  training HetNet ...")
    _train_hetnet(
        model,
        X_train,
        y_train,
        epochs=args.epochs,
        samples=args.train_samples,
        lr=args.lr,
        grad_clip=args.grad_clip,
        device=device,
    )

    # 3. Form mean-MC logits on calib + test.
    calib_logits = _hetnet_mean_logits(model, X_calib, samples=args.inf_samples, device=device)
    test_logits = _hetnet_mean_logits(model, X_test, samples=args.inf_samples, device=device)

    acc = _accuracy(test_logits, y_test)
    ece_uncal = _ece(test_logits, y_test, n_bins=ECE_BINS)
    print(f"\nuncalibrated HetNet: test_acc={acc:.4f}  ECE={ece_uncal:.4f}")

    # 4. Temperature scaling on cached calib logits.
    calibrator: Any = temperature_scaling(torch_identity_logit_model())
    calibrate(calibrator, y_calib, calib_logits)
    test_logits_calibrated = predict_raw(calibrator, test_logits)
    if not isinstance(test_logits_calibrated, torch.Tensor):
        msg = f"expected torch.Tensor calibrated logits, got {type(test_logits_calibrated)}"
        raise TypeError(msg)
    ece_cal = _ece(test_logits_calibrated, y_test, n_bins=ECE_BINS)
    print(f"calibrated HetNet:    test_acc={_accuracy(test_logits_calibrated, y_test):.4f}  ECE={ece_cal:.4f}")

    # 5. Conformal RAPS wrapped around the calibrator.
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
