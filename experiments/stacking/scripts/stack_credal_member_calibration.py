"""Experiment ``credal_member_calibration``: per-member temperature calibration of a credal-wrapper ensemble.

Trains an N-member ensemble via :func:`probly.method.credal_wrapper.credal_wrapper`
on cached CIFAR-10-H embeddings, then asks: does temperature-scaling
**each member independently** before forming the credal probability
intervals shrink the resulting non-dominated prediction sets, and does
coverage hold?

The existing :mod:`stack_credal_wrapper_temp_conformal` script already
applies temperature scaling, but only after pooling the member logits;
that's a different operation than per-member-then-pool. This script
computes, per test point and per path, the credal probability interval
schema used by probly's
:class:`probly.representation.credal_set.torch.TorchProbabilityIntervalsCredalSet`
(``lower(k) = min_i softmax(member_logits_i)[:, k]``, ``upper(k) =
max_i ...``), then emits the canonical naive-credal-classifier set

    ``set(x) = {k : upper(k) >= max_j lower(j)}``,

and reports two scalars per path: ``coverage`` (fraction of test points
whose true hard label lands in the set) and ``avg_set_size``
(efficiency; smaller is more decisive). A diagnostic ``singleton_rate``
is also reported.

Pass ``--results-json <path>`` to write the run's hyperparams and
metrics to disk under the uniform stack-experiments schema; both paths
sit under ``metrics`` so each per-seed file is self-contained.
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
from probly.method.credal_wrapper import credal_wrapper
from probly.predictor import predict_raw

from stacking.data import load_dataset
from stacking.models import build_mlp
from stacking.utils import get_device, set_seed

EXPERIMENT = "credal_member_calibration"
DECISION_RULE = "non_dominated"


def _train_member(
    member: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    """Train one ensemble member with full-batch CE + Adam.

    Mirrors the loop in ``stack_credal_wrapper_temp_conformal.py`` so
    the two scripts share an identical training-time object when given
    the same seed. The credal_wrapper factory already reset-init'd the
    member; this just optimises it from that fresh state.
    """
    member.train()
    member.to(device)
    opt = torch.optim.Adam(member.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = member(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()


@torch.no_grad()
def _stack_member_logits(
    ensemble: nn.ModuleList, x: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Return per-member logits stacked along a new leading axis.

    Output shape: ``(N, B, num_classes)``.
    """
    out: list[torch.Tensor] = []
    for member in ensemble:
        member.eval()
        member.to(device)
        out.append(member(x.to(device)).detach())
    return torch.stack(out, dim=0)


def _per_member_temperature(
    member_calib_logits: torch.Tensor,
    member_test_logits: torch.Tensor,
    y_calib: torch.Tensor,
) -> torch.Tensor:
    """Fit one temperature scalar per member, apply to that member's test logits.

    Each member gets its own
    ``temperature_scaling(torch_identity_logit_model())`` wrapper fitted
    on ``(y_calib, member_calib_logits[i])``; the wrapper is then
    applied to ``member_test_logits[i]`` to produce that member's
    calibrated test logits. The N independent calibrators do not share
    parameters.

    Args:
        member_calib_logits: ``(N, B_calib, K)`` per-member calib logits.
        member_test_logits: ``(N, B_test, K)`` per-member test logits.
        y_calib: ``(B_calib,)`` integer labels on the calibration split.

    Returns:
        Calibrated test logits, shape ``(N, B_test, K)``.
    """
    n_members = member_calib_logits.shape[0]
    calibrated: list[torch.Tensor] = []
    for i in range(n_members):
        cal_i: Any = temperature_scaling(torch_identity_logit_model())
        calibrate(cal_i, y_calib, member_calib_logits[i])
        out = predict_raw(cal_i, member_test_logits[i])
        if not isinstance(out, torch.Tensor):
            msg = f"per-member temperature scaling returned {type(out)!r}, expected torch.Tensor"
            raise TypeError(msg)
        calibrated.append(out)
    return torch.stack(calibrated, dim=0)


def _credal_inclusion(probs: torch.Tensor) -> torch.Tensor:
    """Non-dominated inclusion mask from per-member probabilities.

    Args:
        probs: ``(N, B, K)`` per-member softmax probabilities.

    Returns:
        Boolean tensor of shape ``(B, K)``, where ``[b, k]`` is true iff
        class ``k`` is non-dominated for sample ``b`` -- i.e. its upper
        probability ``max_i probs[i, b, k]`` is at least as large as the
        largest lower probability ``max_j min_i probs[i, b, j]``.
    """
    lower = probs.amin(dim=0)  # (B, K)
    upper = probs.amax(dim=0)  # (B, K)
    threshold = lower.amax(dim=-1, keepdim=True)  # (B, 1)
    return upper >= threshold


def _credal_metrics(inclusion: torch.Tensor, y_test: torch.Tensor) -> dict[str, float]:
    """Coverage, average set size, and singleton rate from an inclusion mask.

    Args:
        inclusion: ``(B, K)`` boolean mask from :func:`_credal_inclusion`.
        y_test: ``(B,)`` integer labels.
    """
    batch = inclusion.shape[0]
    idx = torch.arange(batch, device=inclusion.device)
    coverage = inclusion[idx, y_test].float().mean().item()
    set_sizes = inclusion.float().sum(dim=-1)
    avg_set_size = set_sizes.mean().item()
    singleton_rate = (set_sizes == 1).float().mean().item()
    return {
        "coverage": float(coverage),
        "avg_set_size": float(avg_set_size),
        "singleton_rate": float(singleton_rate),
    }


def _pooled_argmax_acc(probs: torch.Tensor, y_test: torch.Tensor) -> float:
    """Sanity-check accuracy: argmax of the mean per-member probability.

    Args:
        probs: ``(N, B, K)`` per-member softmax probabilities.
        y_test: ``(B,)`` integer labels.
    """
    pooled = probs.mean(dim=0)
    return float((pooled.argmax(dim=-1) == y_test).float().mean().item())


def _write_results(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON dict to ``path``, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"\nwrote results -> {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cifar10h"], default="cifar10h")
    parser.add_argument("--encoder", default="siglip2")
    parser.add_argument("--num-members", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate per member.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-json", type=Path, default=None)
    return parser.parse_args()


def _to_tensors(arr: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=dtype, device=device)


def main() -> None:
    """Run credal_member_calibration on the chosen dataset / encoder."""
    args = _parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    ds = load_dataset(args.dataset, encoder=args.encoder, seed=args.seed)
    print(
        f"Dataset: {ds.meta.get('name')}  in_features={ds.in_features}  num_classes={ds.num_classes}  "
        f"seed={args.seed}"
    )
    print(f"  train={ds.X_train.shape[0]}  calib={ds.X_calib.shape[0]}  test={ds.X_test.shape[0]}")

    x_train = _to_tensors(ds.X_train, dtype=torch.float32, device=device)
    y_train = _to_tensors(ds.y_train, dtype=torch.long, device=device)
    x_calib = _to_tensors(ds.X_calib, dtype=torch.float32, device=device)
    y_calib = _to_tensors(ds.y_calib, dtype=torch.long, device=device)
    x_test = _to_tensors(ds.X_test, dtype=torch.float32, device=device)
    y_test = _to_tensors(ds.y_test, dtype=torch.long, device=device)

    # 1. Build the credal-wrapper ensemble (same as ensemble(...) at training
    #    time; only the inference-time representer differs, which we sidestep
    #    here by computing intervals manually).
    base = build_mlp(in_features=ds.in_features, num_classes=ds.num_classes)
    ensemble: nn.ModuleList = credal_wrapper(base, num_members=args.num_members)  # ty: ignore[invalid-assignment]

    # 2. Train each member with full-batch CE + Adam.
    for i, member in enumerate(ensemble):
        print(f"  training member {i + 1}/{args.num_members} ...")
        _train_member(member, x_train, y_train, epochs=args.epochs, lr=args.lr, device=device)

    # 3. Cache per-member calib + test logits.
    member_calib_logits = _stack_member_logits(ensemble, x_calib, device=device)  # (N, B_c, K)
    member_test_logits = _stack_member_logits(ensemble, x_test, device=device)  # (N, B_t, K)

    # 4. Path A (raw): per-member softmax, no calibration.
    probs_test_raw = torch.softmax(member_test_logits, dim=-1)
    inclusion_raw = _credal_inclusion(probs_test_raw)
    metrics_raw = _credal_metrics(inclusion_raw, y_test)

    # 5. Path B (per-member temperature scaling): fit a fresh T_i for each
    #    member on its own calibration logits; apply to its own test logits.
    member_test_logits_calibrated = _per_member_temperature(
        member_calib_logits=member_calib_logits,
        member_test_logits=member_test_logits,
        y_calib=y_calib,
    )
    probs_test_cal = torch.softmax(member_test_logits_calibrated, dim=-1)
    inclusion_cal = _credal_inclusion(probs_test_cal)
    metrics_cal = _credal_metrics(inclusion_cal, y_test)

    # 6. Sanity-check pooled-argmax accuracy on the raw path.
    pooled_acc = _pooled_argmax_acc(probs_test_raw, y_test)

    # 7. Print a tight 2-row table + deltas.
    print(f"\npooled-argmax test_acc={pooled_acc:.4f}  num_classes={ds.num_classes}  num_members={args.num_members}")
    header = f"{'path':<26} {'coverage':>10} {'avg_set':>10} {'singleton':>10}"
    print(header)
    print("-" * len(header))
    print(
        f"{'A. no calibration':<26} "
        f"{metrics_raw['coverage']:>10.4f} {metrics_raw['avg_set_size']:>10.4f} "
        f"{metrics_raw['singleton_rate']:>10.4f}"
    )
    print(
        f"{'B. per-member T scaling':<26} "
        f"{metrics_cal['coverage']:>10.4f} {metrics_cal['avg_set_size']:>10.4f} "
        f"{metrics_cal['singleton_rate']:>10.4f}"
    )
    delta = {
        key: metrics_cal[key] - metrics_raw[key]
        for key in ("coverage", "avg_set_size", "singleton_rate")
    }
    print(
        f"{'delta (B - A)':<26} "
        f"{delta['coverage']:>+10.4f} {delta['avg_set_size']:>+10.4f} "
        f"{delta['singleton_rate']:>+10.4f}"
    )

    if args.results_json is not None:
        _write_results(
            args.results_json,
            {
                "experiment": EXPERIMENT,
                "calibration": "per_member_temperature_vs_none",
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
                    "lr": args.lr,
                    "decision_rule": DECISION_RULE,
                },
                "metrics": {
                    "test_acc_pooled_argmax": pooled_acc,
                    "no_calibration": metrics_raw,
                    "per_member_calibration": metrics_cal,
                    "delta_with_minus_no": delta,
                },
            },
        )


if __name__ == "__main__":
    main()
