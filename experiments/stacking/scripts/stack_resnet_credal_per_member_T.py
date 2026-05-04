"""Experiment ``resnet_credal_per_member_T``: real-trained ResNet-18 deep ensemble
loaded as a credal-wrapper, per-member temperature scaling vs none, calibrated on
held-out CIFAR-10 training data, evaluated on CIFAR-10-H test set.

Calibration set
    The training run held out ``val_split=0.1`` of CIFAR-10 train (5000 images
    the model never saw) via ``torch.utils.data.random_split`` with
    ``manual_seed=cfg.seed=1``. This script reproduces that exact split, so
    the calibration set matches the data the bench would call ``val`` for the
    same wandb run.

Evaluation set
    Full 10000-image CIFAR-10-H test split (Peterson et al. 2019), which is
    the same images as CIFAR-10's test split but with per-image soft labels
    aggregated from ~50 human votes. Soft targets ``h(k)`` lie in the
    probability simplex over 10 classes.

Pipeline
    1. Download checkpoint artifact ``ensemble_resnet18_cifar10_<seed>:latest``
       from ``probly/cifar10-benchmark`` (cached locally).
    2. Build ``credal_wrapper(ResNet18(num_classes=10), num_members=10,
       predictor_type="logit_classifier")`` and ``load_state_dict``.
    3. Forward calib (5000) and CIFAR-10-H test (10000) through all members,
       cache logits to ``cache/`` for reuse.
    4. Fit per-member temperature on calib hard-label NLL.
    5. For both paths (no-cal vs per-member-T), form per-member softmax,
       envelope ``(lower, upper)`` and the non-dominated set, then report
       multiple coverage notions against soft targets:

        - ``hard_coverage``: argmax of soft target in the credal set.
        - ``plurality_coverage``: same as hard_coverage in CIFAR-10-H since the
          plurality class is what people call the "true" hard label here.
        - ``per_class_interval_coverage``: fraction of (image, class) pairs
          where ``lower(k) <= h(k) <= upper(k)``.
        - ``joint_interval_coverage``: fraction of images where the human
          distribution lies entirely inside the credal envelope (all 10
          classes simultaneously).
        - ``avg_set_size``, ``singleton_rate`` for completeness.

Usage::

    uv run python scripts/stack_resnet_credal_per_member_T.py \\
        --device cpu \\
        --results-json results/resnet_credal_per_member_T/cifar10h_seed0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torchvision.transforms import v2 as T

from probly.calibrator import calibrate
from probly.datasets.torch import CIFAR10H
from probly.method.calibration import temperature_scaling, torch_identity_logit_model
from probly.method.credal_wrapper import credal_wrapper
from probly.predictor import predict_raw

from stacking.utils import get_device, set_seed

EXPERIMENT = "resnet_credal_per_member_T"
DECISION_RULE = "non_dominated"
NUM_CLASSES = 10
NUM_MEMBERS = 10

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

WANDB_ENTITY = "probly"
WANDB_PROJECT = "cifar10-benchmark"
DEFAULT_WANDB_RUN = "d9gv6f84"  # ensemble_resnet18_cifar10 seed=1
ARTIFACT_NAME = "ensemble_resnet18_cifar10_1"

VAL_SPLIT_FRAC = 0.1
TRAINING_SEED = 1  # cfg.seed of the wandb run; controls val_split RNG


# ---------------------------------------------------------------------------
# CIFAR ResNet-18 (inlined from probly_benchmark.resnet)
# ---------------------------------------------------------------------------


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class _CifarResNet18(nn.Module):
    """ResNet-18 for 32x32 inputs (kuangliu/pytorch-cifar layout)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(_BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * _BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ---------------------------------------------------------------------------
# Checkpoint download + cache
# ---------------------------------------------------------------------------


def _wandb_download(run_id: str, cache_dir: Path) -> Path:
    """Download the model artifact logged by ``run_id``; cache locally."""
    cached = cache_dir / f"{ARTIFACT_NAME}.pt"
    if cached.exists():
        print(f"  [wandb] reusing cache: {cached}")
        return cached

    import os
    cache_dir.mkdir(parents=True, exist_ok=True)
    wandb_cache = cache_dir / "wandb_cache"
    wandb_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_cache))
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_cache))
    os.environ.setdefault("WANDB_ARTIFACT_DIR", str(wandb_cache))
    os.environ.setdefault("WANDB_DIR", str(wandb_cache))

    import wandb

    api = wandb.Api(timeout=60)
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}")
    artifacts = list(run.logged_artifacts())
    if not artifacts:
        msg = f"wandb run {run_id} has no logged artifacts"
        raise RuntimeError(msg)
    art = artifacts[0]
    print(f"  [wandb] downloading {art.name} ({art.size / 1e6:.0f} MB) ...")
    art_dir = Path(art.download())
    pts = list(art_dir.glob("*.pt"))
    if len(pts) != 1:
        msg = f"expected 1 .pt in artifact dir, found {len(pts)}"
        raise RuntimeError(msg)
    cached.write_bytes(pts[0].read_bytes())
    print(f"  [wandb] cached -> {cached}")
    return cached


def _build_ensemble(num_members: int, num_classes: int) -> nn.ModuleList:
    """Build ``credal_wrapper(ResNet18(num_classes), num_members)``.

    ``credal_wrapper`` and ``ensemble`` produce the same training-time object
    (an ``nn.ModuleList`` of N freshly init'd copies), so the ensemble state
    dict from the bench loads cleanly.
    """
    base = _CifarResNet18(num_classes=num_classes)
    return credal_wrapper(base, num_members=num_members, predictor_type="logit_classifier")  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _cifar_test_transform() -> T.Compose:
    """The bench's ``TRANSFORMS_TEST['cifar10']`` (also used for cifar10h)."""
    return T.Compose([
        T.Resize((32, 32), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def _cifar10_val_split_loader(
    data_root: Path,
    batch_size: int,
    *,
    val_frac: float,
    training_seed: int,
) -> torch.utils.data.DataLoader:
    """Reproduce the bench's held-out ``val_split`` from CIFAR-10 train.

    Uses ``random_split`` with the same generator seed the bench used for
    training, so the returned subset is exactly the 5000 images that were
    excluded from the model's training set.
    """
    train = torchvision.datasets.CIFAR10(
        root=str(data_root), train=True, download=True, transform=_cifar_test_transform()
    )
    rng = torch.Generator().manual_seed(training_seed)
    val_len = int(len(train) * val_frac)
    cal_len = 0  # bench had cal_split=0
    train_len = len(train) - val_len - cal_len
    _, val_subset, _ = torch.utils.data.random_split(
        train, [train_len, val_len, cal_len], generator=rng
    )
    return torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)


def _cifar10h_test_loader(data_root: Path, batch_size: int) -> torch.utils.data.DataLoader:
    """Full CIFAR-10-H test set; ``__getitem__`` yields ``(image, soft_target)``."""
    ds = CIFAR10H(root=str(data_root), transform=_cifar_test_transform(), download=True)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------


@torch.no_grad()
def _forward_all_members(
    ensemble: nn.ModuleList,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward every member over ``loader``; return ``(logits, targets)``.

    ``targets`` is whatever the loader yields as the second element; for plain
    CIFAR-10 it's hard ``int64`` labels, for CIFAR-10-H it's a ``(B, 10)``
    soft distribution. The caller is responsible for interpreting it.
    """
    ensemble.to(device).eval()
    per_member: list[list[torch.Tensor]] = [[] for _ in range(len(ensemble))]
    targets: list[torch.Tensor] = []
    n_batches = len(loader)
    for batch_i, (x, y) in enumerate(loader):
        x = x.to(device)
        for i, member in enumerate(ensemble):
            per_member[i].append(member(x).detach().cpu())
        targets.append(y if isinstance(y, torch.Tensor) else torch.as_tensor(y))
        if (batch_i + 1) % 20 == 0 or batch_i + 1 == n_batches:
            print(f"    forward batch {batch_i + 1}/{n_batches}")
    logits = torch.stack([torch.cat(chunks, dim=0) for chunks in per_member], dim=0)
    return logits, torch.cat(targets, dim=0)


def _load_or_compute_logits(
    *,
    cache_path: Path,
    ensemble: nn.ModuleList,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache-aware wrapper around :func:`_forward_all_members`."""
    if cache_path.exists():
        print(f"  [logits/{label}] reusing cache: {cache_path}")
        blob = torch.load(cache_path, map_location="cpu", weights_only=True)
        return blob["logits"], blob["targets"]
    print(f"  [logits/{label}] computing per-member logits ...")
    logits, targets = _forward_all_members(ensemble, loader, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"logits": logits, "targets": targets}, cache_path)
    print(f"  [logits/{label}] cached -> {cache_path}  shape={tuple(logits.shape)}")
    return logits, targets


# ---------------------------------------------------------------------------
# Calibration + envelope + metrics
# ---------------------------------------------------------------------------


def _per_member_temperatures(
    member_calib_logits: torch.Tensor,
    y_calib: torch.Tensor,
) -> tuple[list[float], list[Any]]:
    """Fit a separate temperature on each member's calibration logits."""
    n = member_calib_logits.shape[0]
    Ts: list[float] = []
    cals: list[Any] = []
    for i in range(n):
        cal = temperature_scaling(torch_identity_logit_model())
        calibrate(cal, y_calib, member_calib_logits[i])
        Ts.append(float(cal.temperature.item()))
        cals.append(cal)
    return Ts, cals


def _apply_per_member_T(cals: list[Any], member_logits: torch.Tensor) -> torch.Tensor:
    out = [predict_raw(cals[i], member_logits[i]) for i in range(len(cals))]
    return torch.stack(out, dim=0)


def _envelope(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class min/max across members. ``probs`` shape ``(N, B, K)``."""
    return probs.amin(dim=0), probs.amax(dim=0)


def _credal_inclusion(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Non-dominated set rule. ``lower``, ``upper`` shape ``(B, K)``."""
    threshold = lower.amax(dim=-1, keepdim=True)
    return upper >= threshold


def _coverage_metrics(
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    inclusion: torch.Tensor,
    h_soft: torch.Tensor,
) -> dict[str, float]:
    """Coverage / efficiency for a single envelope path on CIFAR-10-H.

    ``h_soft`` is the per-image human distribution ``(B, K)``. The plurality
    (hard) label is its argmax.
    """
    plurality = h_soft.argmax(dim=-1)
    plurality_in_set = inclusion.gather(-1, plurality.unsqueeze(-1)).squeeze(-1).float().mean()
    inside = (h_soft >= lower) & (h_soft <= upper)        # (B, K)
    per_class_interval = inside.float().mean()             # over B*K
    joint_interval = inside.all(dim=-1).float().mean()     # over B
    sizes = inclusion.sum(dim=-1).float()
    return {
        "plurality_coverage": float(plurality_in_set.item()),
        "per_class_interval_coverage": float(per_class_interval.item()),
        "joint_interval_coverage": float(joint_interval.item()),
        "avg_set_size": float(sizes.mean().item()),
        "singleton_rate": float((sizes == 1).float().mean().item()),
    }


# ---------------------------------------------------------------------------
# IO + CLI
# ---------------------------------------------------------------------------


def _write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"\nwrote results -> {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-run", default=DEFAULT_WANDB_RUN)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent.parent / "cache")
    parser.add_argument("--seed", type=int, default=0, help="logging seed only; calib split is fixed by training-seed.")
    parser.add_argument("--training-seed", type=int, default=TRAINING_SEED, help="cfg.seed of the wandb run.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--results-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Calibrate per-member T on CIFAR-10 val_split, eval credal coverage on CIFAR-10-H."""
    args = _parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    ckpt_path = _wandb_download(args.wandb_run, args.cache_dir / "wandb_artifacts")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    num_members = cfg.get("method", {}).get("params", {}).get("num_members", NUM_MEMBERS)
    print(f"Checkpoint: base={cfg.get('base_model')}  method={cfg.get('method', {}).get('name')}  "
          f"num_members={num_members}  dataset={cfg.get('dataset')}")

    ensemble = _build_ensemble(num_members=num_members, num_classes=NUM_CLASSES)
    ensemble.load_state_dict(ckpt["model_state_dict"], strict=True)

    calib_loader = _cifar10_val_split_loader(
        args.cache_dir, args.batch_size, val_frac=VAL_SPLIT_FRAC, training_seed=args.training_seed,
    )
    test_loader = _cifar10h_test_loader(args.cache_dir, args.batch_size)
    print(f"  calib (val_split={VAL_SPLIT_FRAC}, seed={args.training_seed}): "
          f"{len(calib_loader.dataset)} images")  # ty: ignore[unresolved-attribute]
    print(f"  eval (cifar10h test): {len(test_loader.dataset)} images")  # ty: ignore[unresolved-attribute]

    cal_logits, y_calib_hard = _load_or_compute_logits(
        cache_path=args.cache_dir / f"{ARTIFACT_NAME}_calib_logits.pt",
        ensemble=ensemble, loader=calib_loader, device=device, label="calib",
    )
    eval_logits, eval_h_soft = _load_or_compute_logits(
        cache_path=args.cache_dir / f"{ARTIFACT_NAME}_cifar10h_logits.pt",
        ensemble=ensemble, loader=test_loader, device=device, label="cifar10h_test",
    )
    print(f"  calib logits {tuple(cal_logits.shape)}   eval logits {tuple(eval_logits.shape)}")

    pooled_probs = F.softmax(eval_logits, dim=-1).mean(dim=0)
    pooled_acc_plurality = float(
        (pooled_probs.argmax(dim=-1) == eval_h_soft.argmax(dim=-1)).float().mean().item()
    )
    print(f"  pooled-argmax accuracy on plurality label: {pooled_acc_plurality:.4f}")

    Ts, cals = _per_member_temperatures(cal_logits, y_calib_hard.long())
    print(f"  fitted T per member: mean={float(np.mean(Ts)):.4f}  std={float(np.std(Ts)):.4f}  "
          f"range=[{min(Ts):.4f}, {max(Ts):.4f}]")

    # Path A (no calibration)
    probs_raw = F.softmax(eval_logits, dim=-1)
    lower_A, upper_A = _envelope(probs_raw)
    inc_A = _credal_inclusion(lower_A, upper_A)
    metrics_A = _coverage_metrics(lower=lower_A, upper=upper_A, inclusion=inc_A, h_soft=eval_h_soft)

    # Path B (per-member T)
    eval_logits_T = _apply_per_member_T(cals, eval_logits)
    probs_T = F.softmax(eval_logits_T, dim=-1)
    lower_B, upper_B = _envelope(probs_T)
    inc_B = _credal_inclusion(lower_B, upper_B)
    metrics_B = _coverage_metrics(lower=lower_B, upper=upper_B, inclusion=inc_B, h_soft=eval_h_soft)

    delta = {k: metrics_B[k] - metrics_A[k] for k in metrics_A}

    cols = ("plurality_coverage", "per_class_interval_coverage", "joint_interval_coverage",
            "avg_set_size", "singleton_rate")
    print("\npath                          plur_cov  pcov_iv  joint_iv  avg_set  singleton")
    print("-" * 80)
    print(f"A. no calibration             {metrics_A[cols[0]]:>8.4f}  "
          f"{metrics_A[cols[1]]:>7.4f}  {metrics_A[cols[2]]:>8.4f}  "
          f"{metrics_A[cols[3]]:>7.4f}  {metrics_A[cols[4]]:>8.4f}")
    print(f"B. per-member T scaling       {metrics_B[cols[0]]:>8.4f}  "
          f"{metrics_B[cols[1]]:>7.4f}  {metrics_B[cols[2]]:>8.4f}  "
          f"{metrics_B[cols[3]]:>7.4f}  {metrics_B[cols[4]]:>8.4f}")
    print(f"delta (B - A)                {delta[cols[0]]:>+8.4f}  "
          f"{delta[cols[1]]:>+7.4f}  {delta[cols[2]]:>+8.4f}  "
          f"{delta[cols[3]]:>+7.4f}  {delta[cols[4]]:>+8.4f}")

    if args.results_json is not None:
        _write_results(
            args.results_json,
            {
                "experiment": EXPERIMENT,
                "calibration": "per_member_temperature_vs_none",
                "dataset": "cifar10h",
                "encoder": None,
                "base_model": cfg.get("base_model"),
                "wandb_run": args.wandb_run,
                "training_seed": args.training_seed,
                "seed": args.seed,
                "splits": {
                    "calib": int(cal_logits.shape[1]),
                    "eval": int(eval_logits.shape[1]),
                    "calib_source": f"cifar10_train_val_split_{VAL_SPLIT_FRAC}_seed{args.training_seed}",
                    "eval_source": "cifar10h_test_full",
                },
                "hyperparams": {
                    "num_members": num_members,
                    "decision_rule": DECISION_RULE,
                },
                "metrics": {
                    "test_acc_pooled_argmax_plurality": pooled_acc_plurality,
                    "fitted_T_mean": float(np.mean(Ts)),
                    "fitted_T_std": float(np.std(Ts)),
                    "fitted_T_min": float(min(Ts)),
                    "fitted_T_max": float(max(Ts)),
                    "no_calibration": metrics_A,
                    "per_member_calibration": metrics_B,
                    "delta_with_minus_no": delta,
                },
            },
        )


if __name__ == "__main__":
    main()
