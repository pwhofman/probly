"""Explain the sub-ensemble's prediction for an image: Shapley values of every class on shared coalitions.

Superpixels of ``--patch-size`` pixels are the players. Each sampled coalition costs one pass of the
frozen encoder; the logits of all heads are stored, so that every readout (class probabilities,
logits, uncertainty measures, single heads) is explained from the same evaluations, and so that
explanations can be recomputed later without the model.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
import torch
import torchvision.models as tm
from uqxai.classes import CLASS_GROUPS, CLASS_NAMES
from uqxai.explain import shapley_values
from uqxai.game import CoalitionCache, SuperpixelMasker
from uqxai.model import IMAGENET_TRANSFORM, encode, encoder_of, head_logits, load_subensemble
from uqxai.paths import HEADS_CHECKPOINT, RESULT_DIR, RIBEIRO_IMAGE
from uqxai.readouts import READOUTS

from probly.quantification import quantify
from probly.representer import representer

if TYPE_CHECKING:
    from pathlib import Path


def open_world_probabilities(features: torch.Tensor, classifier: torch.nn.Module) -> torch.Tensor:
    """Return the group probabilities of the original 1000-class ImageNet classifier.

    The three groups do not exhaust the 1000 classes, so their probabilities do not sum to one.
    """
    probabilities = classifier(features).softmax(-1)
    return torch.stack([probabilities[:, list(members)].sum(1) for members in CLASS_GROUPS.values()], dim=1)


def run_name(image: Path, patch_size: int, baseline: str) -> str:
    """Return the name of the result directory of a configuration."""
    return f"{image.stem}_patch{patch_size}_{baseline}"


def main() -> None:
    """Evaluate the coalitions, compute the attributions of all readouts, and save them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=type(RIBEIRO_IMAGE), default=RIBEIRO_IMAGE)
    parser.add_argument("--patch-size", type=int, default=16, help="side length of a superpixel in pixels")
    parser.add_argument("--budget", type=int, default=10_000, help="number of coalitions per explanation")
    parser.add_argument("--baseline", choices=["mean", "blur"], default="mean")
    parser.add_argument("--approximator", choices=["oddshap", "kernelshap"], default="oddshap")
    parser.add_argument("--readouts", nargs="+", default=["probability", "mean_logit", "uncertainty", "open_world"])
    parser.add_argument(
        "--seed", type=int, default=0, help="seed of the coalition sampling; other seeds add to the cache"
    )
    args = parser.parse_args()

    out_dir = RESULT_DIR / run_name(args.image, args.patch_size, args.baseline)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "coalitions.npz"

    model = load_subensemble(HEADS_CHECKPOINT)
    encoder = encoder_of(model)
    imagenet_classifier = tm.resnet50(weights="DEFAULT").fc.eval()
    image = IMAGENET_TRANSFORM(Image.open(args.image).convert("RGB"))
    masker = SuperpixelMasker(image, args.patch_size, baseline=args.baseline)

    @torch.no_grad()
    def evaluate(coalitions: np.ndarray) -> dict[str, np.ndarray]:
        features = encode(encoder, masker(coalitions))
        return {
            "head_logits": head_logits(model, features).numpy(),
            "open_world": open_world_probabilities(features, imagenet_classifier).numpy(),
        }

    cache = (
        CoalitionCache.load(cache_path, evaluate) if cache_path.exists() else CoalitionCache(masker.n_players, evaluate)
    )
    print(f"{masker.n_players} players ({masker.grid_shape[0]}x{masker.grid_shape[1]}), {len(cache)} cached coalitions")

    # The prediction for the full image, through probly: a sample of categorical distributions.
    with torch.no_grad():
        sample = representer(model).predict(image[None])
        uncertainty = quantify(sample)
        member_probabilities = torch.stack([member(image[None]).softmax(-1)[0] for member in model])
    summary = {
        "image": args.image.name,
        "patch_size": args.patch_size,
        "grid_shape": masker.grid_shape,
        "n_players": masker.n_players,
        "baseline": args.baseline,
        "budget": args.budget,
        "approximator": args.approximator,
        "seed": args.seed,
        "classes": CLASS_NAMES,
        "prediction": dict(zip(CLASS_NAMES, member_probabilities.mean(0).tolist(), strict=True)),
        "total_uncertainty": uncertainty.total.item(),
        "aleatoric_uncertainty": uncertainty.aleatoric.item(),
        "epistemic_uncertainty": uncertainty.epistemic.item(),
        "readouts": {},
    }

    results: dict[str, np.ndarray] = {"member_probabilities": member_probabilities.numpy()}
    for name in args.readouts:
        if name == "open_world":
            game = lambda coalitions: cache(coalitions)["open_world"]  # noqa: E731
        else:
            game = lambda coalitions, readout=READOUTS[name]: readout(cache(coalitions)["head_logits"])  # noqa: E731
        attribution = shapley_values(
            game, masker.n_players, args.budget, approximator=args.approximator, random_state=args.seed
        )
        cache.save(cache_path)
        results[f"{name}_values"] = attribution.values
        results[f"{name}_empty"] = attribution.empty
        results[f"{name}_full"] = attribution.full
        summary["readouts"][name] = {
            "empty": attribution.empty.tolist(),
            "full": attribution.full.tolist(),
            "largest_attribution": float(np.abs(attribution.values).max()),
            # Zero for exact Shapley values of class probabilities: an estimate of the approximation error.
            "largest_sum_over_outputs": float(np.abs(attribution.values.sum(axis=1)).max()),
        }
        print(name, json.dumps(summary["readouts"][name]))
    suffix = f"_seed{args.seed}" if args.seed else ""
    np.savez_compressed(out_dir / f"attributions{suffix}.npz", cells=masker.cells.numpy(), **results)
    (out_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(cache)} coalitions evaluated -> {out_dir}")


if __name__ == "__main__":
    main()
