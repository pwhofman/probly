"""Explain the prediction of each problem's sub-ensemble for an image: Shapley values of every class.

Superpixels of ``--patch-size`` pixels are the players. Each sampled coalition costs one pass of the
frozen encoder; the logits of all heads are stored, so that every readout (class probabilities,
logits, uncertainty measures, single heads) is explained from the same evaluations, and so that
explanations can be recomputed later without the model.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image
import torch
import torchvision.models as tm
from uqxai.classes import PROBLEMS, Problem
from uqxai.explain import shapley_values
from uqxai.game import CoalitionCache, SuperpixelMasker
from uqxai.model import IMAGENET_TRANSFORM, encode, encoder_of, head_logits, load_subensemble
from uqxai.paths import RIBEIRO_IMAGE, heads_checkpoint, run_dir
from uqxai.readouts import READOUTS

from probly.quantification import quantify
from probly.representer import representer


def explain(problem: Problem, args: argparse.Namespace, imagenet_classifier: torch.nn.Module) -> None:
    """Evaluate the coalitions for one problem, compute the attributions of all readouts, and save them."""
    out_dir = run_dir(problem.name, args.image, args.patch_size, args.baseline)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "coalitions.npz"

    model = load_subensemble(heads_checkpoint(problem.name), len(problem.classes))
    encoder = encoder_of(model)
    image = IMAGENET_TRANSFORM(Image.open(args.image).convert("RGB"))
    masker = SuperpixelMasker(image, args.patch_size, baseline=args.baseline)

    @torch.no_grad()
    def evaluate(coalitions: np.ndarray) -> dict[str, np.ndarray]:
        features = encode(encoder, masker(coalitions))
        return {
            "head_logits": head_logits(model, features).numpy(),
            # The same classes in the original 1000-class classifier: their probabilities do not sum to one.
            "open_world": imagenet_classifier(features).softmax(-1)[:, list(problem.imagenet_classes)].numpy(),
        }

    cache = CoalitionCache(masker.n_players, evaluate)
    if cache_path.exists():
        cache = CoalitionCache.load(cache_path, evaluate)
    print(f"{problem.name}: {masker.n_players} players, {len(cache)} cached coalitions")

    # The prediction for the full image, through probly: a sample of categorical distributions.
    with torch.no_grad():
        uncertainty = quantify(representer(model).predict(image[None]))
        member_probabilities = torch.stack([member(image[None]).softmax(-1)[0] for member in model])
    summary = {
        "problem": problem.name,
        "image": args.image.name,
        "patch_size": args.patch_size,
        "grid_shape": masker.grid_shape,
        "n_players": masker.n_players,
        "baseline": args.baseline,
        "budget": args.budget,
        "approximator": args.approximator,
        "seed": args.seed,
        "classes": problem.class_names,
        "prediction": dict(zip(problem.class_names, member_probabilities.mean(0).tolist(), strict=True)),
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
        print(" ", name, json.dumps(summary["readouts"][name]))
    suffix = f"_seed{args.seed}" if args.seed else ""
    np.savez_compressed(out_dir / f"attributions{suffix}.npz", cells=masker.cells.numpy(), **results)
    (out_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  {len(cache)} coalitions evaluated -> {out_dir}")


def main() -> None:
    """Explain the image for every chosen problem."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS), default=list(PROBLEMS))
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
    imagenet_classifier = tm.resnet50(weights="DEFAULT").fc.eval()
    for name in args.problems:
        explain(PROBLEMS[name], args, imagenet_classifier)


if __name__ == "__main__":
    main()
