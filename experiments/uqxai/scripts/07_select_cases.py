"""Select test images for three cases of predictive uncertainty, from the ImageNet validation images.

- ``certain``: one clear golden retriever. No model is uncertain about it.
- ``conflict``: the dog-with-guitar image. The models are uncertain because different regions of the
  image speak for different classes.
- ``ambiguous``: per problem with two similar classes, the validation image whose prediction is
  closest to an even split between them. The model is uncertain because the same object fits two
  classes. Images that show a person or more than one instrument are excluded by hand.

The images are chosen from the cached encoder features and the trained heads, without any
explanation, and written to ``assets/cases`` with a manifest.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import torch
from uqxai import readouts
from uqxai.classes import GOLDEN_RETRIEVER, PROBLEMS
from uqxai.data import class_file
from uqxai.model import head_logits, load_subensemble
from uqxai.paths import ASSET_DIR, CASES_MANIFEST, DATA_DIR, RIBEIRO_IMAGE, heads_checkpoint

# Problems with two similar classes, and the output indices of the two.
SIMILAR_CLASSES = {"electric": (0, 2), "labrador": (1, 2)}
# Validation images (ImageNet class, index) that are not used although their split is more even.
EXCLUDED = {
    (402, 36): "shows a person",
    (546, 35): "several instruments",
    (546, 19): "an acoustic and an electric guitar side by side",
    (402, 22): "two instruments",
    (402, 10): "shows a person",
}


def predictions(problem: str, imagenet_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a model's probabilities and total uncertainty for the validation images of a class."""
    model = load_subensemble(heads_checkpoint(problem), pretrained=False)  # only the heads are used
    logits = head_logits(model, torch.load(DATA_DIR / "features" / "validation" / f"{imagenet_class}.pt")).numpy()
    return readouts.probability(logits), readouts.uncertainty(logits)[:, 0]


def save_image(name: str, imagenet_class: int, index: int) -> str:
    """Write a validation image to the assets and return its path relative to them."""
    images = pq.read_table(class_file(DATA_DIR, "validation", imagenet_class)).column("image")
    path = ASSET_DIR / "cases" / f"{name}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(images[index].as_py())
    return str(path.relative_to(ASSET_DIR))


def main() -> None:
    """Select the images and write the manifest."""
    cases = []
    # Certain: the golden retriever with the lowest uncertainty in its most uncertain model.
    uncertainty = np.max([predictions(problem, GOLDEN_RETRIEVER)[1] for problem in SIMILAR_CLASSES], axis=0)
    certain = int(uncertainty.argmin())
    image = save_image("certain_retriever", GOLDEN_RETRIEVER, certain)
    for problem in SIMILAR_CLASSES:
        cases.append({"problem": problem, "case": "certain", "image": image})
        cases.append({"problem": problem, "case": "conflict", "image": str(RIBEIRO_IMAGE.relative_to(ASSET_DIR))})

    # Ambiguous: the most even split between the two similar classes, among their validation images.
    for problem, (first, second) in SIMILAR_CLASSES.items():
        best: tuple[float, int, int] = (-1.0, -1, -1)
        for imagenet_class in (PROBLEMS[problem].imagenet_classes[first], PROBLEMS[problem].imagenet_classes[second]):
            probabilities, _ = predictions(problem, imagenet_class)
            evenness = np.minimum(probabilities[:, first], probabilities[:, second])
            evenness[[index for excluded_class, index in EXCLUDED if excluded_class == imagenet_class]] = -1
            best = max(best, (float(evenness.max()), imagenet_class, int(evenness.argmax())))
        image = save_image(f"ambiguous_{problem}", best[1], best[2])
        cases.append({"problem": problem, "case": "ambiguous", "image": image, "true_imagenet_class": best[1]})

    CASES_MANIFEST.write_text(json.dumps(cases, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
