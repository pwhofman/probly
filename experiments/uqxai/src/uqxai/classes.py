"""The three-class problems: triples of ImageNet-1k classes, one sub-ensemble per triple.

All problems share acoustic guitar and golden retriever, the two classes that the 1000-class
ResNet-50 ranks first for the dog-with-guitar image, and differ in their third class:

- ``electric``: electric guitar, the third-ranked class, which shares features with the first.
- ``banjo``: another class that is close to the acoustic guitar.
- ``hat``: cowboy hat, a third object of the image with its own region.
- ``labrador``: Labrador retriever, a class that is close to the second class instead of the first.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    """A classification problem over a few ImageNet-1k classes.

    Attributes:
        name: Name of the problem; its results are in the directory of this name.
        classes: ImageNet-1k class index of each class, by class name. The order of the classes is
            the order of the model's outputs.
    """

    name: str
    classes: dict[str, int]

    @property
    def class_names(self) -> tuple[str, ...]:
        """Names of the model's output classes, in output order."""
        return tuple(self.classes)

    @property
    def imagenet_classes(self) -> tuple[int, ...]:
        """ImageNet-1k class indices of the model's output classes, in output order."""
        return tuple(self.classes.values())


ACOUSTIC_GUITAR, GOLDEN_RETRIEVER = 402, 207
_SHARED = {"acoustic guitar": ACOUSTIC_GUITAR, "golden retriever": GOLDEN_RETRIEVER}

PROBLEMS: dict[str, Problem] = {
    problem.name: problem
    for problem in (
        Problem("electric", {**_SHARED, "electric guitar": 546}),
        Problem("banjo", {**_SHARED, "banjo": 420}),
        Problem("hat", {**_SHARED, "cowboy hat": 515}),
        Problem("labrador", {**_SHARED, "Labrador retriever": 208}),
    )
}
"""All problems by name."""
