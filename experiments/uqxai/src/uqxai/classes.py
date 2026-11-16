"""The three-class problem: groups of ImageNet-1k classes that form "guitar", "dog" and "human".

Each group is the set of ImageNet-1k classes below one WordNet synset:

- ``guitar``: the two descendants of ``guitar.n.01``.
- ``dog``: the 118 breeds below ``dog.n.01`` (ImageNet-1k indices 151 to 268).
- ``human``: the only three descendants of ``person.n.01`` in ImageNet-1k.
"""

from __future__ import annotations

CLASS_GROUPS: dict[str, tuple[int, ...]] = {
    "guitar": (402, 546),  # acoustic guitar, electric guitar
    "dog": tuple(range(151, 269)),  # Chihuahua ... Mexican hairless
    "human": (981, 982, 983),  # ballplayer, groom, scuba diver
}
"""ImageNet-1k class indices of each group; the order of the keys is the order of the model's outputs."""

CLASS_NAMES: tuple[str, ...] = tuple(CLASS_GROUPS)
"""Names of the model's output classes, in output order."""

IMAGENET_TO_GROUP: dict[int, int] = {
    imagenet_index: group_index
    for group_index, members in enumerate(CLASS_GROUPS.values())
    for imagenet_index in members
}
"""Maps an ImageNet-1k class index to the index of its group."""
