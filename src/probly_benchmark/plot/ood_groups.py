"""Near-OOD vs far-OOD groupings per ID dataset.

The exact dataset name strings here must match the ones logged into wandb
summary keys by ``ood_detection.py`` (i.e. the ``cfg.ood_dataset`` value).
"""

from __future__ import annotations

NEAR_OOD: dict[str, tuple[str, ...]] = {
    "cifar10": ("cifar100",),
    "cifar100": ("cifar10",),
    "imagenet": ("ssb_hard", "ninco"),
    "imagenet200": ("ssb_hard", "ninco"),
    "imagenet1k": ("ssb_hard", "ninco"),
}

FAR_OOD: dict[str, tuple[str, ...]] = {
    "cifar10": ("mnist", "svhn", "textures", "places365"),
    "cifar100": ("mnist", "svhn", "textures", "places365"),
    "imagenet": ("inaturalist", "textures"),
    "imagenet200": ("inaturalist", "textures"),
    "imagenet1k": ("inaturalist", "textures"),
}


def near_ood_for(id_dataset: str) -> tuple[str, ...]:
    """Return the near-OOD dataset names configured for ``id_dataset``."""
    if id_dataset not in NEAR_OOD:
        msg = f"No near-OOD group defined for ID dataset {id_dataset!r}. Add it to ood_groups.NEAR_OOD."
        raise KeyError(msg)
    return NEAR_OOD[id_dataset]


def far_ood_for(id_dataset: str) -> tuple[str, ...]:
    """Return the far-OOD dataset names configured for ``id_dataset``."""
    if id_dataset not in FAR_OOD:
        msg = f"No far-OOD group defined for ID dataset {id_dataset!r}. Add it to ood_groups.FAR_OOD."
        raise KeyError(msg)
    return FAR_OOD[id_dataset]


def all_ood_for(id_dataset: str) -> tuple[str, ...]:
    """Return the union of near-OOD and far-OOD dataset names."""
    return tuple(list(near_ood_for(id_dataset)) + list(far_ood_for(id_dataset)))
