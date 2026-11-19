"""Near-OOD vs far-OOD groupings per ID dataset.

Dataset name strings must match ``cfg.ood_dataset`` values logged into wandb.
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
        msg = f"No near-OOD group defined for ID dataset {id_dataset!r}."
        raise KeyError(msg)
    return NEAR_OOD[id_dataset]


def far_ood_for(id_dataset: str) -> tuple[str, ...]:
    """Return the far-OOD dataset names configured for ``id_dataset``."""
    if id_dataset not in FAR_OOD:
        msg = f"No far-OOD group defined for ID dataset {id_dataset!r}."
        raise KeyError(msg)
    return FAR_OOD[id_dataset]
