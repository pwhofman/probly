"""Unified Evidential Training - common (lazy)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lazy_dispatch import lazydispatch

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.utils.data import DataLoader


@lazydispatch
def _uet_dispatch(
    _model: Module,
    *,
    _mode: str,
    _dataloader: DataLoader,
    _loss_fn: object = None,
    _oodloader: DataLoader = None,
    _flow: object = None,
    _class_count: object = None,
    _epochs: int = 5,
    _lr: float = 1e-3,
    _device: str = "cpu",
) -> None:
    msg = "No support for given framework available."
    raise TypeError(msg)


def unified_evidential_train(
    *,
    mode: Literal["PostNet", "NatPostNet", "EDL", "PrNet", "IRD", "DER", "RPN"],
    model: Module,
    dataloader: DataLoader,
    loss_fn: object = None,
    oodloader: DataLoader = None,
    flow: object = None,
    class_count: object = None,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    """Unified entrypoint (backend-agnostic)."""
    _uet_dispatch(
        model,
        mode=mode,
        dataloader=dataloader,
        loss_fn=loss_fn,
        oodloader=oodloader,
        flow=flow,
        class_count=class_count,
        epochs=epochs,
        lr=lr,
        device=device,
    )
