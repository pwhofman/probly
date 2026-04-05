"""Script to train models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn, optim
from tqdm import tqdm
import wandb

from probly_benchmark import data, utils
from probly_benchmark.train_funcs import train_epoch

LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
}


def get_loss(name: str, **kwargs: Any) -> nn.Module:  # noqa: ANN401
    """Get loss function."""
    name = name.lower()
    if name not in LOSSES:
        msg = f"Unknown loss: {name}"
        raise ValueError(msg)
    return LOSSES[name](**kwargs)


OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
}


def get_optimizer(name: str, params: Iterable[nn.Parameter], **kwargs: Any) -> optim.Optimizer:  # noqa: ANN401
    """Get optimizer function."""
    name = name.lower()
    if name not in OPTIMIZERS:
        msg = f"Unknown optimizer: {name}"
        raise ValueError(msg)
    return OPTIMIZERS[name](params, **kwargs)


@hydra.main(version_base=None, config_path="configs/", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run the training script."""
    print("=== Training configuration ===")
    print(OmegaConf.to_yaml(cfg))

    run = wandb.init(
        project="paul-hofman/test",
        config=cfg.__dict__,
        mode="online" if cfg.wandb else "disabled",
        save_code=True,
    )

    device = torch.device("mps")

    utils.set_seed(cfg.seed) if cfg.seed else None

    # get data placeholder
    train_loader, _ = data.load_mnist(cfg.dataset)

    # get model placeholder
    model = nn.Sequential(
        nn.Identity(),
    ).to(device)
    criterion = get_loss(cfg.loss)
    optimizer = get_optimizer(cfg.optimizer, model.parameters())

    for _ in tqdm(range(cfg.epochs)):
        model.train()
        running_loss = 0.0
        for inputs_, targets_ in train_loader:
            inputs, targets = inputs_.to(device), targets_.to(device)
            running_loss += train_epoch(model, inputs, targets, criterion, optimizer)
        running_loss /= len(train_loader)

        # add validation function
        model.eval()

        run.log(data={"running_loss": running_loss})


if __name__ == "__main__":
    main()
