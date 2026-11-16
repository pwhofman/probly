"""The uncertainty-aware model: a sub-ensemble of small MLP heads on a frozen ImageNet ResNet-50.

The model is the sub-ensemble of the probly benchmark (``probly_benchmark``), built as its
``_subensemble_builder`` builds it, with the ``mlp_small`` head and one output per class of the
problem. Only the heads are trained, on backbone features that are computed once.
"""

from __future__ import annotations

from importlib import resources
import io
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import v2 as T
from tqdm import tqdm
import yaml

from probly.method.subensemble import subensemble
from probly.traverse_nn.utils import get_output_dim
from probly_benchmark import models

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

IMAGE_SIZE = 224
# The evaluation transform of the benchmark's ImageNet recipe (probly_benchmark.data.TRANSFORMS_TEST).
IMAGENET_TRANSFORM = T.Compose(
    [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def benchmark_subensemble_params(head: str = "mlp_small") -> dict[str, Any]:
    """Return the parameters of the benchmark's sub-ensemble method, with the given head.

    Args:
        head: Name of the head, resolved by :func:`probly_benchmark.models.get_encoder_head`.

    Returns:
        The ``params`` of ``probly_benchmark/configs/method/subensemble.yaml`` with ``head`` replaced.
    """
    config = resources.files("probly_benchmark") / "configs" / "method" / "subensemble.yaml"
    params = dict(yaml.safe_load(config.read_text())["params"])
    params["head"] = head
    return params


def build_subensemble(
    n_classes: int = 3, head: str = "mlp_small", pretrained: bool = True, seed: int = 0
) -> nn.ModuleList:
    """Build the benchmark's sub-ensemble on the ImageNet ResNet-50 encoder.

    Args:
        n_classes: Number of output classes.
        head: Name of the head of each member.
        pretrained: Whether the encoder loads its ImageNet weights. Only tests turn this off.
        seed: Seed of the initialization of the heads.

    Returns:
        The sub-ensemble: a list of members, each the shared frozen encoder followed by its own head.
    """
    torch.manual_seed(seed)
    params = benchmark_subensemble_params(head)
    encoder = models.get_base_model("resnet50_encoder", n_classes, pretrained=pretrained)
    head_module = models.get_encoder_head(params.pop("head"), get_output_dim(encoder), n_classes)
    model = subensemble(encoder, head=head_module, predictor_type="logit_classifier", **params)
    return cast("nn.ModuleList", model).eval()


def encoder_of(model: nn.ModuleList) -> nn.Module:
    """Return the frozen encoder that all members of the sub-ensemble share."""
    return cast("nn.Sequential", model[0])[0]


def heads_of(model: nn.ModuleList) -> list[nn.Module]:
    """Return the head of each member of the sub-ensemble."""
    return [cast("nn.Sequential", member)[1] for member in model]


@torch.no_grad()
def encode(encoder: nn.Module, images: torch.Tensor, batch_size: int = 64, progress: bool = False) -> torch.Tensor:
    """Compute the encoder's features of preprocessed images.

    Args:
        encoder: The frozen encoder.
        images: Preprocessed images of shape ``(m, 3, height, width)``.
        batch_size: Number of images per forward pass.
        progress: Whether to show a progress bar.

    Returns:
        The features of shape ``(m, feature_dim)``.
    """
    encoder.eval()
    batches = range(0, len(images), batch_size)
    features = [
        encoder(images[start : start + batch_size].contiguous(memory_format=torch.channels_last))
        for start in tqdm(batches, desc="encoding", disable=not progress)
    ]
    return torch.cat(features)


def encode_image_bytes(
    encoder: nn.Module, images: Sequence[bytes], batch_size: int = 64, progress: bool = True
) -> torch.Tensor:
    """Decode, preprocess and encode images that are given as encoded bytes (e.g. JPEG).

    Args:
        encoder: The frozen encoder.
        images: The encoded images.
        batch_size: Number of images per forward pass.
        progress: Whether to show a progress bar.

    Returns:
        The features of shape ``(m, feature_dim)``.
    """
    features = []
    for start in tqdm(range(0, len(images), batch_size), desc="encoding", disable=not progress):
        batch = [
            IMAGENET_TRANSFORM(Image.open(io.BytesIO(image)).convert("RGB"))
            for image in images[start : start + batch_size]
        ]
        features.append(encode(encoder, torch.stack(batch), batch_size=batch_size))
    return torch.cat(features)


@torch.no_grad()
def head_logits(model: nn.ModuleList, features: torch.Tensor) -> torch.Tensor:
    """Compute the logits of every head from encoder features.

    Args:
        model: The sub-ensemble.
        features: Encoder features of shape ``(m, feature_dim)``.

    Returns:
        The logits of shape ``(m, num_heads, num_classes)``.
    """
    return torch.stack([head(features) for head in heads_of(model)], dim=1)


def train_heads(
    model: nn.ModuleList,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    seed: int = 0,
) -> None:
    """Train every head on cached encoder features with cross-entropy.

    The heads differ in their random initialization (``reset_params`` of the sub-ensemble) and in
    the order of their mini-batches. The classes are weighted inversely to their frequency.

    Args:
        model: The sub-ensemble, trained in place.
        features: Encoder features of the training images, of shape ``(m, feature_dim)``.
        labels: Class indices of shape ``(m,)``.
        epochs: Number of passes over the features per head.
        batch_size: Mini-batch size.
        lr: Peak learning rate of AdamW, annealed to zero with a cosine schedule.
        weight_decay: Weight decay of AdamW.
        seed: Seed of the mini-batch order; head ``i`` uses ``seed + i``.
    """
    counts = torch.bincount(labels)
    class_weights = len(labels) / (len(counts) * counts)
    steps = epochs * int(np.ceil(len(features) / batch_size))
    for index, head in enumerate(heads_of(model)):
        generator = torch.Generator().manual_seed(seed + index)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        head.train()
        for _ in range(epochs):
            for batch in torch.randperm(len(features), generator=generator).split(batch_size):
                loss = nn.functional.cross_entropy(head(features[batch]), labels[batch], weight=class_weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        head.eval()


def save_heads(model: nn.ModuleList, path: Path) -> None:
    """Save the heads of the sub-ensemble in half precision; the encoder is the public ImageNet checkpoint.

    Reload the model with :func:`load_subensemble` before using it, so that results come from the
    stored weights.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    states = [{name: value.half() for name, value in head.state_dict().items()} for head in heads_of(model)]
    torch.save(states, path)


def load_subensemble(path: Path, n_classes: int = 3, head: str = "mlp_small", pretrained: bool = True) -> nn.ModuleList:
    """Build the sub-ensemble and load its trained heads.

    Args:
        path: File written by :func:`save_heads`.
        n_classes: Number of output classes.
        head: Name of the head of each member.
        pretrained: Whether the encoder loads its ImageNet weights. Only tests turn this off.

    Returns:
        The trained sub-ensemble in evaluation mode.
    """
    model = build_subensemble(n_classes, head, pretrained=pretrained)
    states = torch.load(path, map_location="cpu")
    for head_module, state in zip(heads_of(model), states, strict=True):
        head_module.load_state_dict({name: value.float() for name, value in state.items()})
    return model.eval()
