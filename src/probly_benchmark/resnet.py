"""ResNet in PyTorch. Source: https://github.com/kuangliu/pytorch-cifar.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic two-conv residual block used in ResNet-18/34."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """Initialize the block.

        Args:
            in_planes: Number of input channels.
            planes: Number of output channels (before expansion).
            stride: Stride of the first convolution.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Run the residual block on ``x``."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Three-conv bottleneck residual block used in ResNet-50/101/152."""

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """Initialize the block.

        Args:
            in_planes: Number of input channels.
            planes: Number of bottleneck channels (output is ``planes * expansion``).
            stride: Stride of the 3x3 convolution.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Run the bottleneck block on ``x``."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet backbone parameterized by block type and per-stage depth."""

    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        num_blocks: list[int],
        num_classes: int = 10,
    ) -> None:
        """Initialize the network.

        Args:
            block: Residual block class to use.
            num_blocks: Number of blocks in each of the four stages.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build one stage as a sequence of ``num_blocks`` residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:  # noqa: PLR1704
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Run the full network on ``x`` and return class logits."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18() -> ResNet:  # noqa: N802
    """Return a ResNet-18 for CIFAR-sized inputs."""
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34() -> ResNet:  # noqa: N802
    """Return a ResNet-34 for CIFAR-sized inputs."""
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50() -> ResNet:  # noqa: N802
    """Return a ResNet-50 for CIFAR-sized inputs."""
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101() -> ResNet:  # noqa: N802
    """Return a ResNet-101 for CIFAR-sized inputs."""
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152() -> ResNet:  # noqa: N802
    """Return a ResNet-152 for CIFAR-sized inputs."""
    return ResNet(Bottleneck, [3, 8, 36, 3])
