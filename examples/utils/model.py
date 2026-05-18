import torch
from torch import nn
from torch.nn import functional as F
from probly.predictor import LogitClassifier


class MLPClassifier(nn.Module, LogitClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SequentialModel(nn.Sequential):

    def __init__(self) -> None:
        super().__init__(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )


class SmallResidualBlock1D(nn.Module):
    """Residual block using 1D convolutions for 2-feature data."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return F.relu(out)


class MiniResNet1D(nn.Module):
    """ResNet adapted for 2D point data using 1D convolutions."""

    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.block1 = SmallResidualBlock1D(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.block2 = SmallResidualBlock1D(32)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = self.block1(x)

        x = F.relu(self.conv2(x))
        x = self.block2(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)  # [N, 32]

        return self.fc(x)
class ResFFNLayer(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.relu(self.norm(self.linear(x)))


class ResFFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(2, 128)
        self.layers = nn.ModuleList([ResFFNLayer(128) for _ in range(12)])
        self.last = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.first(x))
        for layer in self.layers:
            x = layer(x)
        return self.last(x)
