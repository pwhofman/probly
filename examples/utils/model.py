import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(
        self, in_features: int = 2, hidden_features: int = 64, out_features: int = 2
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
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


class ResFFNLayer(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.relu(self.norm(self.linear(x)))


class ResFFN(nn.Module):
    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 128,
        out_features: int = 2,
    ) -> None:
        super().__init__()

        self.first = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([ResFFNLayer(hidden_features) for _ in range(12)])
        self.last = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.first(x))
        for layer in self.layers:
            x = layer(x)
        return self.last(x)
