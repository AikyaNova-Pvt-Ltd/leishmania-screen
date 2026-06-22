import torch
import torch.nn as nn


class _LeishNet(nn.Module):
    """Neural network architecture — must match training exactly."""

    def __init__(self, d: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.30),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
