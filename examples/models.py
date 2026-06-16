"""FedProto reference model definitions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn


class FedProtoCNNMnist(nn.Module):
    """MNIST CNN copied from the FedProto reference implementation."""

    def __init__(self, num_channels: int = 1, out_channels: int = 20, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320 / 20 * out_channels), 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.prototype_dim = 50

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(F.max_pool2d(self.conv1(images), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        prototypes = F.relu(self.fc1(x))
        x = F.dropout(prototypes, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), prototypes


def build_model(
    dataset_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> nn.Module:
    if dataset_name == "mnist":
        return FedProtoCNNMnist(
            num_channels=input_shape[0],
            out_channels=20,
            num_classes=num_classes,
        )
    raise ValueError(
        f"Dataset {dataset_name!r} does not have a FedProto-aligned model in examples/models.py"
    )
