"""FedProto reference model definitions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn


MODEL_NAMES = ("mlp", "cnn", "mini_resnet")


class FedProtoMLPMnist(nn.Module):
    """Small fully connected MNIST model with a shared-size embedding."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.embedding = nn.Linear(128, 50)
        self.classifier = nn.Linear(50, num_classes)
        self.prototype_dim = 50

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = images.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        prototypes = F.relu(self.embedding(x))
        logits = self.classifier(prototypes)
        return F.log_softmax(logits, dim=1), prototypes


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(4, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(4, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class FedProtoMiniResNetMnist(nn.Module):
    """Compact residual MNIST model with a 50-dimensional embedding."""

    def __init__(self, num_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 32),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(32, 50)
        self.classifier = nn.Linear(50, num_classes)
        self.prototype_dim = 50

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.blocks(self.stem(images))
        x = self.pool(x).flatten(start_dim=1)
        prototypes = F.relu(self.embedding(x))
        logits = self.classifier(prototypes)
        return F.log_softmax(logits, dim=1), prototypes


def model_name_for_client(client_id: int, num_clients: int, model_config: str) -> str:
    if model_config == "homogeneous":
        return "cnn"
    if model_config != "heterogeneous":
        raise ValueError(f"Unsupported model config: {model_config}")
    if num_clients < len(MODEL_NAMES):
        raise ValueError(
            "Heterogeneous model config requires at least "
            f"{len(MODEL_NAMES)} clients"
        )
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"Client id {client_id} is outside [0, {num_clients - 1}]")

    base_size, remainder = divmod(num_clients, len(MODEL_NAMES))
    boundary = 0
    for index, model_name in enumerate(MODEL_NAMES):
        boundary += base_size + (1 if index < remainder else 0)
        if client_id < boundary:
            return model_name
    raise AssertionError("Client model assignment did not cover every client")


def build_model(
    dataset_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    model_name: str = "cnn",
) -> nn.Module:
    if dataset_name != "mnist":
        raise ValueError(
            f"Dataset {dataset_name!r} does not have a FedProto-aligned model "
            "in fl/python/models.py"
        )
    if model_name == "mlp":
        if input_shape != (1, 28, 28):
            raise ValueError(f"MNIST MLP requires input shape (1, 28, 28), got {input_shape}")
        return FedProtoMLPMnist(num_classes=num_classes)
    if model_name == "cnn":
        return FedProtoCNNMnist(
            num_channels=input_shape[0],
            out_channels=20,
            num_classes=num_classes,
        )
    if model_name == "mini_resnet":
        return FedProtoMiniResNetMnist(
            num_channels=input_shape[0],
            num_classes=num_classes,
        )
    supported = ", ".join(MODEL_NAMES)
    raise ValueError(f"Unsupported model {model_name!r}. Choose one of: {supported}")
