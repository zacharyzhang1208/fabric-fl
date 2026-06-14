"""Model definitions for prototype-distillation experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        channels, _, _ = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.prototype_dim = embed_dim

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images)
        embeddings = self.encoder(features)
        log_probs = F.log_softmax(self.classifier(embeddings), dim=1)
        return log_probs, embeddings


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
    fedproto_reference: bool,
) -> nn.Module:
    if fedproto_reference and dataset_name == "mnist":
        return FedProtoCNNMnist(
            num_channels=input_shape[0],
            out_channels=20,
            num_classes=num_classes,
        )
    return ImageClassifier(input_shape=input_shape, num_classes=num_classes)
