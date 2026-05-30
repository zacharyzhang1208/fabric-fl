"""Model definitions for prototype-distillation experiments."""

from __future__ import annotations

import torch
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

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images)
        embeddings = self.encoder(features)
        logits = self.classifier(embeddings)
        return embeddings, logits
