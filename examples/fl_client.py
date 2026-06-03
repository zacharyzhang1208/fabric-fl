"""Client-side logic for local image federated-learning simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import ImageClassifier


class PrototypeCompressorProtocol(Protocol):
    def compress(self, prototypes: torch.Tensor, counts: torch.Tensor) -> bytes:
        ...

    def decompress(self, payload_bytes: bytes, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
        ...


@dataclass
class ClientUpdate:
    round_id: int
    client_id: int
    prototypes: torch.Tensor
    counts: torch.Tensor
    raw_bytes: int
    compressed_bytes: int


@dataclass
class ModelUpdate:
    round_id: int
    client_id: int
    state_dict: dict[str, torch.Tensor]
    num_samples: int
    raw_bytes: int


@dataclass
class TrainMetrics:
    loss: float
    ce_loss: float


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        prototype_loader: DataLoader,
        device: torch.device,
        lr: float,
        input_shape: tuple[int, int, int],
        num_classes: int,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.prototype_loader = prototype_loader
        self.device = device
        self.num_classes = num_classes
        self.lr = lr
        self.model = ImageClassifier(input_shape=input_shape, num_classes=num_classes).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_round(
        self,
        local_epochs: int,
        global_prototypes: torch.Tensor | None,
        proto_weight: float,
    ) -> TrainMetrics:
        metrics = TrainMetrics(loss=0.0, ce_loss=0.0)
        for _ in range(local_epochs):
            metrics = self._train_epoch(global_prototypes, proto_weight)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        seen = 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, logits = self.model(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
        return correct / seen

    def build_update(
        self,
        round_id: int,
        compressor: PrototypeCompressorProtocol,
    ) -> ClientUpdate:
        prototypes, counts = self._compute_local_prototypes()
        compressed = compressor.compress(prototypes, counts)
        restored_prototypes, restored_counts, raw_bytes = compressor.decompress(compressed, self.device)
        return ClientUpdate(
            round_id=round_id,
            client_id=self.client_id,
            prototypes=restored_prototypes,
            counts=restored_counts,
            raw_bytes=raw_bytes,
            compressed_bytes=len(compressed),
        )

    def get_model_state(self) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def load_model_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        device_state = {
            name: tensor.to(self.device)
            for name, tensor in state_dict.items()
        }
        self.model.load_state_dict(device_state)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def build_model_update(self, round_id: int) -> ModelUpdate:
        state_dict = self.get_model_state()
        raw_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
        return ModelUpdate(
            round_id=round_id,
            client_id=self.client_id,
            state_dict=state_dict,
            num_samples=len(self.train_loader.dataset),
            raw_bytes=raw_bytes,
        )

    def _train_epoch(
        self,
        global_prototypes: torch.Tensor | None,
        proto_weight: float,
    ) -> TrainMetrics:
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        seen = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            embeddings, logits = self.model(images)
            ce_loss = F.cross_entropy(logits, labels)

            proto_loss = torch.tensor(0.0, device=self.device)
            if global_prototypes is not None and proto_weight > 0.0:
                proto_loss = F.mse_loss(embeddings, global_prototypes[labels])

            loss = ce_loss + proto_weight * proto_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_ce += ce_loss.item() * batch_size
            seen += batch_size

        return TrainMetrics(loss=total_loss / seen, ce_loss=total_ce / seen)

    @torch.no_grad()
    def _compute_local_prototypes(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        embed_dim = self.model.classifier.in_features
        sums = torch.zeros(self.num_classes, embed_dim, device=self.device)
        counts = torch.zeros(self.num_classes, device=self.device)

        for images, labels in self.prototype_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            embeddings, _ = self.model(images)
            for label in range(self.num_classes):
                mask = labels == label
                if mask.any():
                    sums[label] += embeddings[mask].sum(dim=0)
                    counts[label] += mask.sum()

        prototypes = torch.zeros_like(sums)
        present = counts > 0
        prototypes[present] = sums[present] / counts[present].unsqueeze(1)
        return prototypes, counts
