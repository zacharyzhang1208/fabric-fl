"""Client-side logic for local image federated-learning simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import build_model


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
    proto_loss: float = 0.0
    subspace_loss: float = 0.0


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
        dataset_name: str,
        optimizer_name: str,
        fedproto_reference: bool,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.prototype_loader = prototype_loader
        self.device = device
        self.num_classes = num_classes
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.model = build_model(
            dataset_name=dataset_name,
            input_shape=input_shape,
            num_classes=num_classes,
            fedproto_reference=fedproto_reference,
        ).to(device)
        self.optimizer = self._build_optimizer()
        self.last_prototypes: torch.Tensor | None = None
        self.last_counts: torch.Tensor | None = None

    def train_round(
        self,
        local_epochs: int,
        global_prototypes: torch.Tensor | None,
        global_counts: torch.Tensor | None,
        proto_weight: float,
        global_bases: torch.Tensor | None = None,
        subspace_weight: float = 0.0,
    ) -> TrainMetrics:
        metrics = TrainMetrics(loss=0.0, ce_loss=0.0)
        for _ in range(local_epochs):
            metrics = self._train_epoch(
                global_prototypes,
                global_counts,
                proto_weight,
                global_bases,
                subspace_weight,
            )
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        seen = 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            log_probs, _ = self.model(images)
            correct += (log_probs.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
        return correct / seen

    def build_update(
        self,
        round_id: int,
        compressor: PrototypeCompressorProtocol | None = None,
    ) -> ClientUpdate:
        if self.last_prototypes is None or self.last_counts is None:
            prototypes, counts = self._compute_local_prototypes()
        else:
            prototypes = self.last_prototypes
            counts = self.last_counts
        raw_bytes = prototypes.numel() * prototypes.element_size() + counts.numel() * counts.element_size()
        if compressor is not None:
            compressed = compressor.compress(prototypes, counts)
            prototypes, counts, raw_bytes = compressor.decompress(compressed, self.device)
            compressed_bytes = len(compressed)
        else:
            prototypes = prototypes.detach().clone()
            counts = counts.detach().clone()
            compressed_bytes = raw_bytes

        return ClientUpdate(
            round_id=round_id,
            client_id=self.client_id,
            prototypes=prototypes,
            counts=counts,
            raw_bytes=raw_bytes,
            compressed_bytes=compressed_bytes,
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
        self.optimizer = self._build_optimizer()

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
        global_counts: torch.Tensor | None,
        proto_weight: float,
        global_bases: torch.Tensor | None,
        subspace_weight: float,
    ) -> TrainMetrics:
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_proto = 0.0
        total_subspace = 0.0
        seen = 0
        proto_dim = int(getattr(self.model, "prototype_dim"))
        proto_sums = torch.zeros(self.num_classes, proto_dim, device=self.device)
        proto_counts = torch.zeros(self.num_classes, device=self.device)

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            log_probs, embeddings = self.model(images)
            ce_loss = F.nll_loss(log_probs, labels)

            proto_loss = torch.tensor(0.0, device=self.device)
            if global_prototypes is not None and proto_weight > 0.0:
                if global_counts is None:
                    mask = torch.ones_like(labels, dtype=torch.bool)
                else:
                    mask = global_counts[labels] > 0
                if mask.any():
                    proto_loss = F.mse_loss(embeddings[mask], global_prototypes[labels[mask]])

            subspace_loss = torch.tensor(0.0, device=self.device)
            if global_prototypes is not None and global_bases is not None and subspace_weight > 0.0:
                centered = embeddings - global_prototypes[labels]
                bases = global_bases[labels]
                coeffs = (centered.unsqueeze(1) * bases).sum(dim=2)
                projection = (coeffs.unsqueeze(2) * bases).sum(dim=1)
                residual = centered - projection
                subspace_loss = residual.pow(2).mean()

            loss = ce_loss + proto_weight * proto_loss + subspace_weight * subspace_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            for label in range(self.num_classes):
                mask = labels == label
                if mask.any():
                    proto_sums[label] += embeddings.detach()[mask].sum(dim=0)
                    proto_counts[label] += mask.sum()
            total_loss += loss.item() * batch_size
            total_ce += ce_loss.item() * batch_size
            total_proto += proto_loss.item() * batch_size
            total_subspace += subspace_loss.item() * batch_size
            seen += batch_size

        self.last_prototypes = torch.zeros_like(proto_sums)
        present = proto_counts > 0
        self.last_prototypes[present] = proto_sums[present] / proto_counts[present].unsqueeze(1)
        self.last_counts = proto_counts

        return TrainMetrics(
            loss=total_loss / seen,
            ce_loss=total_ce / seen,
            proto_loss=total_proto / seen,
            subspace_loss=total_subspace / seen,
        )

    @torch.no_grad()
    def _compute_local_prototypes(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        embed_dim = int(getattr(self.model, "prototype_dim"))
        sums = torch.zeros(self.num_classes, embed_dim, device=self.device)
        counts = torch.zeros(self.num_classes, device=self.device)

        for images, labels in self.prototype_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, embeddings = self.model(images)
            for label in range(self.num_classes):
                mask = labels == label
                if mask.any():
                    sums[label] += embeddings[mask].sum(dim=0)
                    counts[label] += mask.sum()

        prototypes = torch.zeros_like(sums)
        present = counts > 0
        prototypes[present] = sums[present] / counts[present].unsqueeze(1)
        return prototypes, counts

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
