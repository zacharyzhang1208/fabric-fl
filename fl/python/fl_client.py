"""Client-side logic for local image federated-learning simulations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import build_model
from prototype_clustering import as_multi_prototypes, spherical_kmeans
from prototype_synthesis import SynthesisResult, synthesize_prototype_images


def cosine_similarity_logits(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("prototype temperature must be positive")
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    normalized_prototypes = F.normalize(prototypes, p=2, dim=1)
    return normalized_embeddings @ normalized_prototypes.transpose(0, 1) / temperature


def prototype_classification_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    global_prototypes: torch.Tensor,
    global_counts: torch.Tensor | None,
    allowed_classes: set[int] | None,
    temperature: float,
) -> torch.Tensor:
    multi_prototypes, multi_counts = as_multi_prototypes(
        global_prototypes,
        global_counts
        if global_counts is not None
        else torch.ones(global_prototypes.shape[:-1], device=global_prototypes.device),
    )
    num_classes = multi_prototypes.shape[0]
    candidates = [
        label
        for label in range(num_classes)
        if (allowed_classes is None or label in allowed_classes)
        and multi_counts[label].sum().item() > 0
    ]
    if not candidates:
        return embeddings.sum() * 0.0

    candidate_labels = torch.tensor(candidates, dtype=torch.long, device=embeddings.device)
    candidate_prototypes = multi_prototypes.to(embeddings.device)[candidate_labels]
    candidate_counts = multi_counts.to(embeddings.device)[candidate_labels]
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    normalized_prototypes = F.normalize(candidate_prototypes, p=2, dim=2)
    center_logits = torch.einsum(
        "bd,ckd->bck",
        normalized_embeddings,
        normalized_prototypes,
    ) / temperature
    center_logits = center_logits.masked_fill(
        candidate_counts.unsqueeze(0) <= 0,
        torch.finfo(center_logits.dtype).min,
    )
    logits = center_logits.max(dim=2).values

    label_positions = torch.full(
        (num_classes,),
        -1,
        dtype=torch.long,
        device=embeddings.device,
    )
    label_positions[candidate_labels] = torch.arange(
        len(candidates),
        device=embeddings.device,
    )
    targets = label_positions[labels]
    valid = targets >= 0
    if not valid.any():
        return embeddings.sum() * 0.0

    return F.cross_entropy(logits[valid], targets[valid])


@dataclass
class ClientUpdate:
    round_id: int
    client_id: int
    prototypes: torch.Tensor
    counts: torch.Tensor
    payload_bytes: int


@dataclass
class ModelUpdate:
    round_id: int
    client_id: int
    state_dict: dict[str, torch.Tensor]
    num_samples: int
    payload_bytes: int


@dataclass
class TrainMetrics:
    loss: float
    ce_loss: float
    proto_loss: float = 0.0
    prox_loss: float = 0.0
    synthetic_loss: float = 0.0
    synthetic_samples: int = 0


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
        model_name: str = "cnn",
    ) -> None:
        self.client_id = client_id
        self.model_name = model_name
        self.train_loader = train_loader
        self.prototype_loader = prototype_loader
        self.device = device
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.dataset_name = dataset_name
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.model = build_model(
            dataset_name=dataset_name,
            input_shape=input_shape,
            num_classes=num_classes,
            model_name=model_name,
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
        proto_temperature: float = 1.0,
        prototype_classes: set[int] | None = None,
        prototypes_per_class: int = 1,
        min_samples_per_prototype: int = 10,
        synthetic_images: torch.Tensor | None = None,
        synthetic_labels: torch.Tensor | None = None,
        synthetic_weight: float = 0.0,
        proximal_state: dict[str, torch.Tensor] | None = None,
        fedprox_mu: float = 0.0,
    ) -> TrainMetrics:
        metrics = TrainMetrics(loss=0.0, ce_loss=0.0)
        device_proximal_state = self._state_to_device(proximal_state) if proximal_state is not None else None
        for _ in range(local_epochs):
            metrics = self._train_epoch(
                global_prototypes,
                global_counts,
                proto_weight,
                proto_temperature,
                prototype_classes,
                prototypes_per_class,
                min_samples_per_prototype,
                synthetic_images,
                synthetic_labels,
                synthetic_weight,
                device_proximal_state,
                fedprox_mu,
            )
        return metrics

    def synthesize_from_prototypes(
        self,
        global_prototypes: torch.Tensor,
        global_counts: torch.Tensor,
        class_counts: list[int],
        target_count: int,
        samples_per_class: int,
        steps: int,
        learning_rate: float,
        temperature: float,
        min_margin: float,
        tv_weight: float,
        seed: int,
    ) -> SynthesisResult:
        if self.dataset_name != "mnist":
            raise ValueError("Prototype-guided image synthesis currently supports MNIST only")
        return synthesize_prototype_images(
            model=self.model,
            global_prototypes=global_prototypes,
            global_counts=global_counts,
            class_counts=class_counts,
            input_shape=self.input_shape,
            normalization_mean=(0.1307,),
            normalization_std=(0.3081,),
            target_count=target_count,
            samples_per_class=samples_per_class,
            steps=steps,
            learning_rate=learning_rate,
            temperature=temperature,
            min_margin=min_margin,
            tv_weight=tv_weight,
            seed=seed,
        )

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        allowed_classes: set[int] | None = None,
    ) -> float:
        self.model.eval()
        correct = 0
        seen = 0
        candidates = None
        if allowed_classes is not None:
            if not allowed_classes:
                raise ValueError("allowed_classes must not be empty")
            candidates = torch.tensor(
                sorted(allowed_classes),
                dtype=torch.long,
                device=self.device,
            )
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            log_probs, _ = self.model(images)
            if candidates is None:
                predictions = log_probs.argmax(dim=1)
            else:
                predictions = candidates[log_probs[:, candidates].argmax(dim=1)]
            correct += (predictions == labels).sum().item()
            seen += labels.size(0)
        return correct / seen

    @torch.no_grad()
    def evaluate_with_prototypes(
        self,
        loader: DataLoader,
        global_prototypes: torch.Tensor,
        global_counts: torch.Tensor,
        allowed_classes: set[int],
    ) -> float:
        self.model.eval()
        multi_prototypes, multi_counts = as_multi_prototypes(
            global_prototypes,
            global_counts,
        )
        candidates = [
            label
            for label in sorted(allowed_classes)
            if multi_counts[label].sum().item() > 0
        ]
        if not candidates:
            raise ValueError("No global prototypes are available for the allowed classes")
        candidate_labels = torch.tensor(candidates, dtype=torch.long, device=self.device)
        candidate_prototypes = multi_prototypes.to(self.device)[candidate_labels]
        candidate_counts = multi_counts.to(self.device)[candidate_labels]
        correct = 0
        seen = 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, embeddings = self.model(images)
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            normalized_prototypes = F.normalize(candidate_prototypes, p=2, dim=2)
            center_similarities = torch.einsum(
                "bd,ckd->bck",
                normalized_embeddings,
                normalized_prototypes,
            )
            center_similarities = center_similarities.masked_fill(
                candidate_counts.unsqueeze(0) <= 0,
                torch.finfo(center_similarities.dtype).min,
            )
            class_similarities = center_similarities.max(dim=2).values
            predictions = candidate_labels[class_similarities.argmax(dim=1)]
            correct += (predictions == labels).sum().item()
            seen += labels.size(0)
        return correct / seen

    @torch.no_grad()
    def evaluate_target_rate(self, loader: DataLoader, source_class: int, target_class: int) -> float:
        self.model.eval()
        target_hits = 0
        source_seen = 0
        for images, labels in loader:
            source_mask = labels == source_class
            if not source_mask.any():
                continue
            images = images[source_mask].to(self.device)
            log_probs, _ = self.model(images)
            target_hits += (log_probs.argmax(dim=1) == target_class).sum().item()
            source_seen += int(source_mask.sum().item())
        return target_hits / source_seen if source_seen else 0.0

    def build_update(self, round_id: int) -> ClientUpdate:
        if self.last_prototypes is None or self.last_counts is None:
            prototypes, counts = self._compute_local_prototypes()
        else:
            prototypes = self.last_prototypes
            counts = self.last_counts
        payload_bytes = prototypes.numel() * prototypes.element_size() + counts.numel() * counts.element_size()
        prototypes = prototypes.detach().clone()
        counts = counts.detach().clone()

        return ClientUpdate(
            round_id=round_id,
            client_id=self.client_id,
            prototypes=prototypes,
            counts=counts,
            payload_bytes=payload_bytes,
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
        payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
        return ModelUpdate(
            round_id=round_id,
            client_id=self.client_id,
            state_dict=state_dict,
            num_samples=len(self.train_loader.dataset),
            payload_bytes=payload_bytes,
        )

    def _train_epoch(
        self,
        global_prototypes: torch.Tensor | None,
        global_counts: torch.Tensor | None,
        proto_weight: float,
        proto_temperature: float,
        prototype_classes: set[int] | None,
        prototypes_per_class: int,
        min_samples_per_prototype: int,
        synthetic_images: torch.Tensor | None,
        synthetic_labels: torch.Tensor | None,
        synthetic_weight: float,
        proximal_state: dict[str, torch.Tensor] | None,
        fedprox_mu: float,
    ) -> TrainMetrics:
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_proto = 0.0
        total_prox = 0.0
        total_synthetic = 0.0
        synthetic_seen = 0
        seen = 0
        class_embeddings: list[list[torch.Tensor]] = [
            [] for _ in range(self.num_classes)
        ]

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            log_probs, embeddings = self.model(images)
            ce_loss = F.nll_loss(log_probs, labels)

            proto_loss = torch.tensor(0.0, device=self.device)
            if global_prototypes is not None and proto_weight > 0.0:
                proto_loss = prototype_classification_loss(
                    embeddings=embeddings,
                    labels=labels,
                    global_prototypes=global_prototypes,
                    global_counts=global_counts,
                    allowed_classes=prototype_classes,
                    temperature=proto_temperature,
                )

            prox_loss = torch.tensor(0.0, device=self.device)
            if proximal_state is not None and fedprox_mu > 0.0:
                prox_loss = self._proximal_loss(proximal_state)

            loss = ce_loss + proto_weight * proto_loss + 0.5 * fedprox_mu * prox_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            for label in range(self.num_classes):
                mask = labels == label
                if mask.any():
                    normalized = F.normalize(embeddings.detach()[mask], p=2, dim=1)
                    class_embeddings[label].append(normalized)
            total_loss += loss.item() * batch_size
            total_ce += ce_loss.item() * batch_size
            total_proto += proto_loss.item() * batch_size
            total_prox += prox_loss.item() * batch_size
            seen += batch_size

        if (
            synthetic_images is not None
            and synthetic_labels is not None
            and synthetic_images.shape[0] > 0
            and synthetic_weight > 0.0
        ):
            synthetic_images = synthetic_images.to(self.device)
            synthetic_labels = synthetic_labels.to(self.device)
            permutation = torch.randperm(synthetic_labels.shape[0], device=self.device)
            synthetic_batch_size = min(
                self.train_loader.batch_size or synthetic_labels.shape[0],
                synthetic_labels.shape[0],
            )
            for begin in range(0, synthetic_labels.shape[0], synthetic_batch_size):
                indices = permutation[begin : begin + synthetic_batch_size]
                images = synthetic_images[indices]
                labels = synthetic_labels[indices]
                log_probs, embeddings = self.model(images)
                synthetic_ce = F.nll_loss(log_probs, labels)
                synthetic_proto = prototype_classification_loss(
                    embeddings=embeddings,
                    labels=labels,
                    global_prototypes=global_prototypes,
                    global_counts=global_counts,
                    allowed_classes=None,
                    temperature=proto_temperature,
                )
                synthetic_loss = synthetic_ce + proto_weight * synthetic_proto
                self.optimizer.zero_grad()
                (synthetic_weight * synthetic_loss).backward()
                self.optimizer.step()
                batch_size = labels.size(0)
                total_synthetic += synthetic_loss.item() * batch_size
                synthetic_seen += batch_size

        self.last_prototypes, self.last_counts = self._cluster_class_embeddings(
            class_embeddings,
            prototypes_per_class,
            min_samples_per_prototype,
        )

        return TrainMetrics(
            loss=total_loss / seen,
            ce_loss=total_ce / seen,
            proto_loss=total_proto / seen,
            prox_loss=total_prox / seen,
            synthetic_loss=total_synthetic / synthetic_seen if synthetic_seen else 0.0,
            synthetic_samples=synthetic_seen,
        )

    def _state_to_device(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: tensor.to(self.device)
            for name, tensor in state_dict.items()
            if tensor.is_floating_point()
        }

    def _proximal_loss(self, proximal_state: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in proximal_state:
                loss = loss + torch.sum((param - proximal_state[name]) ** 2)
        return loss

    @torch.no_grad()
    def _compute_local_prototypes(
        self,
        prototypes_per_class: int = 1,
        min_samples_per_prototype: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        class_embeddings: list[list[torch.Tensor]] = [
            [] for _ in range(self.num_classes)
        ]

        for images, labels in self.prototype_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, embeddings = self.model(images)
            for label in range(self.num_classes):
                mask = labels == label
                if mask.any():
                    normalized = F.normalize(embeddings[mask], p=2, dim=1)
                    class_embeddings[label].append(normalized)
        return self._cluster_class_embeddings(
            class_embeddings,
            prototypes_per_class,
            min_samples_per_prototype,
        )

    def _cluster_class_embeddings(
        self,
        class_embeddings: list[list[torch.Tensor]],
        prototypes_per_class: int,
        min_samples_per_prototype: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prototypes_per_class <= 0:
            raise ValueError("prototypes_per_class must be positive")
        if min_samples_per_prototype <= 0:
            raise ValueError("min_samples_per_prototype must be positive")

        embed_dim = int(getattr(self.model, "prototype_dim"))
        prototypes = torch.zeros(
            self.num_classes,
            prototypes_per_class,
            embed_dim,
            device=self.device,
        )
        counts = torch.zeros(
            self.num_classes,
            prototypes_per_class,
            device=self.device,
        )
        for label, batches in enumerate(class_embeddings):
            if not batches:
                continue
            vectors = torch.cat(batches, dim=0)
            effective_clusters = min(
                prototypes_per_class,
                max(1, vectors.shape[0] // min_samples_per_prototype),
            )
            centers, cluster_counts = spherical_kmeans(
                vectors,
                effective_clusters,
            )
            prototypes[label, :effective_clusters] = centers
            counts[label, :effective_clusters] = cluster_counts

        if prototypes_per_class == 1:
            return prototypes[:, 0], counts[:, 0]
        return prototypes, counts

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
