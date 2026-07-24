"""Prototype-guided input synthesis for data-free client augmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from prototype_clustering import as_multi_prototypes


@dataclass
class SynthesisResult:
    images: torch.Tensor
    labels: torch.Tensor
    attempted: int
    accepted: int
    classes: list[int]
    average_margin: float


def total_variation(images: torch.Tensor) -> torch.Tensor:
    """Return mean anisotropic total variation for a batch of images."""
    vertical = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs().mean()
    horizontal = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean()
    return vertical + horizontal


def prototype_class_logits(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    counts: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return active-class logits and their original class labels."""
    multi_prototypes, multi_counts = as_multi_prototypes(prototypes, counts)
    active = multi_counts.sum(dim=1) > 0
    active_labels = active.nonzero(as_tuple=False).flatten()
    if active_labels.numel() == 0:
        raise ValueError("No active global prototypes are available")

    candidate_prototypes = multi_prototypes[active_labels]
    candidate_counts = multi_counts[active_labels]
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
    return center_logits.max(dim=2).values, active_labels


def synthesize_prototype_images(
    model: torch.nn.Module,
    global_prototypes: torch.Tensor,
    global_counts: torch.Tensor,
    class_counts: list[int],
    input_shape: tuple[int, int, int],
    normalization_mean: tuple[float, ...],
    normalization_std: tuple[float, ...],
    target_count: int,
    samples_per_class: int,
    steps: int,
    learning_rate: float,
    temperature: float,
    min_margin: float,
    tv_weight: float,
    seed: int,
) -> SynthesisResult:
    """Invert rare-class prototypes into filtered normalized input tensors."""
    if len(class_counts) != global_prototypes.shape[0]:
        raise ValueError("class_counts must contain one value per class")
    if len(normalization_mean) != input_shape[0] or len(normalization_std) != input_shape[0]:
        raise ValueError("normalization statistics must match the input channels")

    device = global_prototypes.device
    multi_prototypes, multi_counts = as_multi_prototypes(
        global_prototypes,
        global_counts,
    )
    target_classes = [
        label
        for label, count in enumerate(class_counts)
        if count < target_count and multi_counts[label].sum().item() > 0
    ]
    attempted = len(target_classes) * samples_per_class
    empty_images = torch.empty((0, *input_shape), device=device)
    empty_labels = torch.empty(0, dtype=torch.long, device=device)
    if attempted == 0:
        return SynthesisResult(
            images=empty_images,
            labels=empty_labels,
            attempted=0,
            accepted=0,
            classes=[],
            average_margin=0.0,
        )

    target_labels: list[int] = []
    target_centers: list[int] = []
    for label in target_classes:
        active_centers = (
            (multi_counts[label] > 0)
            .nonzero(as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        for sample_index in range(samples_per_class):
            target_labels.append(label)
            target_centers.append(active_centers[sample_index % len(active_centers)])
    targets = torch.tensor(target_labels, dtype=torch.long, device=device)
    center_indices = torch.tensor(target_centers, dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    raw_parameters = torch.randn(
        (attempted, *input_shape),
        generator=generator,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([raw_parameters], lr=learning_rate)
    mean = torch.tensor(normalization_mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(normalization_std, device=device).view(1, -1, 1, 1)

    parameter_grad_states = [parameter.requires_grad for parameter in model.parameters()]
    was_training = model.training
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    try:
        for _ in range(steps):
            unit_images = raw_parameters.sigmoid()
            normalized_images = (unit_images - mean) / std
            _, embeddings = model(normalized_images)
            logits, active_labels = prototype_class_logits(
                embeddings,
                multi_prototypes,
                multi_counts,
                temperature,
            )
            label_positions = torch.full(
                (multi_prototypes.shape[0],),
                -1,
                dtype=torch.long,
                device=device,
            )
            label_positions[active_labels] = torch.arange(
                active_labels.numel(),
                device=device,
            )
            target_positions = label_positions[targets]
            classification_loss = F.cross_entropy(logits, target_positions)
            assigned_prototypes = multi_prototypes[targets, center_indices]
            target_similarities = F.cosine_similarity(
                embeddings,
                assigned_prototypes,
                dim=1,
            )
            inversion_loss = (1.0 - target_similarities).mean()
            loss = (
                classification_loss
                + inversion_loss
                + tv_weight * total_variation(unit_images)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            unit_images = raw_parameters.sigmoid()
            normalized_images = (unit_images - mean) / std
            _, embeddings = model(normalized_images)
            logits, active_labels = prototype_class_logits(
                embeddings,
                multi_prototypes,
                multi_counts,
                temperature=1.0,
            )
            top_values, top_positions = logits.topk(
                k=min(2, logits.shape[1]),
                dim=1,
            )
            predictions = active_labels[top_positions[:, 0]]
            if logits.shape[1] > 1:
                margins = top_values[:, 0] - top_values[:, 1]
            else:
                margins = torch.full_like(top_values[:, 0], float("inf"))
            accepted_mask = (predictions == targets) & (margins >= min_margin)
            accepted_images = normalized_images[accepted_mask].detach()
            accepted_labels = targets[accepted_mask].detach()
            accepted_margins = margins[accepted_mask]
    finally:
        for parameter, requires_grad in zip(model.parameters(), parameter_grad_states):
            parameter.requires_grad_(requires_grad)
        model.train(was_training)

    accepted_classes = sorted(set(accepted_labels.detach().cpu().tolist()))
    finite_margins = accepted_margins[torch.isfinite(accepted_margins)]
    average_margin = (
        finite_margins.mean().item()
        if finite_margins.numel()
        else 0.0
    )
    return SynthesisResult(
        images=accepted_images,
        labels=accepted_labels,
        attempted=attempted,
        accepted=int(accepted_mask.sum().item()),
        classes=accepted_classes,
        average_margin=average_margin,
    )
