"""Deterministic spherical clustering helpers for class prototypes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def spherical_kmeans(
    vectors: torch.Tensor,
    num_clusters: int,
    weights: torch.Tensor | None = None,
    initial_centers: torch.Tensor | None = None,
    max_iterations: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("vectors must have shape [samples, dimension] with at least one sample")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    vectors = F.normalize(vectors, p=2, dim=1)
    sample_count = vectors.shape[0]
    cluster_count = min(num_clusters, sample_count)
    if weights is None:
        weights = torch.ones(sample_count, dtype=vectors.dtype, device=vectors.device)
    else:
        weights = weights.to(device=vectors.device, dtype=vectors.dtype)
        if weights.ndim != 1 or weights.shape[0] != sample_count:
            raise ValueError("weights must have one value per sample")
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("weights must contain finite positive values")

    centers = _initialize_centers(
        vectors,
        weights,
        cluster_count,
        initial_centers,
    )
    for _ in range(max_iterations):
        assignments = (vectors @ centers.transpose(0, 1)).argmax(dim=1)
        updated = centers.clone()
        for cluster_id in range(cluster_count):
            members = assignments == cluster_id
            if members.any():
                weighted_sum = (vectors[members] * weights[members].unsqueeze(1)).sum(dim=0)
                updated[cluster_id] = F.normalize(weighted_sum.unsqueeze(0), p=2, dim=1)[0]
        if torch.allclose(updated, centers, atol=1e-6, rtol=0):
            centers = updated
            break
        centers = updated

    assignments = (vectors @ centers.transpose(0, 1)).argmax(dim=1)
    cluster_weights = torch.zeros(cluster_count, dtype=weights.dtype, device=weights.device)
    cluster_weights.scatter_add_(0, assignments, weights)
    return centers, cluster_weights


def _initialize_centers(
    vectors: torch.Tensor,
    weights: torch.Tensor,
    cluster_count: int,
    initial_centers: torch.Tensor | None,
) -> torch.Tensor:
    centers: list[torch.Tensor] = []
    if initial_centers is not None and initial_centers.numel() > 0:
        initial_centers = initial_centers.to(device=vectors.device, dtype=vectors.dtype)
        if initial_centers.ndim != 2 or initial_centers.shape[1] != vectors.shape[1]:
            raise ValueError("initial_centers must match the vector dimension")
        normalized = F.normalize(initial_centers, p=2, dim=1)
        centers.extend(normalized[:cluster_count].unbind(dim=0))

    chosen = torch.zeros(vectors.shape[0], dtype=torch.bool, device=vectors.device)
    if not centers:
        weighted_mean = (vectors * weights.unsqueeze(1)).sum(dim=0)
        first = F.normalize(weighted_mean.unsqueeze(0), p=2, dim=1)[0]
        if first.norm().item() == 0:
            first_index = int(weights.argmax().item())
            first = vectors[first_index]
            chosen[first_index] = True
        centers.append(first)

    while len(centers) < cluster_count:
        stacked = torch.stack(centers)
        nearest_similarity = (vectors @ stacked.transpose(0, 1)).max(dim=1).values
        nearest_similarity = nearest_similarity.masked_fill(chosen, float("inf"))
        next_index = int(nearest_similarity.argmin().item())
        centers.append(vectors[next_index])
        chosen[next_index] = True

    return torch.stack(centers)


def as_multi_prototypes(
    prototypes: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prototypes.ndim == 2 and counts.ndim == 1:
        return prototypes.unsqueeze(1), counts.unsqueeze(1)
    if prototypes.ndim == 3 and counts.ndim == 2:
        if prototypes.shape[:2] != counts.shape:
            raise ValueError("prototype and count shapes do not match")
        return prototypes, counts
    raise ValueError(
        "prototypes/counts must have shapes [classes, dimension]/[classes] "
        "or [classes, centers, dimension]/[classes, centers]"
    )
