"""Shared helpers for FL algorithm runners."""

from __future__ import annotations

import torch

from fl_client import ClientUpdate, FederatedClient, ModelUpdate
from logging_utils import format_bytes
from prototype_clustering import as_multi_prototypes, spherical_kmeans
from .attacks import poison_prototype_update, poison_model_update

EvalLoaders = dict[str, list]


def average_accuracy(clients: list[FederatedClient], loaders, client_ids: list[int] | None = None) -> float:
    if client_ids is None:
        client_ids = list(range(len(clients)))
    if not client_ids:
        raise ValueError("No clients available for accuracy evaluation")
    return sum(clients[client_id].evaluate(loaders[client_id]) for client_id in client_ids) / len(client_ids)


def format_client_accuracies(client: FederatedClient, eval_loaders: EvalLoaders) -> str:
    parts = []
    for scope, loaders in eval_loaders.items():
        acc = client.evaluate(loaders[client.client_id])
        parts.append(f"{scope}_test_acc={acc * 100:5.2f}%")
    return " ".join(parts)


def print_aggregator_accuracies(
    clients: list[FederatedClient],
    eval_loaders: EvalLoaders,
    evaluation_clients: list[int],
    malicious_clients: set[int],
    round_comm_bytes: int,
) -> None:
    acc = average_accuracy(clients, eval_loaders["local"], evaluation_clients)
    metric_name = "benign_avg_acc" if malicious_clients else "avg_acc"
    print(
        f"  aggregator: {metric_name}={acc * 100:5.2f}% "
        f"round_payload={round_comm_bytes}B"
    )


def print_shared_model_aggregator_accuracies(
    clients: list[FederatedClient],
    eval_loaders: EvalLoaders,
    evaluation_clients: list[int],
    malicious_clients: set[int],
    round_comm_bytes: int,
) -> None:
    acc = average_accuracy(clients, eval_loaders["local"], evaluation_clients)
    metric_name = "benign_avg_acc" if malicious_clients else "avg_acc"
    print(
        f"  aggregator: {metric_name}={acc * 100:5.2f}% "
        f"round_payload={round_comm_bytes}B"
    )


def print_model_group_accuracies(
    clients: list[FederatedClient],
    eval_loaders: EvalLoaders,
    evaluation_clients: list[int],
) -> None:
    groups: dict[str, list[int]] = {}
    for client_id in evaluation_clients:
        model_name = clients[client_id].model_name
        groups.setdefault(model_name, []).append(client_id)

    parts = []
    group_accuracies = []
    loaders = eval_loaders["local"]
    for model_name, client_ids in groups.items():
        accuracy = average_accuracy(clients, loaders, client_ids)
        group_accuracies.append(accuracy)
        parts.append(f"{model_name}_avg_acc={accuracy * 100:5.2f}%")

    if group_accuracies:
        parts.append(f"worst_group_acc={min(group_accuracies) * 100:5.2f}%")
    print(f"  model_groups: {' '.join(parts)}")


def print_local_class_accuracies(
    clients: list[FederatedClient],
    local_loaders,
    evaluation_clients: list[int],
    client_label_sets: list[set[int]],
    global_prototypes: torch.Tensor,
    global_counts: torch.Tensor,
    malicious_clients: set[int],
) -> None:
    head_accuracy = sum(
        clients[client_id].evaluate(
            local_loaders[client_id],
            allowed_classes=client_label_sets[client_id],
        )
        for client_id in evaluation_clients
    ) / len(evaluation_clients)
    prototype_accuracy = sum(
        clients[client_id].evaluate_with_prototypes(
            local_loaders[client_id],
            global_prototypes,
            global_counts,
            client_label_sets[client_id],
        )
        for client_id in evaluation_clients
    ) / len(evaluation_clients)
    prefix = "benign_" if malicious_clients else ""
    print(
        "  local_class_eval: "
        f"{prefix}head_local_classes_avg_acc={head_accuracy * 100:5.2f}% "
        f"{prefix}prototype_local_classes_avg_acc={prototype_accuracy * 100:5.2f}%"
    )


def aggregate_prototypes(
    payloads: list[ClientUpdate],
    device: torch.device,
    num_classes: int,
    previous_prototypes: torch.Tensor | None = None,
    previous_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not payloads:
        raise ValueError("No client payloads to aggregate")

    if payloads[0].prototypes.ndim == 3:
        return aggregate_multiple_prototypes(
            payloads,
            device,
            num_classes,
            previous_prototypes,
            previous_counts,
        )

    embed_dim = payloads[0].prototypes.shape[1]
    sums = torch.zeros(num_classes, embed_dim, device=device)
    counts = torch.zeros(num_classes, device=device)

    for payload in payloads:
        client_counts = payload.counts.to(device)
        present = client_counts > 0
        weights = client_counts[present].unsqueeze(1)
        sums[present] += payload.prototypes.to(device)[present] * weights
        counts[present] += client_counts[present]

    global_prototypes = torch.zeros_like(sums)
    present = counts > 0
    global_prototypes[present] = sums[present] / counts[present].unsqueeze(1)
    return global_prototypes, counts


def aggregate_multiple_prototypes(
    payloads: list[ClientUpdate],
    device: torch.device,
    num_classes: int,
    previous_prototypes: torch.Tensor | None = None,
    previous_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_prototypes, first_counts = as_multi_prototypes(
        payloads[0].prototypes,
        payloads[0].counts,
    )
    if first_prototypes.shape[0] != num_classes:
        raise ValueError("client prototype class count does not match num_classes")
    prototypes_per_class = first_prototypes.shape[1]
    embed_dim = first_prototypes.shape[2]
    global_prototypes = torch.zeros(
        num_classes,
        prototypes_per_class,
        embed_dim,
        device=device,
    )
    global_counts = torch.zeros(
        num_classes,
        prototypes_per_class,
        device=device,
    )

    previous_multi = None
    previous_multi_counts = None
    if previous_prototypes is not None and previous_counts is not None:
        previous_multi, previous_multi_counts = as_multi_prototypes(
            previous_prototypes,
            previous_counts,
        )

    for label in range(num_classes):
        centers = []
        weights = []
        for payload in payloads:
            client_prototypes, client_counts = as_multi_prototypes(
                payload.prototypes.to(device),
                payload.counts.to(device),
            )
            if client_prototypes.shape != first_prototypes.shape:
                raise ValueError("all client multi-prototype shapes must match")
            present = client_counts[label] > 0
            if present.any():
                centers.append(client_prototypes[label, present])
                weights.append(client_counts[label, present])
        if not centers:
            continue

        initial = None
        if previous_multi is not None and previous_multi_counts is not None:
            previous_present = previous_multi_counts[label] > 0
            if previous_present.any():
                initial = previous_multi[label, previous_present]
        class_centers, class_counts = spherical_kmeans(
            torch.cat(centers, dim=0),
            prototypes_per_class,
            weights=torch.cat(weights, dim=0),
            initial_centers=initial,
        )
        active = class_centers.shape[0]
        global_prototypes[label, :active] = class_centers
        global_counts[label, :active] = class_counts

    return global_prototypes, global_counts


def aggregate_model_updates(
    updates: list[ModelUpdate],
    aggregation: str = "mean",
    trim_count: int = 0,
    krum_f: int = 1,
) -> dict[str, torch.Tensor]:
    if aggregation == "mean":
        return aggregate_model_updates_mean(updates)
    if aggregation == "trimmed_mean":
        return aggregate_model_updates_trimmed_mean(updates, trim_count)
    if aggregation == "multi_krum":
        return aggregate_model_updates_multi_krum(updates, krum_f)
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def aggregate_model_updates_mean(updates: list[ModelUpdate]) -> dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("No client model updates to aggregate")

    total_samples = sum(update.num_samples for update in updates)
    if total_samples <= 0:
        raise ValueError("No client samples to aggregate")

    averaged: dict[str, torch.Tensor] = {}
    first_state = updates[0].state_dict
    for name, first_tensor in first_state.items():
        if not first_tensor.is_floating_point():
            averaged[name] = first_tensor.clone()
            continue

        tensor_sum = torch.zeros_like(first_tensor, dtype=torch.float32)
        for update in updates:
            weight = update.num_samples / total_samples
            tensor_sum += update.state_dict[name].float() * weight
        averaged[name] = tensor_sum.to(dtype=first_tensor.dtype)
    return averaged


def aggregate_model_updates_trimmed_mean(
    updates: list[ModelUpdate],
    trim_count: int,
) -> dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("No client model updates to aggregate")
    if trim_count < 0:
        raise ValueError("trim_count must be non-negative")
    if trim_count * 2 >= len(updates):
        raise ValueError("trim_count trims all model updates")

    averaged: dict[str, torch.Tensor] = {}
    first_state = updates[0].state_dict
    for name, first_tensor in first_state.items():
        if not first_tensor.is_floating_point():
            averaged[name] = first_tensor.clone()
            continue

        stacked = torch.stack(
            [update.state_dict[name].float() for update in updates],
            dim=0,
        )
        sorted_values, _ = torch.sort(stacked, dim=0)
        if trim_count > 0:
            sorted_values = sorted_values[trim_count:-trim_count]
        averaged[name] = sorted_values.mean(dim=0).to(dtype=first_tensor.dtype)
    return averaged


def aggregate_model_updates_multi_krum(
    updates: list[ModelUpdate],
    krum_f: int,
) -> dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("No client model updates to aggregate")
    if krum_f < 0:
        raise ValueError("--krum-f must be non-negative")

    n_updates = len(updates)
    neighbor_count = n_updates - krum_f - 2
    if neighbor_count < 1:
        raise ValueError("Multi-Krum requires at least f + 3 model updates")
    if n_updates <= 2 * krum_f + 2:
        raise ValueError("Multi-Krum requires num_clients > 2 * krum_f + 2")

    vectors = [flatten_floating_state(update.state_dict) for update in updates]
    scores: list[tuple[float, int]] = []
    for index, vector in enumerate(vectors):
        distances = []
        for other_index, other_vector in enumerate(vectors):
            if index == other_index:
                continue
            distance = torch.sum((vector - other_vector) ** 2).item()
            distances.append(distance)
        distances.sort()
        scores.append((sum(distances[:neighbor_count]), index))

    scores.sort(key=lambda item: (item[0], item[1]))
    selected_count = neighbor_count
    selected_indices = [index for _, index in scores[:selected_count]]
    selected_updates = [updates[index] for index in selected_indices]
    return aggregate_model_updates_unweighted_mean(selected_updates)


def flatten_floating_state(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    tensors = [
        tensor.detach().float().reshape(-1).cpu()
        for tensor in state_dict.values()
        if tensor.is_floating_point()
    ]
    if not tensors:
        raise ValueError("Model update does not contain floating-point tensors")
    return torch.cat(tensors)


def aggregate_model_updates_unweighted_mean(updates: list[ModelUpdate]) -> dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("No client model updates to aggregate")

    averaged: dict[str, torch.Tensor] = {}
    first_state = updates[0].state_dict
    for name, first_tensor in first_state.items():
        if not first_tensor.is_floating_point():
            averaged[name] = first_tensor.clone()
            continue

        stacked = torch.stack(
            [update.state_dict[name].float() for update in updates],
            dim=0,
        )
        averaged[name] = stacked.mean(dim=0).to(dtype=first_tensor.dtype)
    return averaged


# Poisoning attack implementations live in `fl/python/algorithms/attacks.py` and
# are imported above so callers can continue to reference
# `poison_prototype_update` and `poison_model_update` from this module.


def print_communication(round_comm_bytes: int, total_comm_bytes: int, num_clients: int) -> None:
    avg_client_comm = round_comm_bytes // num_clients if num_clients else 0
    print(
        "  communication: "
        f"round={format_bytes(round_comm_bytes)} "
        f"avg_client={format_bytes(avg_client_comm)} "
        f"total={format_bytes(total_comm_bytes)}"
    )
