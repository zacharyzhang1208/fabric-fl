"""Shared helpers for FL algorithm runners."""

from __future__ import annotations

import torch

from fl_client import ClientUpdate, FederatedClient, ModelUpdate
from logging_utils import format_bytes
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
    parts = []
    single_scope = len(eval_loaders) == 1

    if single_scope:
        # Single scope (local or global): print a single average accuracy metric.
        scope, loaders = next(iter(eval_loaders.items()))
        acc = average_accuracy(clients, loaders, evaluation_clients)
        metric_name = "benign_avg_acc" if malicious_clients else "avg_acc"
        parts.append(f"{metric_name}={acc * 100:5.2f}%")
    else:
        # Multiple scopes (e.g., both): print each scope separately.
        for scope, loaders in eval_loaders.items():
            acc = average_accuracy(clients, loaders, evaluation_clients)
            if malicious_clients:
                metric_name = f"benign_{scope}_avg_acc"
            else:
                metric_name = f"{scope}_avg_acc"
            parts.append(f"{metric_name}={acc * 100:5.2f}%")

    print(
        f"  aggregator: {' '.join(parts)} "
        f"round_payload={round_comm_bytes}B"
    )


def print_shared_model_aggregator_accuracies(
    clients: list[FederatedClient],
    eval_loaders: EvalLoaders,
    evaluation_clients: list[int],
    malicious_clients: set[int],
    round_comm_bytes: int,
) -> None:
    parts = []
    single_scope = len(eval_loaders) == 1
    eval_client_id = evaluation_clients[0]

    if single_scope:
        scope, loaders = next(iter(eval_loaders.items()))
        if scope == "global":
            acc = clients[eval_client_id].evaluate(loaders[eval_client_id])
        else:
            acc = average_accuracy(clients, loaders, evaluation_clients)
        metric_name = "benign_avg_acc" if malicious_clients else "avg_acc"
        parts.append(f"{metric_name}={acc * 100:5.2f}%")
    else:
        for scope, loaders in eval_loaders.items():
            if scope == "global":
                acc = clients[eval_client_id].evaluate(loaders[eval_client_id])
            else:
                acc = average_accuracy(clients, loaders, evaluation_clients)
            if malicious_clients:
                metric_name = f"benign_{scope}_avg_acc"
            else:
                metric_name = f"{scope}_avg_acc"
            parts.append(f"{metric_name}={acc * 100:5.2f}%")

    print(
        f"  aggregator: {' '.join(parts)} "
        f"round_payload={round_comm_bytes}B"
    )


def aggregate_prototypes(
    payloads: list[ClientUpdate],
    device: torch.device,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not payloads:
        raise ValueError("No client payloads to aggregate")

    embed_dim = payloads[0].prototypes.shape[1]
    sums = torch.zeros(num_classes, embed_dim, device=device)
    counts = torch.zeros(num_classes, device=device)

    for payload in payloads:
        client_counts = payload.counts.to(device)
        present = client_counts > 0
        sums[present] += payload.prototypes.to(device)[present]
        counts[present] += 1

    global_prototypes = torch.zeros_like(sums)
    present = counts > 0
    global_prototypes[present] = sums[present] / counts[present].unsqueeze(1)
    return global_prototypes, counts


def aggregate_model_updates(
    updates: list[ModelUpdate],
    aggregation: str = "mean",
    trim_ratio: float = 0.0,
) -> dict[str, torch.Tensor]:
    if aggregation == "mean":
        return aggregate_model_updates_mean(updates)
    if aggregation == "trimmed_mean":
        return aggregate_model_updates_trimmed_mean(updates, trim_ratio)
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
    trim_ratio: float,
) -> dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("No client model updates to aggregate")
    if trim_ratio < 0 or trim_ratio >= 0.5:
        raise ValueError("--trim-ratio must be in [0, 0.5)")

    trim_count = int(len(updates) * trim_ratio)
    if trim_count * 2 >= len(updates):
        raise ValueError("--trim-ratio trims all model updates")

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
