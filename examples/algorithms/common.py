"""Shared helpers for FL algorithm runners."""

from __future__ import annotations

import copy

import torch

from fl_client import ClientUpdate, FederatedClient, ModelUpdate
from logging_utils import format_bytes


def average_accuracy(clients: list[FederatedClient], loaders, client_ids: list[int] | None = None) -> float:
    if client_ids is None:
        client_ids = list(range(len(clients)))
    if not client_ids:
        raise ValueError("No clients available for accuracy evaluation")
    return sum(clients[client_id].evaluate(loaders[client_id]) for client_id in client_ids) / len(client_ids)


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


def aggregate_model_updates(updates: list[ModelUpdate]) -> dict[str, torch.Tensor]:
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


def poison_prototype_update(
    payload: ClientUpdate,
    attack: str,
    attack_scale: float,
    num_classes: int,
) -> ClientUpdate:
    prototypes = payload.prototypes.detach().clone()
    counts = payload.counts.detach().clone()
    present = counts > 0

    if attack == "zero":
        prototypes[present] = 0
    elif attack == "noise":
        std = prototypes[present].std().item() if present.any() else 1.0
        if std == 0:
            std = 1.0
        prototypes[present] += torch.randn_like(prototypes[present]) * std * attack_scale
    elif attack == "sign_flip":
        prototypes[present] = -attack_scale * prototypes[present]
    elif attack == "scale":
        prototypes[present] = attack_scale * prototypes[present]
    elif attack == "label_shift":
        shift = 1 % num_classes
        prototypes = torch.roll(prototypes, shifts=shift, dims=0)
        counts = torch.roll(counts, shifts=shift, dims=0)
    else:
        raise ValueError(f"Unsupported prototype attack: {attack}")

    return ClientUpdate(
        round_id=payload.round_id,
        client_id=payload.client_id,
        prototypes=prototypes,
        counts=counts,
        payload_bytes=payload.payload_bytes,
    )


def poison_model_update(
    update: ModelUpdate,
    attack: str,
    attack_scale: float,
) -> ModelUpdate:
    if attack == "label_shift":
        raise ValueError("--attack label_shift only applies to --algorithm prototype")

    state_dict = copy.deepcopy(update.state_dict)
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point():
            continue
        if attack == "zero":
            state_dict[name] = torch.zeros_like(tensor)
        elif attack == "noise":
            std = tensor.std().item()
            if std == 0:
                std = 1.0
            state_dict[name] = tensor + torch.randn_like(tensor) * std * attack_scale
        elif attack == "sign_flip":
            state_dict[name] = -attack_scale * tensor
        elif attack == "scale":
            state_dict[name] = attack_scale * tensor
        else:
            raise ValueError(f"Unsupported model attack: {attack}")

    return ModelUpdate(
        round_id=update.round_id,
        client_id=update.client_id,
        state_dict=state_dict,
        num_samples=update.num_samples,
        payload_bytes=update.payload_bytes,
    )


def print_communication(round_comm_bytes: int, total_comm_bytes: int, num_clients: int) -> None:
    avg_client_comm = round_comm_bytes // num_clients if num_clients else 0
    print(
        "  communication: "
        f"round={format_bytes(round_comm_bytes)} "
        f"avg_client={format_bytes(avg_client_comm)} "
        f"total={format_bytes(total_comm_bytes)}"
    )
