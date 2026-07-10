"""Poisoning attack implementations separated into their own module.

This module provides `poison_prototype_update` and `poison_model_update`.
"""

from __future__ import annotations

import copy

import torch

from fl_client import ClientUpdate, ModelUpdate


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
