"""Prototype sharing with a synchronized global classifier head."""

from __future__ import annotations

import torch

from algorithms.common import (
    aggregate_model_updates_mean,
    aggregate_prototypes,
    format_client_accuracies,
    print_aggregator_accuracies,
    print_communication,
    print_model_group_accuracies,
)
from fl_client import ClientUpdate, FederatedClient, ModelUpdate


def run_prototype_head(
    args,
    clients: list[FederatedClient],
    eval_loaders,
    evaluation_clients: list[int],
    device: torch.device,
    num_classes: int,
) -> int:
    global_prototypes: torch.Tensor | None = None
    global_counts: torch.Tensor | None = None
    total_comm_bytes = 0

    global_classifier = clients[0].get_classifier_state()
    _validate_classifier_compatibility(clients, global_classifier)
    for client in clients:
        client.load_classifier_state(global_classifier)

    for round_id in range(1, args.rounds + 1):
        print(f"\nRound {round_id}")
        round_comm_bytes = 0
        prototype_payloads: list[ClientUpdate] = []
        classifier_payloads: list[ModelUpdate] = []

        for client in clients:
            metrics = client.train_round(
                local_epochs=args.local_epochs,
                global_prototypes=global_prototypes,
                global_counts=global_counts,
                proto_weight=args.proto_weight,
            )
            prototype_payload = client.build_update(round_id=round_id)
            classifier_payload = client.build_classifier_update(round_id=round_id)
            prototype_payloads.append(prototype_payload)
            classifier_payloads.append(classifier_payload)

            client_payload_bytes = (
                prototype_payload.payload_bytes + classifier_payload.payload_bytes
            )
            round_comm_bytes += client_payload_bytes
            acc_text = format_client_accuracies(client, eval_loaders)
            print(
                f"  client {client.client_id}: "
                f"loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f} "
                f"{acc_text} prototype_payload={prototype_payload.payload_bytes}B "
                f"head_payload={classifier_payload.payload_bytes}B "
                f"payload={client_payload_bytes}B"
            )

        global_prototypes, global_counts = aggregate_prototypes(
            prototype_payloads,
            device,
            num_classes,
        )
        global_classifier = aggregate_model_updates_mean(classifier_payloads)
        for client in clients:
            client.load_classifier_state(global_classifier)

        print_aggregator_accuracies(
            clients,
            eval_loaders,
            evaluation_clients,
            set(),
            round_comm_bytes,
        )
        if args.model_config == "heterogeneous":
            print_model_group_accuracies(clients, eval_loaders, evaluation_clients)

        total_comm_bytes += round_comm_bytes
        print(
            "  shared_head: "
            f"parameters={sum(tensor.numel() for tensor in global_classifier.values())} "
            f"payload={classifier_payloads[0].payload_bytes}B/client"
        )
        print_communication(round_comm_bytes, total_comm_bytes, args.num_clients)

    return total_comm_bytes


def _validate_classifier_compatibility(
    clients: list[FederatedClient],
    reference_state: dict[str, torch.Tensor],
) -> None:
    reference_shapes = {
        name: tuple(tensor.shape)
        for name, tensor in reference_state.items()
    }
    for client in clients:
        state = client.get_classifier_state()
        shapes = {
            name: tuple(tensor.shape)
            for name, tensor in state.items()
        }
        if shapes != reference_shapes:
            raise ValueError(
                "prototype_head requires compatible classifier heads; "
                f"client 0 has {reference_shapes}, client {client.client_id} has {shapes}"
            )
