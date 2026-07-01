"""Prototype-sharing runner."""

from __future__ import annotations

import torch

from algorithms.common import (
    aggregate_prototypes,
    average_accuracy,
    poison_prototype_update,
    print_communication,
)
from fl_client import ClientUpdate, FederatedClient


def run_prototype(
    args,
    clients: list[FederatedClient],
    test_loaders,
    evaluation_clients: list[int],
    device: torch.device,
    num_classes: int,
    malicious_clients: set[int],
) -> int:
    global_prototypes: torch.Tensor | None = None
    global_counts: torch.Tensor | None = None
    total_comm_bytes = 0

    for round_id in range(1, args.rounds + 1):
        print(f"\nRound {round_id}")
        round_comm_bytes = 0
        payloads: list[ClientUpdate] = []

        for client in clients:
            metrics = client.train_round(
                local_epochs=args.local_epochs,
                global_prototypes=global_prototypes,
                global_counts=global_counts,
                proto_weight=args.proto_weight,
            )
            payload = client.build_update(round_id=round_id)
            if client.client_id in malicious_clients:
                payload = poison_prototype_update(
                    payload,
                    attack=args.attack,
                    attack_scale=args.attack_scale,
                    num_classes=num_classes,
                )
            payloads.append(payload)
            round_comm_bytes += payload.payload_bytes

            acc = client.evaluate(test_loaders[client.client_id])
            attack_marker = " malicious_upload" if client.client_id in malicious_clients else ""
            metric_text = f"loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f}"
            print(
                f"  client {client.client_id}: {metric_text} "
                f"test_acc={acc * 100:5.2f}% payload={payload.payload_bytes}B{attack_marker}"
            )

        global_prototypes, global_counts = aggregate_prototypes(payloads, device, num_classes)
        avg_acc = average_accuracy(clients, test_loaders, evaluation_clients)
        acc_name = "benign_avg_acc" if malicious_clients else "avg_acc"
        print(
            f"  aggregator: {acc_name}={avg_acc * 100:5.2f}% "
            f"round_payload={round_comm_bytes}B"
        )

        total_comm_bytes += round_comm_bytes
        print_communication(round_comm_bytes, total_comm_bytes, args.num_clients)

    return total_comm_bytes
