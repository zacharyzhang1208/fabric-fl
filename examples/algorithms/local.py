"""Pure local training runner."""

from __future__ import annotations

from algorithms.common import average_accuracy, print_communication
from fl_client import FederatedClient


def run_local(args, clients: list[FederatedClient], test_loaders, evaluation_clients: list[int]) -> int:
    total_comm_bytes = 0

    for round_id in range(1, args.rounds + 1):
        print(f"\nRound {round_id}")
        round_comm_bytes = 0

        for client in clients:
            metrics = client.train_round(
                local_epochs=args.local_epochs,
                global_prototypes=None,
                global_counts=None,
                proto_weight=0.0,
            )
            acc = client.evaluate(test_loaders[client.client_id])
            print(
                f"  client {client.client_id}: loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f} "
                f"local_test_acc={acc * 100:5.2f}% payload=0B"
            )

        avg_acc = average_accuracy(clients, test_loaders, evaluation_clients)
        print(
            f"  local: avg_acc={avg_acc * 100:5.2f}% "
            "round_payload=0B"
        )

        total_comm_bytes += round_comm_bytes
        print_communication(round_comm_bytes, total_comm_bytes, args.num_clients)

    return total_comm_bytes
