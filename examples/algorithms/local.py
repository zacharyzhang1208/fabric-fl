"""Pure local training runner."""

from __future__ import annotations

from algorithms.common import format_client_accuracies, print_aggregator_accuracies, print_communication
from fl_client import FederatedClient


def run_local(args, clients: list[FederatedClient], eval_loaders, evaluation_clients: list[int]) -> int:
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
            acc_text = format_client_accuracies(client, eval_loaders)
            print(
                f"  client {client.client_id}: loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f} "
                f"{acc_text} payload=0B"
            )

        print_aggregator_accuracies(
            clients,
            eval_loaders,
            evaluation_clients,
            malicious_clients=set(),
            round_comm_bytes=round_comm_bytes,
            default_clean_name="avg_acc",
        )

        total_comm_bytes += round_comm_bytes
        print_communication(round_comm_bytes, total_comm_bytes, args.num_clients)

    return total_comm_bytes
