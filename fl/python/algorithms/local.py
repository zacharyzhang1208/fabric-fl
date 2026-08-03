"""Pure local training runner."""

from __future__ import annotations

from algorithms.common import (
    CommunicationTotals,
    format_client_accuracies,
    print_aggregator_accuracies,
    print_communication,
    print_model_group_accuracies,
)
from fl_client import FederatedClient


def run_local(
    args,
    clients: list[FederatedClient],
    eval_loaders,
    evaluation_clients: list[int],
) -> CommunicationTotals:
    communication = CommunicationTotals()

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
        )
        if args.model_config == "heterogeneous":
            print_model_group_accuracies(clients, eval_loaders, evaluation_clients)

        communication.add_round(upload_bytes=0, download_bytes=0)
        print_communication(0, 0, communication, args.num_clients)

    return communication
