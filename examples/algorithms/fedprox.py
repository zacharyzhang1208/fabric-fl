"""FedProx runner."""

from __future__ import annotations

from algorithms.common import (
    aggregate_model_updates,
    format_client_accuracies,
    poison_model_update,
    print_aggregator_accuracies,
    print_communication,
)
from fl_client import FederatedClient, ModelUpdate


def run_fedprox(
    args,
    clients: list[FederatedClient],
    eval_loaders,
    evaluation_clients: list[int],
    malicious_clients: set[int],
) -> int:
    global_model_state = clients[0].get_model_state()
    for client in clients:
        client.load_model_state(global_model_state)

    total_comm_bytes = 0
    for round_id in range(1, args.rounds + 1):
        print(f"\nRound {round_id}")
        round_comm_bytes = 0
        model_updates: list[ModelUpdate] = []

        for client in clients:
            client.load_model_state(global_model_state)
            metrics = client.train_round(
                local_epochs=args.local_epochs,
                global_prototypes=None,
                global_counts=None,
                proto_weight=0.0,
                proximal_state=global_model_state,
                fedprox_mu=args.fedprox_mu,
            )
            update = client.build_model_update(round_id=round_id)
            if client.client_id in malicious_clients:
                update = poison_model_update(
                    update,
                    attack=args.attack,
                    attack_scale=args.attack_scale,
                )
            model_updates.append(update)
            round_comm_bytes += update.payload_bytes
            acc_text = format_client_accuracies(client, eval_loaders)
            attack_marker = " malicious_upload" if client.client_id in malicious_clients else ""
            print(
                f"  client {client.client_id}: loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f} "
                f"prox={metrics.prox_loss:.4f} {acc_text} "
                f"payload={update.payload_bytes}B{attack_marker}"
            )

        global_model_state = aggregate_model_updates(model_updates)
        for client in clients:
            client.load_model_state(global_model_state)
        print_aggregator_accuracies(
            clients,
            eval_loaders,
            evaluation_clients,
            malicious_clients,
            round_comm_bytes,
            default_clean_name="global_acc",
        )

        total_comm_bytes += round_comm_bytes
        print_communication(round_comm_bytes, total_comm_bytes, args.num_clients)

    return total_comm_bytes
