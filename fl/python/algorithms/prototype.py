"""Prototype-sharing runner."""

from __future__ import annotations

import torch

from algorithms.common import (
    aggregate_prototypes,
    format_client_accuracies,
    poison_prototype_update,
    print_aggregator_accuracies,
    print_communication,
)
from fabric_adapter import FabricAdapterClient, PrototypePayload
from fabric_traffic import FabricTrafficMonitor
from fl_client import ClientUpdate, FederatedClient


def run_prototype(
    args,
    clients: list[FederatedClient],
    eval_loaders,
    evaluation_clients: list[int],
    device: torch.device,
    num_classes: int,
    malicious_clients: set[int],
) -> int:
    global_prototypes: torch.Tensor | None = None
    global_counts: torch.Tensor | None = None
    total_comm_bytes = 0
    adapter = None
    traffic_monitor = None
    if args.prototype_backend == "fabric":
        adapter = FabricAdapterClient(
            base_url=args.fabric_adapter_url,
            timeout=args.fabric_timeout,
        )
        if args.fabric_traffic:
            traffic_monitor = FabricTrafficMonitor()
            print(
                "Fabric traffic monitor: containers=10 "
                "(5 peers + 5 orderers), interface=eth0"
            )

    for round_id in range(1, args.rounds + 1):
        print(f"\nRound {round_id}")
        round_comm_bytes = 0
        payloads: list[ClientUpdate] = []
        ledger_round_id = None

        if adapter is not None:
            ledger_round_id = args.fabric_round_base + round_id - 1
            dimension = int(getattr(clients[0].model, "prototype_dim"))
            adapter.create_round(
                round_id=ledger_round_id,
                experiment_id=args.fabric_round_base,
                sequence=round_id,
                expected_clients=len(clients),
                num_classes=num_classes,
                dimension=dimension,
                scale=args.prototype_scale,
            )

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

            acc_text = format_client_accuracies(client, eval_loaders)
            attack_marker = " malicious_upload" if client.client_id in malicious_clients else ""
            metric_text = f"loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f}"
            print(
                f"  client {client.client_id}: {metric_text} "
                f"{acc_text} payload={payload.payload_bytes}B{attack_marker}"
            )

        if adapter is None:
            global_prototypes, global_counts = aggregate_prototypes(payloads, device, num_classes)
        else:
            assert ledger_round_id is not None
            global_prototypes, global_counts = aggregate_prototypes_via_fabric(
                adapter=adapter,
                ledger_round_id=ledger_round_id,
                payloads=payloads,
                device=device,
                num_classes=num_classes,
                scale=args.prototype_scale,
            )
            print(f"  fabric: ledger_round={ledger_round_id} status=FINALIZED")
            report = adapter.get_round_reputation_report(ledger_round_id)
            print_reputation_report(report, malicious_clients)
        print_aggregator_accuracies(
            clients,
            eval_loaders,
            evaluation_clients,
            malicious_clients,
            round_comm_bytes,
        )

        total_comm_bytes += round_comm_bytes
        print_communication(round_comm_bytes, total_comm_bytes, args.num_clients)
        if traffic_monitor is not None:
            round_traffic, total_traffic = traffic_monitor.round_delta()
            print(
                "  fabric_traffic: "
                f"round_rx={round_traffic.rx_bytes}B "
                f"round_tx={round_traffic.tx_bytes}B "
                f"round_total={round_traffic.total_bytes}B "
                f"total_rx={total_traffic.rx_bytes}B "
                f"total_tx={total_traffic.tx_bytes}B "
                f"total={total_traffic.total_bytes}B"
            )

    return total_comm_bytes


def aggregate_prototypes_via_fabric(
    adapter: FabricAdapterClient,
    ledger_round_id: int,
    payloads: list[ClientUpdate],
    device: torch.device,
    num_classes: int,
    scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not payloads:
        raise ValueError("No client payloads to aggregate")

    dimension = int(payloads[0].prototypes.shape[1])
    for payload in payloads:
        wire_payload = PrototypePayload.from_tensors(
            round_id=ledger_round_id,
            client_id=payload.client_id,
            prototypes=payload.prototypes,
            counts=payload.counts,
            scale=scale,
        )
        adapter.upload_prototype(wire_payload)

    adapter.finalize_round(ledger_round_id)
    global_payload = adapter.get_global_prototype(ledger_round_id)
    if global_payload.shape != (num_classes, dimension):
        raise ValueError(
            "Global prototype shape "
            f"{global_payload.shape} does not match expected {(num_classes, dimension)}"
        )
    if global_payload.scale != scale:
        raise ValueError(
            f"Global prototype scale {global_payload.scale} does not match expected {scale}"
        )
    return global_payload.to_tensors(device=device)


def print_reputation_report(report, malicious_clients: set[int]) -> None:
    anomalous = {item.client_id for item in report.assessments if item.anomalous}
    excluded = {item.client_id for item in report.assessments if not item.included}
    scores = ", ".join(
        f"{item.client_id}:{item.new_score}/{item.status}" for item in report.assessments
    )
    mode = "warmup" if report.warmup else "filtering"
    print(
        f"  reputation: mode={mode} threshold={report.threshold} "
        f"anomalous={sorted(anomalous)} excluded={sorted(excluded)}"
    )
    print(f"  reputation_scores: {scores}")

    if malicious_clients:
        all_clients = {item.client_id for item in report.assessments}
        true_positive = len(anomalous & malicious_clients)
        false_positive = len(anomalous - malicious_clients)
        false_negative = len(malicious_clients - anomalous)
        true_negative = len(all_clients - malicious_clients - anomalous)
        precision = true_positive / (true_positive + false_positive) if anomalous else 0.0
        recall = true_positive / len(malicious_clients)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr_denominator = false_positive + true_negative
        fpr = false_positive / fpr_denominator if fpr_denominator else 0.0
        print(
            f"  detection: precision={precision:.3f} recall={recall:.3f} "
            f"f1={f1:.3f} fpr={fpr:.3f} fn={false_negative}"
        )
