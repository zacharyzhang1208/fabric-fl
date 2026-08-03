"""Prototype-sharing runner."""

from __future__ import annotations

import torch

from algorithms.common import (
    aggregate_prototypes,
    format_client_accuracies,
    poison_prototype_update,
    print_aggregator_accuracies,
    print_communication,
    print_local_class_accuracies,
    print_model_group_accuracies,
)
from fabric_adapter import FabricAdapterClient, PrototypePayload, ReputationReport
from fabric_traffic import FabricTrafficMonitor
from fl_client import ClientUpdate, FederatedClient
from logging_utils import format_bytes


def run_prototype(
    args,
    clients: list[FederatedClient],
    eval_loaders,
    evaluation_clients: list[int],
    device: torch.device,
    num_classes: int,
    malicious_clients: set[int],
    client_label_sets: list[set[int]],
) -> int:
    global_prototypes: torch.Tensor | None = None
    global_counts: torch.Tensor | None = None
    total_comm_bytes = 0
    adapter = None
    traffic_monitor = None
    if args.backend == "fabric":
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

        for client in clients:
            metrics = client.train_round(
                local_epochs=args.local_epochs,
                global_prototypes=global_prototypes,
                global_counts=global_counts,
                proto_weight=args.proto_weight,
                proto_temperature=args.proto_temperature,
                prototype_classes=client_label_sets[client.client_id],
                prototypes_per_class=args.prototypes_per_class,
                min_samples_per_prototype=args.min_samples_per_prototype,
            )
            payload = client.build_update(round_id=round_id)
            if client.client_id in malicious_clients:
                payload = poison_prototype_update(
                    payload,
                    attack=args.attack,
                    attack_scale=args.attack_scale,
                    num_classes=num_classes,
                    flip_source_class=args.flip_source_class,
                    flip_target_class=args.flip_target_class,
                )
            payloads.append(payload)
            round_comm_bytes += payload.payload_bytes

            acc_text = format_client_accuracies(client, eval_loaders)
            attack_marker = " malicious_upload" if client.client_id in malicious_clients else ""
            metric_text = (
                f"loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f} "
                f"proto_cls={metrics.proto_loss:.4f}"
            )
            print(
                f"  client {client.client_id}: {metric_text} "
                f"{acc_text} payload={payload.payload_bytes}B{attack_marker}"
            )

        if adapter is None:
            global_prototypes, global_counts = aggregate_prototypes(
                payloads,
                device,
                num_classes,
                previous_prototypes=global_prototypes,
                previous_counts=global_counts,
            )
        else:
            assert ledger_round_id is not None
            global_prototypes, global_counts, report = aggregate_prototypes_via_fabric(
                adapter=adapter,
                ledger_round_id=ledger_round_id,
                experiment_id=args.fabric_round_base,
                sequence=round_id,
                payloads=payloads,
                device=device,
                num_classes=num_classes,
                scale=args.prototype_scale,
            )
            print(
                f"  fabric_batch: client_submissions={len(payloads)} "
                "round_transactions=1 total_write_transactions=1"
            )
            print(f"  fabric: ledger_round={ledger_round_id} status=FINALIZED")
            print_reputation_report(report, malicious_clients)
        print_aggregator_accuracies(
            clients,
            eval_loaders,
            evaluation_clients,
            malicious_clients,
            round_comm_bytes,
        )
        if "local" in eval_loaders:
            print_local_class_accuracies(
                clients,
                eval_loaders["local"],
                evaluation_clients,
                client_label_sets,
                global_prototypes,
                global_counts,
                malicious_clients,
            )
        if args.model_config == "heterogeneous":
            print_model_group_accuracies(clients, eval_loaders, evaluation_clients)
        if args.attack == "targeted_label_flip":
            print_targeted_flip_rate(
                clients,
                eval_loaders,
                evaluation_clients,
                args.flip_source_class,
                args.flip_target_class,
            )

        total_comm_bytes += round_comm_bytes
        print_communication(round_comm_bytes, total_comm_bytes, args.num_clients)
        if traffic_monitor is not None:
            round_traffic, total_traffic = traffic_monitor.round_delta()
            print(
                "  fabric_traffic: "
                f"round_rx={format_bytes(round_traffic.rx_bytes)} "
                f"round_tx={format_bytes(round_traffic.tx_bytes)} "
                f"round_total={format_bytes(round_traffic.total_bytes)} "
                f"total_rx={format_bytes(total_traffic.rx_bytes)} "
                f"total_tx={format_bytes(total_traffic.tx_bytes)} "
                f"total={format_bytes(total_traffic.total_bytes)}"
            )

    return total_comm_bytes


def aggregate_prototypes_via_fabric(
    adapter: FabricAdapterClient,
    ledger_round_id: int,
    experiment_id: int,
    sequence: int,
    payloads: list[ClientUpdate],
    device: torch.device,
    num_classes: int,
    scale: int,
) -> tuple[torch.Tensor, torch.Tensor, ReputationReport]:
    if not payloads:
        raise ValueError("No client payloads to aggregate")

    dimension = int(payloads[0].prototypes.shape[1])
    wire_payloads = [
        PrototypePayload.from_tensors(
            round_id=ledger_round_id,
            client_id=payload.client_id,
            prototypes=payload.prototypes,
            counts=payload.counts,
            scale=scale,
        )
        for payload in payloads
    ]
    result = adapter.upload_prototype_batch(
        wire_payloads,
        experiment_id=experiment_id,
        sequence=sequence,
    )
    global_payload = result.global_prototype
    if global_payload.shape != (num_classes, dimension):
        raise ValueError(
            "Global prototype shape "
            f"{global_payload.shape} does not match expected {(num_classes, dimension)}"
        )
    if global_payload.scale != scale:
        raise ValueError(
            f"Global prototype scale {global_payload.scale} does not match expected {scale}"
        )
    prototypes, counts = global_payload.to_tensors(device=device)
    return prototypes, counts, result.reputation_report


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


def print_targeted_flip_rate(
    clients,
    eval_loaders,
    evaluation_clients: list[int],
    source_class: int,
    target_class: int,
) -> None:
    if "global" not in eval_loaders:
        return
    loader = eval_loaders["global"][evaluation_clients[0]]
    rates = [
        clients[client_id].evaluate_target_rate(
            loader,
            source_class=source_class,
            target_class=target_class,
        )
        for client_id in evaluation_clients
    ]
    target_rate = sum(rates) / len(rates) if rates else 0.0
    print(
        "  targeted_label_flip: "
        f"source={source_class} target={target_class} "
        f"target_rate={target_rate * 100:5.2f}%"
    )
