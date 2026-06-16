#!/usr/bin/env python3
"""Local multi-client FL simulation on image datasets.

This is step 1 before wiring the same payloads into Fabric PDC:

1. Split the selected dataset into non-IID client datasets.
2. Run either prototype distillation or standard FedAvg.
3. Report round accuracy and communication bytes for comparison.

Run:
    python3 examples/main.py

Dependencies:
    pip install -r examples/requirements.txt
"""

from __future__ import annotations

import argparse
import random
import sys

try:
    import torch
    import numpy as np
    from data import (
        DATASET_SPECS,
        class_histogram,
        load_image_dataset,
        make_client_loaders,
        make_client_test_loaders,
        make_dirichlet_client_subsets,
        make_kn_client_subsets,
    )
    from fl_client import ClientUpdate, FederatedClient, ModelUpdate
    from logging_utils import format_bytes, make_log_path, redirect_output_to_log
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    print(f"Missing dependency: {missing}", file=sys.stderr)
    print("Install demo dependencies with:", file=sys.stderr)
    print("  python3 -m venv .venv", file=sys.stderr)
    print("  source .venv/bin/activate", file=sys.stderr)
    print("  python -m pip install -r examples/requirements.txt", file=sys.stderr)
    raise SystemExit(1) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def average_accuracy(clients: list[FederatedClient], loaders) -> float:
    return sum(client.evaluate(loader) for client, loader in zip(clients, loaders)) / len(clients)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local multi-client FL simulation")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="mnist")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--algorithm", choices=["local", "prototype", "fedavg"], default="prototype")
    parser.add_argument("--mode", choices=["task_heter", "dirichlet"], default="task_heter")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--ways", type=int, default=3, help="K/N classes per client center")
    parser.add_argument("--shots", type=int, default=100, help="K/N samples per class center")
    parser.add_argument("--stdev", type=int, default=2, help="K/N ways/shots random spread")
    parser.add_argument("--train-shots-max", type=int, default=110, help="K/N per-class index stride")
    parser.add_argument("--samples-per-client", type=int, default=300, help="Only used when --mode dirichlet")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5, help="Only used when --mode dirichlet")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--proto-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-dir", default="log")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.dirichlet_alpha <= 0:
        raise ValueError("--dirichlet-alpha must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train_data, test_data, dataset_spec = load_image_dataset(
            args.dataset,
            args.data_dir,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    if args.mode == "task_heter":
        client_subsets = make_kn_client_subsets(
            train_data,
            num_classes=dataset_spec.num_classes,
            num_clients=args.num_clients,
            ways=args.ways,
            shots=args.shots,
            stdev=args.stdev,
            train_shots_max=args.train_shots_max,
            seed=args.seed,
        )
    else:
        client_subsets = make_dirichlet_client_subsets(
            train_data,
            num_classes=dataset_spec.num_classes,
            num_clients=args.num_clients,
            samples_per_client=args.samples_per_client,
            alpha=args.dirichlet_alpha,
            seed=args.seed + 1,
        )
    client_loaders, proto_loaders = make_client_loaders(client_subsets, args.batch_size)
    test_loaders = make_client_test_loaders(
        client_subsets,
        train_data,
        test_data,
        args.batch_size,
        args.test_limit,
    )

    clients = [
        FederatedClient(
            client_id=client_id,
            train_loader=client_loaders[client_id],
            prototype_loader=proto_loaders[client_id],
            device=device,
            lr=args.lr,
            input_shape=dataset_spec.input_shape,
            num_classes=dataset_spec.num_classes,
            dataset_name=dataset_spec.name,
            optimizer_name=args.optimizer,
        )
        for client_id in range(args.num_clients)
    ]
    global_prototypes: torch.Tensor | None = None
    global_counts: torch.Tensor | None = None
    total_comm_bytes = 0

    print(f"Log file: {args.log_path}")
    print(f"Command: {' '.join(sys.argv)}")
    print()
    print("Local FL simulation")
    print("===================")
    print(f"Dataset: {dataset_spec.name}")
    print(f"Device: {device}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Mode: {args.mode}")
    print(f"Clients: {args.num_clients}")
    if args.mode == "task_heter":
        print(f"K/N ways/shots/stdev: {args.ways}/{args.shots}/{args.stdev}")
        print(f"K/N train_shots_max: {args.train_shots_max}")
    if args.mode == "dirichlet":
        print(f"Dirichlet alpha: {args.dirichlet_alpha}")
    print(f"Rounds: {args.rounds}")
    if args.test_limit is not None:
        print(f"Per-client local test limit: {args.test_limit}")
    if args.algorithm == "prototype":
        print(f"Prototype loss weight: {args.proto_weight}")
        print(f"Optimizer: {args.optimizer}")
    print()
    print("Client label histograms:")
    for client_id, subset in enumerate(client_subsets):
        print(f"  client {client_id}: {class_histogram(subset, train_data, dataset_spec.num_classes)}")
    print("Client local test label histograms:")
    for client_id, loader in enumerate(test_loaders):
        print(f"  client {client_id}: {class_histogram(loader.dataset, test_data, dataset_spec.num_classes)}")

    if args.algorithm == "fedavg":
        global_model_state = clients[0].get_model_state()
        for client in clients:
            client.load_model_state(global_model_state)

    for round_id in range(1, args.rounds + 1):
        print(f"\nRound {round_id}")
        round_comm_bytes = 0

        if args.algorithm == "local":
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
            avg_acc = average_accuracy(clients, test_loaders)
            print(
                f"  local: avg_acc={avg_acc * 100:5.2f}% "
                "round_payload=0B"
            )
        elif args.algorithm == "prototype":
            payloads: list[ClientUpdate] = []

            for client in clients:
                metrics = client.train_round(
                    local_epochs=args.local_epochs,
                    global_prototypes=global_prototypes,
                    global_counts=global_counts,
                    proto_weight=args.proto_weight,
                )
                payload = client.build_update(round_id=round_id)
                payloads.append(payload)
                round_comm_bytes += payload.payload_bytes

                acc = client.evaluate(test_loaders[client.client_id])
                metric_text = f"loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f}"
                print(
                    f"  client {client.client_id}: {metric_text} "
                    f"test_acc={acc * 100:5.2f}% payload={payload.payload_bytes}B"
                )

            global_prototypes, global_counts = aggregate_prototypes(payloads, device, dataset_spec.num_classes)
            avg_acc = average_accuracy(clients, test_loaders)
            print(
                f"  aggregator: avg_acc={avg_acc * 100:5.2f}% "
                f"round_payload={round_comm_bytes}B"
            )
        else:
            model_updates: list[ModelUpdate] = []
            for client in clients:
                client.load_model_state(global_model_state)
                metrics = client.train_round(
                    local_epochs=args.local_epochs,
                    global_prototypes=None,
                    global_counts=None,
                    proto_weight=0.0,
                )
                update = client.build_model_update(round_id=round_id)
                model_updates.append(update)
                round_comm_bytes += update.payload_bytes
                acc = client.evaluate(test_loaders[client.client_id])
                print(
                    f"  client {client.client_id}: loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f} "
                    f"local_test_acc={acc * 100:5.2f}% payload={update.payload_bytes}B"
                )

            global_model_state = aggregate_model_updates(model_updates)
            for client in clients:
                client.load_model_state(global_model_state)
            avg_acc = average_accuracy(clients, test_loaders)
            print(
                f"  aggregator: global_acc={avg_acc * 100:5.2f}% "
                f"round_payload={round_comm_bytes}B"
            )

        total_comm_bytes += round_comm_bytes
        avg_client_comm = round_comm_bytes // args.num_clients if args.num_clients else 0
        print(
            "  communication: "
            f"round={format_bytes(round_comm_bytes)} "
            f"avg_client={format_bytes(avg_client_comm)} "
            f"total={format_bytes(total_comm_bytes)}"
        )

    print("\nFinal communication summary")
    print("===========================")
    if args.algorithm == "fedavg":
        payload_name = "model"
    elif args.algorithm == "local":
        payload_name = "local"
    else:
        payload_name = "prototype"
    print(f"Total {payload_name} communication: {format_bytes(total_comm_bytes)}")


def main() -> None:
    args = parse_args()
    args.log_path = make_log_path(args)
    with redirect_output_to_log(args.log_path):
        run(args)


if __name__ == "__main__":
    main()
