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
        make_kn_client_test_loaders,
        make_kn_client_subsets,
    )
    from algorithms.fedavg import run_fedavg
    from algorithms.fedprox import run_fedprox
    from algorithms.local import run_local
    from algorithms.prototype import run_prototype
    from fl_client import FederatedClient
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


def parse_client_ids(raw_ids: str, num_clients: int) -> set[int]:
    if not raw_ids:
        return set()

    client_ids: set[int] = set()
    for item in raw_ids.split(","):
        item = item.strip()
        if not item:
            continue
        client_id = int(item)
        if client_id < 0 or client_id >= num_clients:
            raise ValueError(f"Malicious client id {client_id} is outside [0, {num_clients - 1}]")
        client_ids.add(client_id)
    return client_ids


def select_malicious_clients(args: argparse.Namespace) -> set[int]:
    explicit_clients = parse_client_ids(args.malicious_clients, args.num_clients)
    if explicit_clients:
        return explicit_clients

    if args.malicious_fraction <= 0:
        return set()
    if args.malicious_fraction > 1:
        raise ValueError("--malicious-fraction must be between 0 and 1")

    rng = random.Random(args.attack_seed)
    count = max(1, int(args.num_clients * args.malicious_fraction))
    return set(rng.sample(range(args.num_clients), count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local multi-client FL simulation")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="mnist")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--algorithm", choices=["local", "prototype", "fedavg", "fedprox"], default="prototype")
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
    parser.add_argument("--test-shots-per-class", type=int, default=40, help="Only used when --mode task_heter")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--proto-weight", type=float, default=1.0)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attack", choices=["none", "zero", "noise", "sign_flip", "scale", "label_shift"], default="none")
    parser.add_argument("--attack-scale", type=float, default=10.0)
    parser.add_argument("--attack-seed", type=int, default=2026)
    parser.add_argument("--malicious-clients", default="", help="Comma-separated client ids, e.g. 0,3,7")
    parser.add_argument("--malicious-fraction", type=float, default=0.0)
    parser.add_argument("--log-dir", default="log")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.dirichlet_alpha <= 0:
        raise ValueError("--dirichlet-alpha must be positive")
    if args.attack_scale < 0:
        raise ValueError("--attack-scale must be non-negative")
    if args.fedprox_mu < 0:
        raise ValueError("--fedprox-mu must be non-negative")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    malicious_clients = select_malicious_clients(args)
    if args.attack == "none" and malicious_clients:
        raise ValueError("Malicious clients were configured but --attack is none")
    if args.attack != "none" and not malicious_clients:
        raise ValueError("Set --malicious-clients or --malicious-fraction when --attack is not none")
    if args.algorithm == "local" and args.attack != "none":
        raise ValueError("Upload attacks require --algorithm prototype, fedavg, or fedprox")
    if args.algorithm in {"fedavg", "fedprox"} and args.attack == "label_shift":
        raise ValueError("--attack label_shift only applies to --algorithm prototype")
    evaluation_clients = [
        client_id
        for client_id in range(args.num_clients)
        if client_id not in malicious_clients
    ]
    if not evaluation_clients:
        raise ValueError("At least one honest client is required for accuracy evaluation")

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
    if args.mode == "task_heter":
        test_loaders = make_kn_client_test_loaders(
            client_subsets,
            train_data,
            test_data,
            args.batch_size,
            args.test_shots_per_class,
            args.test_limit,
        )
    else:
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
        print(f"K/N test_shots_per_class: {args.test_shots_per_class}")
    if args.mode == "dirichlet":
        print(f"Dirichlet alpha: {args.dirichlet_alpha}")
    print(f"Rounds: {args.rounds}")
    if args.attack == "none":
        print("Attack: none")
    else:
        print(f"Attack: {args.attack}")
        print(f"Attack scale: {args.attack_scale}")
        print(f"Malicious clients: {sorted(malicious_clients)}")
        print(f"Accuracy clients: {evaluation_clients}")
    if args.test_limit is not None:
        print(f"Per-client local test limit: {args.test_limit}")
    if args.algorithm == "prototype":
        print(f"Prototype loss weight: {args.proto_weight}")
        print(f"Optimizer: {args.optimizer}")
    if args.algorithm == "fedprox":
        print(f"FedProx mu: {args.fedprox_mu}")
        print(f"Optimizer: {args.optimizer}")
    print()
    print("Client label histograms:")
    for client_id, subset in enumerate(client_subsets):
        print(f"  client {client_id}: {class_histogram(subset, train_data, dataset_spec.num_classes)}")
    print("Client local test label histograms:")
    for client_id, loader in enumerate(test_loaders):
        print(f"  client {client_id}: {class_histogram(loader.dataset, test_data, dataset_spec.num_classes)}")

    if args.algorithm == "local":
        total_comm_bytes = run_local(args, clients, test_loaders, evaluation_clients)
    elif args.algorithm == "prototype":
        total_comm_bytes = run_prototype(
            args,
            clients,
            test_loaders,
            evaluation_clients,
            device,
            dataset_spec.num_classes,
            malicious_clients,
        )
    elif args.algorithm == "fedavg":
        total_comm_bytes = run_fedavg(args, clients, test_loaders, evaluation_clients, malicious_clients)
    elif args.algorithm == "fedprox":
        total_comm_bytes = run_fedprox(args, clients, test_loaders, evaluation_clients, malicious_clients)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    print("\nFinal communication summary")
    print("===========================")
    if args.algorithm in {"fedavg", "fedprox"}:
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
