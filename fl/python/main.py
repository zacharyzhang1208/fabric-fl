#!/usr/bin/env python3
"""Local multi-client FL simulation on image datasets.

This is step 1 before wiring the same payloads into Fabric PDC:

1. Split the selected dataset into non-IID client datasets.
2. Run either prototype distillation or standard FedAvg.
3. Report round accuracy and communication bytes for comparison.

Run:
    python3 fl/python/main.py --dataset mnist --algorithm prototype

Dependencies:
    pip install -r fl/python/requirements.txt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path


FL_ROOT = Path(__file__).resolve().parents[1]
DATASET_CHOICES = ("cifar10", "cifar100", "mnist")

from logging_utils import format_bytes, make_log_path, redirect_output_to_log


def load_runtime_dependencies() -> None:
    global DATASET_SPECS
    global FederatedClient
    global class_histogram
    global load_image_dataset
    global make_client_loaders
    global make_client_test_loaders
    global make_dirichlet_client_subsets
    global make_global_test_loaders
    global make_kn_client_subsets
    global make_kn_client_test_loaders
    global model_name_for_client
    global subset_label_set
    global np
    global run_fedavg
    global run_fedprox
    global run_local
    global run_prototype
    global torch

    try:
        import numpy as np
        import torch
        from algorithms.fedavg import run_fedavg
        from algorithms.fedprox import run_fedprox
        from algorithms.local import run_local
        from algorithms.prototype import run_prototype
        from data import (
            DATASET_SPECS,
            class_histogram,
            load_image_dataset,
            make_client_loaders,
            make_client_test_loaders,
            make_dirichlet_client_subsets,
            make_global_test_loaders,
            make_kn_client_subsets,
            make_kn_client_test_loaders,
            subset_label_set,
        )
        from fl_client import FederatedClient
        from models import model_name_for_client
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        print(f"Missing dependency: {missing}", file=sys.stderr)
        print("Install demo dependencies with:", file=sys.stderr)
        print("  python3 -m venv .venv", file=sys.stderr)
        print("  source .venv/bin/activate", file=sys.stderr)
        print("  python -m pip install -r fl/python/requirements.txt", file=sys.stderr)
        raise SystemExit(1) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_malicious_clients(args: argparse.Namespace) -> set[int]:
    if args.malicious_fraction <= 0:
        return set()
    if args.malicious_fraction > 1:
        raise ValueError("--malicious-fraction must be between 0 and 1")

    rng = random.Random(args.attack_seed)
    count = max(1, int(args.num_clients * args.malicious_fraction))
    return set(rng.sample(range(args.num_clients), count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local multi-client FL simulation")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--data-dir", default=str(FL_ROOT / "data"))
    parser.add_argument(
        "--algorithm",
        choices=[
            "local",
            "prototype",
            "fedavg",
            "fedprox",
            "trimmed_mean",
            "multi_krum",
        ],
        required=True,
    )
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument(
        "--model-config",
        choices=["homogeneous", "heterogeneous"],
        default="homogeneous",
        help="Homogeneous CNNs or automatic MLP/CNN/MiniResNet client groups",
    )
    parser.add_argument(
        "--partition",
        choices=["beta", "kn"],
        default="beta",
        help="Client data partition: Dirichlet beta or n-way k-shot",
    )
    parser.add_argument("--samples-per-client", type=int, default=300)
    parser.add_argument("--beta", type=float, default=0.5, help="Dirichlet beta for non-IID client label distributions")
    parser.add_argument("--ways", type=int, default=3)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--stdev", type=int, default=2)
    parser.add_argument("--train-shots-max", type=int, default=110)
    parser.add_argument("--test-shots-per-class", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Evaluation batch size; does not affect local training updates",
    )
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--eval-scope", choices=["local", "global", "both"], default="local")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--proto-weight", type=float, default=0.5)
    parser.add_argument(
        "--proto-temperature",
        type=float,
        default=0.1,
        help="Temperature for softmax over normalized prototype cosine similarities",
    )
    parser.add_argument("--prototypes-per-class", type=int, default=1)
    parser.add_argument("--min-samples-per-prototype", type=int, default=10)
    parser.add_argument(
        "--prototype-synthesis",
        action="store_true",
        help="Use filtered prototype-guided MNIST pseudo-images for rare local classes",
    )
    parser.add_argument("--synthesis-start-round", type=int, default=5)
    parser.add_argument("--synthesis-target-count", type=int, default=10)
    parser.add_argument("--synthetic-samples-per-class", type=int, default=4)
    parser.add_argument("--synthesis-steps", type=int, default=40)
    parser.add_argument("--synthesis-lr", type=float, default=0.1)
    parser.add_argument("--synthesis-weight", type=float, default=0.25)
    parser.add_argument("--synthesis-min-margin", type=float, default=0.05)
    parser.add_argument("--synthesis-tv-weight", type=float, default=0.001)
    parser.add_argument(
        "--backend",
        dest="backend",
        choices=["memory", "fabric"],
        default="memory",
        help="Aggregation backend. The fabric backend currently applies to prototype only.",
    )
    parser.add_argument("--prototype-scale", type=int, default=1_000_000)
    parser.add_argument(
        "--fabric-adapter-url",
        default=os.environ.get("FABRIC_ADAPTER_URL", "http://127.0.0.1:18080"),
        help="Fabric adapter HTTP URL, or FABRIC_ADAPTER_URL",
    )
    parser.add_argument("--fabric-timeout", type=float, default=45.0)
    parser.add_argument(
        "--fabric-traffic",
        action="store_true",
        help="Print real Fabric peer/orderer Docker network byte deltas each round",
    )
    parser.add_argument(
        "--fabric-round-base",
        type=int,
        default=None,
        help="First ledger round id; defaults to the current Unix time in milliseconds",
    )
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--attack",
        choices=["none", "zero", "noise", "sign_flip", "scale", "label_shift", "targeted_label_flip"],
        default="none",
    )
    parser.add_argument("--attack-scale", type=float, default=10.0)
    parser.add_argument("--attack-seed", type=int, default=2026)
    parser.add_argument("--flip-source-class", type=int, default=0)
    parser.add_argument("--flip-target-class", type=int, default=1)
    parser.add_argument("--malicious-fraction", type=float, default=0.0)
    parser.add_argument("--log-dir", default=str(FL_ROOT / "log"))
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.partition == "beta":
        if args.beta <= 0:
            raise ValueError("--beta must be positive")
        if args.samples_per_client <= 0:
            raise ValueError("--samples-per-client must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.attack_scale < 0:
        raise ValueError("--attack-scale must be non-negative")
    if args.fedprox_mu < 0:
        raise ValueError("--fedprox-mu must be non-negative")
    if args.proto_temperature <= 0:
        raise ValueError("--proto-temperature must be positive")
    if args.prototypes_per_class <= 0:
        raise ValueError("--prototypes-per-class must be positive")
    if args.min_samples_per_prototype <= 0:
        raise ValueError("--min-samples-per-prototype must be positive")
    if args.prototype_synthesis:
        if args.algorithm != "prototype":
            raise ValueError("--prototype-synthesis requires --algorithm prototype")
        if args.backend != "memory":
            raise ValueError("--prototype-synthesis currently requires --backend memory")
        if args.dataset != "mnist":
            raise ValueError("--prototype-synthesis currently supports --dataset mnist only")
        if args.attack != "none":
            raise ValueError("--prototype-synthesis currently supports clean experiments only")
    if args.synthesis_start_round < 2:
        raise ValueError("--synthesis-start-round must be at least 2")
    if args.synthesis_target_count <= 0:
        raise ValueError("--synthesis-target-count must be positive")
    if args.synthetic_samples_per_class <= 0:
        raise ValueError("--synthetic-samples-per-class must be positive")
    if args.synthesis_steps <= 0:
        raise ValueError("--synthesis-steps must be positive")
    if args.synthesis_lr <= 0:
        raise ValueError("--synthesis-lr must be positive")
    if args.synthesis_weight <= 0:
        raise ValueError("--synthesis-weight must be positive")
    if args.synthesis_min_margin < 0:
        raise ValueError("--synthesis-min-margin must be non-negative")
    if args.synthesis_tv_weight < 0:
        raise ValueError("--synthesis-tv-weight must be non-negative")
    if args.prototype_scale <= 0:
        raise ValueError("--prototype-scale must be positive")
    if args.fabric_timeout <= 0:
        raise ValueError("--fabric-timeout must be positive")
    if args.fabric_round_base is not None and args.fabric_round_base < 1:
        raise ValueError("--fabric-round-base must be positive")
    if args.algorithm != "prototype" and args.backend != "memory":
        raise ValueError("--backend fabric requires --algorithm prototype")
    if args.prototypes_per_class > 1 and args.algorithm != "prototype":
        raise ValueError("--prototypes-per-class greater than 1 requires --algorithm prototype")
    if args.prototypes_per_class > 1 and args.backend != "memory":
        raise ValueError("multiple prototypes per class currently require --backend memory")
    if args.prototypes_per_class > 1 and args.attack != "none":
        raise ValueError("multiple prototypes per class currently support clean experiments only")
    if args.model_config == "heterogeneous":
        if args.dataset != "mnist":
            raise ValueError("--model-config heterogeneous currently requires --dataset mnist")
        if args.num_clients < 3:
            raise ValueError("--model-config heterogeneous requires at least 3 clients")
        if args.algorithm not in {"local", "prototype"}:
            raise ValueError(
                "--model-config heterogeneous requires --algorithm local or prototype; "
                "model parameters are incompatible with aggregation"
            )
    if args.algorithm == "prototype" and args.backend == "fabric" and args.fabric_round_base is None:
        args.fabric_round_base = time.time_ns() // 1_000_000

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    malicious_clients = select_malicious_clients(args)
    args.krum_f = len(malicious_clients)
    args.trim_count = len(malicious_clients)
    if args.algorithm == "multi_krum" and args.num_clients <= 2 * args.krum_f + 2:
        raise ValueError(
            "--algorithm multi_krum requires num_clients > 2 * malicious_clients + 2"
        )
    if args.algorithm == "trimmed_mean" and args.trim_count * 2 >= args.num_clients:
        raise ValueError(
            "--algorithm trimmed_mean requires num_clients > 2 * malicious_clients"
        )
    if args.attack == "none" and malicious_clients:
        raise ValueError("Malicious clients were configured but --attack is none")
    if args.attack != "none" and not malicious_clients:
        raise ValueError("Set --malicious-fraction greater than 0 when --attack is not none")
    if args.algorithm == "local" and args.attack != "none":
        raise ValueError("Upload attacks require --algorithm prototype, fedavg, fedprox, trimmed_mean, or multi_krum")
    if args.algorithm in {"fedavg", "fedprox", "trimmed_mean", "multi_krum"} and args.attack == "label_shift":
        raise ValueError("--attack label_shift only applies to --algorithm prototype")
    if args.algorithm in {"fedavg", "fedprox", "trimmed_mean", "multi_krum"} and args.attack == "targeted_label_flip":
        raise ValueError("--attack targeted_label_flip only applies to --algorithm prototype")
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
    if args.partition == "beta":
        client_subsets = make_dirichlet_client_subsets(
            train_data,
            num_classes=dataset_spec.num_classes,
            num_clients=args.num_clients,
            samples_per_client=args.samples_per_client,
            alpha=args.beta,
            seed=args.seed + 1,
        )
    else:
        client_subsets = make_kn_client_subsets(
            train_data,
            num_classes=dataset_spec.num_classes,
            num_clients=args.num_clients,
            ways=args.ways,
            shots=args.shots,
            stdev=args.stdev,
            train_shots_max=args.train_shots_max,
            seed=args.seed + 1,
        )
    if args.attack == "targeted_label_flip":
        if args.flip_source_class == args.flip_target_class:
            raise ValueError("--flip-source-class and --flip-target-class must differ")
        for value, name in (
            (args.flip_source_class, "--flip-source-class"),
            (args.flip_target_class, "--flip-target-class"),
        ):
            if value < 0 or value >= dataset_spec.num_classes:
                raise ValueError(f"{name} must be in [0, {dataset_spec.num_classes - 1}]")
    client_loaders, proto_loaders = make_client_loaders(client_subsets, args.batch_size)
    if args.partition == "beta":
        local_test_loaders = make_client_test_loaders(
            client_subsets,
            train_data,
            test_data,
            args.eval_batch_size,
            args.test_limit,
            seed=args.seed + 2,
        )
    else:
        local_test_loaders = make_kn_client_test_loaders(
            client_subsets,
            train_data,
            test_data,
            args.eval_batch_size,
            args.test_shots_per_class,
            args.test_limit,
        )
    global_test_loaders = make_global_test_loaders(
        test_data,
        num_classes=dataset_spec.num_classes,
        num_clients=args.num_clients,
        batch_size=args.eval_batch_size,
        test_limit=args.test_limit,
    )
    eval_loaders = {}
    if args.eval_scope in {"local", "both"}:
        eval_loaders["local"] = local_test_loaders
    if args.eval_scope in {"global", "both"}:
        eval_loaders["global"] = global_test_loaders

    model_assignments = [
        model_name_for_client(
            client_id=client_id,
            num_clients=args.num_clients,
            model_config=args.model_config,
        )
        for client_id in range(args.num_clients)
    ]
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
            model_name=model_assignments[client_id],
        )
        for client_id in range(args.num_clients)
    ]
    client_label_sets = [
        subset_label_set(subset, train_data)
        for subset in client_subsets
    ]
    client_class_counts = [
        class_histogram(subset, train_data, dataset_spec.num_classes)
        for subset in client_subsets
    ]

    print(f"Log file: {args.log_path}")
    print(f"Command: {' '.join(sys.argv)}")
    print()
    print("Local FL simulation")
    print("===================")
    print(f"Dataset: {dataset_spec.name}")
    print(f"Device: {device}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Model config: {args.model_config}")
    print(f"Partition: {args.partition}")
    print(f"Clients: {args.num_clients}")
    if args.partition == "beta":
        print(f"Samples per client: {args.samples_per_client}")
        print(f"Beta: {args.beta}")
    else:
        print(f"Ways: {args.ways}")
        print(f"Shots: {args.shots}")
        print(f"Stdev: {args.stdev}")
        print(f"Test shots per class: {args.test_shots_per_class}")
    print(f"Rounds: {args.rounds}")
    print(f"Training batch size: {args.batch_size}")
    print(f"Evaluation batch size: {args.eval_batch_size}")
    print(f"Evaluation scope: {args.eval_scope}")
    local_test_partition = "distribution_matched" if args.partition == "beta" else "kn_label_space"
    print(f"Local test partition: {local_test_partition}")
    model_groups: dict[str, list[int]] = {}
    for client in clients:
        model_groups.setdefault(client.model_name, []).append(client.client_id)
    print("Model groups:")
    for model_name, client_ids in model_groups.items():
        model = clients[client_ids[0]].model
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"  {model_name}: clients={client_ids} "
            f"parameters={parameter_count} prototype_dim={model.prototype_dim}"
        )
    if args.attack == "none":
        print("Attack: none")
    else:
        print(f"Attack: {args.attack}")
        print(f"Attack scale: {args.attack_scale}")
        if args.attack == "targeted_label_flip":
            print(
                "Targeted label flip: "
                f"source_class={args.flip_source_class} "
                f"target_class={args.flip_target_class}"
            )
        print(f"Malicious clients: {sorted(malicious_clients)}")
        print(f"Accuracy clients: {evaluation_clients}")
    if args.test_limit is not None:
        print(f"Per-client local test limit: {args.test_limit}")
    if args.algorithm == "prototype":
        print(f"Prototype loss weight: {args.proto_weight}")
        print(f"Prototype temperature: {args.proto_temperature}")
        print(f"Prototypes per class: {args.prototypes_per_class}")
        print(f"Minimum samples per prototype: {args.min_samples_per_prototype}")
        print(f"Prototype-guided synthesis: {args.prototype_synthesis}")
        if args.prototype_synthesis:
            print(f"Synthesis start round: {args.synthesis_start_round}")
            print(f"Synthesis rare-class threshold: {args.synthesis_target_count}")
            print(f"Synthetic samples per rare class: {args.synthetic_samples_per_class}")
            print(f"Synthesis optimization steps: {args.synthesis_steps}")
            print(f"Synthesis learning rate: {args.synthesis_lr}")
            print(f"Synthetic training weight: {args.synthesis_weight}")
            print(f"Synthesis minimum margin: {args.synthesis_min_margin}")
            print(f"Synthesis TV weight: {args.synthesis_tv_weight}")
        print(f"Backend: {args.backend}")
        if args.backend == "fabric":
            print(f"Fabric adapter: {args.fabric_adapter_url}")
            print(f"Fabric ledger round base: {args.fabric_round_base}")
            print(f"Prototype fixed-point scale: {args.prototype_scale}")
        print(f"Optimizer: {args.optimizer}")
    if args.algorithm == "trimmed_mean":
        print(f"Trim count: {args.trim_count}")
    if args.algorithm == "multi_krum":
        print(f"Multi-Krum f: {args.krum_f}")
    if args.algorithm == "fedprox":
        print(f"FedProx mu: {args.fedprox_mu}")
        print(f"Optimizer: {args.optimizer}")
    print()
    print("Client label histograms:")
    for client_id, subset in enumerate(client_subsets):
        print(f"  client {client_id}: {class_histogram(subset, train_data, dataset_spec.num_classes)}")
    print("Client local test label histograms:")
    for client_id, loader in enumerate(local_test_loaders):
        print(f"  client {client_id}: {class_histogram(loader.dataset, test_data, dataset_spec.num_classes)}")
    if args.eval_scope in {"global", "both"}:
        print("Global test label histogram:")
        print(f"  all clients: {class_histogram(global_test_loaders[0].dataset, test_data, dataset_spec.num_classes)}")

    if args.algorithm == "local":
        total_comm_bytes = run_local(args, clients, eval_loaders, evaluation_clients)
    elif args.algorithm == "prototype":
        total_comm_bytes = run_prototype(
            args,
            clients,
            eval_loaders,
            evaluation_clients,
            device,
            dataset_spec.num_classes,
            malicious_clients,
            client_label_sets,
            client_class_counts,
        )
    elif args.algorithm in {"fedavg", "trimmed_mean", "multi_krum"}:
        total_comm_bytes = run_fedavg(args, clients, eval_loaders, evaluation_clients, malicious_clients)
    elif args.algorithm == "fedprox":
        total_comm_bytes = run_fedprox(args, clients, eval_loaders, evaluation_clients, malicious_clients)
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
    load_runtime_dependencies()
    args.log_path = make_log_path(args)
    with redirect_output_to_log(args.log_path):
        run(args)


if __name__ == "__main__":
    main()
