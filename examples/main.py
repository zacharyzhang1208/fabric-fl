#!/usr/bin/env python3
"""Local multi-client FL simulation on image datasets.

This is step 1 before wiring the same payloads into Fabric PDC:

1. Split the selected dataset into non-IID client datasets.
2. Run either prototype distillation or standard FedAvg.
3. Optionally compress prototype payloads when running prototype mode.
4. Report round accuracy and communication bytes for comparison.

Run:
    python3 examples/main.py

Dependencies:
    pip install -r examples/requirements.txt
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import random
import struct
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
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    print(f"Missing dependency: {missing}", file=sys.stderr)
    print("Install demo dependencies with:", file=sys.stderr)
    print("  python3 -m venv .venv", file=sys.stderr)
    print("  source .venv/bin/activate", file=sys.stderr)
    print("  python -m pip install -r examples/requirements.txt", file=sys.stderr)
    raise SystemExit(1) from exc


class PrototypeCompressor:
    """Small compressor abstraction for the bytes that later go into Fabric PDC."""

    def __init__(self, method: str) -> None:
        if method not in {"fp32", "fp16", "int8"}:
            raise ValueError(f"Unsupported compression method: {method}")
        self.method = method

    def compress(self, prototypes: torch.Tensor, counts: torch.Tensor) -> bytes:
        prototypes = prototypes.detach().cpu()
        counts = counts.detach().cpu().to(torch.int32)
        raw_bytes = prototypes.numel() * 4 + counts.numel() * 4
        header = struct.pack(
            "<4sBIII",
            b"PDC1",
            {"fp32": 1, "fp16": 2, "int8": 3}[self.method],
            prototypes.shape[0],
            prototypes.shape[1],
            raw_bytes,
        )
        counts_bytes = counts.contiguous().numpy().tobytes()

        if self.method == "fp32":
            proto_bytes = prototypes.float().contiguous().numpy().tobytes()
            return header + counts_bytes + proto_bytes
        elif self.method == "fp16":
            proto_bytes = prototypes.half().contiguous().numpy().tobytes()
            return header + counts_bytes + proto_bytes

        max_abs = prototypes.abs().max().clamp_min(1e-8)
        scale = max_abs / 127.0
        quantized = torch.round(prototypes / scale).clamp(-127, 127).to(torch.int8)
        return header + counts_bytes + struct.pack("<f", float(scale)) + quantized.contiguous().numpy().tobytes()

    def decompress(self, payload_bytes: bytes, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
        header_size = struct.calcsize("<4sBIII")
        magic, method_id, rows, cols, raw_bytes = struct.unpack("<4sBIII", payload_bytes[:header_size])
        if magic != b"PDC1":
            raise ValueError("Invalid prototype payload")

        offset = header_size
        counts_size = rows * 4
        counts_buffer = bytearray(payload_bytes[offset : offset + counts_size])
        counts = torch.frombuffer(counts_buffer, dtype=torch.int32).to(device=device, dtype=torch.float32)
        offset += counts_size

        numel = rows * cols
        if method_id == 1:
            proto_buffer = bytearray(payload_bytes[offset : offset + numel * 4])
            prototypes = torch.frombuffer(proto_buffer, dtype=torch.float32).reshape(rows, cols)
        elif method_id == 2:
            proto_buffer = bytearray(payload_bytes[offset : offset + numel * 2])
            prototypes = torch.frombuffer(proto_buffer, dtype=torch.float16).reshape(rows, cols).float()
        elif method_id == 3:
            scale = struct.unpack("<f", payload_bytes[offset : offset + 4])[0]
            offset += 4
            proto_buffer = bytearray(payload_bytes[offset : offset + numel])
            prototypes = torch.frombuffer(proto_buffer, dtype=torch.int8).reshape(rows, cols).float() * scale
        else:
            raise ValueError(f"Unknown payload method id: {method_id}")

        prototypes = prototypes.to(device=device, dtype=torch.float32)
        return prototypes, counts, raw_bytes


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


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


def aggregate_prototype_subspaces(
    payloads: list[ClientUpdate],
    device: torch.device,
    num_classes: int,
    num_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not payloads:
        raise ValueError("No client payloads to aggregate")

    embed_dim = payloads[0].prototypes.shape[1]
    global_prototypes, global_counts = aggregate_prototypes(payloads, device, num_classes)
    bases = torch.zeros(num_classes, num_components, embed_dim, device=device)
    component_counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    if num_components <= 0:
        return global_prototypes, bases, component_counts

    for label in range(num_classes):
        class_vectors = []
        class_counts = []
        for payload in payloads:
            count = payload.counts[label].item()
            if count > 0:
                class_vectors.append(payload.prototypes[label].to(device=device, dtype=torch.float32))
                class_counts.append(float(count))

        if len(class_vectors) < 2:
            continue

        vectors = torch.stack(class_vectors)
        weights = torch.tensor(class_counts, device=device, dtype=torch.float32)
        weights = weights / weights.sum()
        centered = vectors - global_prototypes[label].unsqueeze(0)
        weighted_centered = centered * weights.sqrt().unsqueeze(1)
        _, _, vh = torch.linalg.svd(weighted_centered, full_matrices=False)
        components = min(num_components, vh.shape[0], len(class_vectors) - 1)
        if components > 0:
            bases[label, :components] = vh[:components]
            component_counts[label] = components

    return global_prototypes, bases, component_counts


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
    parser.add_argument("--algorithm", choices=["prototype", "prototype_pca", "fedavg"], default="prototype")
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
    parser.add_argument("--subspace-weight", type=float, default=0.2)
    parser.add_argument("--pca-components", type=int, default=2)
    parser.add_argument("--pca-history", type=int, default=5, help="Rounds of prototype history used by prototype_pca")
    parser.add_argument("--compression", choices=["none", "fp32", "fp16", "int8"], default="none")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-dir", default="log")
    return parser.parse_args()


def make_log_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.mode == "task_heter":
        mode_text = f"task_heter_w{args.ways}_s{args.shots}_sd{args.stdev}"
    else:
        mode_text = f"dirichlet_a{args.dirichlet_alpha}_samples{args.samples_per_client}"
    filename = f"{timestamp}_{args.dataset}_{args.algorithm}_{mode_text}_clients{args.num_clients}_rounds{args.rounds}.log"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / filename


def run(args: argparse.Namespace) -> None:
    if args.pca_components < 0:
        raise ValueError("--pca-components must be non-negative")
    if args.pca_history <= 0:
        raise ValueError("--pca-history must be positive")
    if args.dirichlet_alpha <= 0:
        raise ValueError("--dirichlet-alpha must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compressor = None if args.compression == "none" else PrototypeCompressor(args.compression)

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
            fedproto_reference=args.algorithm == "prototype",
        )
        for client_id in range(args.num_clients)
    ]
    global_prototypes: torch.Tensor | None = None
    global_counts: torch.Tensor | None = None
    global_bases: torch.Tensor | None = None
    prototype_history: list[ClientUpdate] = []
    total_wire_bytes = 0
    total_raw_bytes = 0

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
    if args.algorithm in {"prototype", "prototype_pca"}:
        print(f"Prototype payload: {args.compression}")
        print(f"Prototype loss weight: {args.proto_weight}")
        print(f"Optimizer: {args.optimizer}")
    if args.algorithm == "prototype_pca":
        print(f"Subspace loss weight: {args.subspace_weight}")
        print(f"PCA components: {args.pca_components}")
        print(f"PCA history rounds: {args.pca_history}")
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
        round_wire_bytes = 0
        round_raw_bytes = 0

        if args.algorithm in {"prototype", "prototype_pca"}:
            payloads: list[ClientUpdate] = []

            for client in clients:
                metrics = client.train_round(
                    local_epochs=args.local_epochs,
                    global_prototypes=global_prototypes,
                    global_counts=global_counts,
                    proto_weight=args.proto_weight,
                    global_bases=global_bases,
                    subspace_weight=args.subspace_weight if args.algorithm == "prototype_pca" else 0.0,
                )
                payload = client.build_update(round_id=round_id, compressor=compressor)
                payloads.append(payload)
                round_wire_bytes += payload.compressed_bytes
                round_raw_bytes += payload.raw_bytes

                acc = client.evaluate(test_loaders[client.client_id])
                metric_text = f"loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f}"
                if args.algorithm == "prototype_pca":
                    metric_text += f" proto={metrics.proto_loss:.4f} subspace={metrics.subspace_loss:.4f}"
                print(
                    f"  client {client.client_id}: {metric_text} "
                    f"test_acc={acc * 100:5.2f}% payload={payload.compressed_bytes}B"
                )

            if args.algorithm == "prototype_pca":
                prototype_history.extend(payloads)
                max_history_updates = args.pca_history * args.num_clients
                prototype_history = prototype_history[-max_history_updates:]
                global_prototypes, global_bases, component_counts = aggregate_prototype_subspaces(
                    prototype_history,
                    device,
                    dataset_spec.num_classes,
                    args.pca_components,
                )
                global_counts = torch.ones(dataset_spec.num_classes, device=device)
            else:
                global_prototypes, global_counts = aggregate_prototypes(payloads, device, dataset_spec.num_classes)
            avg_acc = average_accuracy(clients, test_loaders)
            ratio = round_wire_bytes / round_raw_bytes if round_raw_bytes else 0.0
            subspace_text = ""
            if args.algorithm == "prototype_pca":
                active_classes = int((component_counts > 0).sum().item())
                active_components = int(component_counts.sum().item())
                subspace_text = f" pca_classes={active_classes} pca_components={active_components}"
            print(
                f"  aggregator: avg_acc={avg_acc * 100:5.2f}% "
                f"round_payload={round_wire_bytes}B raw={round_raw_bytes}B ratio={ratio:.3f}"
                f"{subspace_text}"
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
                round_wire_bytes += update.raw_bytes
                round_raw_bytes += update.raw_bytes
                acc = client.evaluate(test_loaders[client.client_id])
                print(
                    f"  client {client.client_id}: loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f} "
                    f"local_test_acc={acc * 100:5.2f}% payload={update.raw_bytes}B"
                )

            global_model_state = aggregate_model_updates(model_updates)
            for client in clients:
                client.load_model_state(global_model_state)
            avg_acc = average_accuracy(clients, test_loaders)
            print(
                f"  aggregator: global_acc={avg_acc * 100:5.2f}% "
                f"round_payload={round_wire_bytes}B raw={round_raw_bytes}B ratio=1.000"
            )

        total_wire_bytes += round_wire_bytes
        total_raw_bytes += round_raw_bytes

    total_ratio = total_wire_bytes / total_raw_bytes if total_raw_bytes else 0.0
    print("\nFinal communication summary")
    print("===========================")
    payload_name = "model" if args.algorithm == "fedavg" else "prototype"
    print(f"Raw {payload_name} bytes:        {total_raw_bytes}")
    print(f"Compressed/wire bytes:      {total_wire_bytes}")
    print(f"Compression ratio:          {total_ratio:.3f}")


def main() -> None:
    args = parse_args()
    args.log_path = make_log_path(args)
    with args.log_path.open("w", encoding="utf-8") as log_file:
        stdout = Tee(sys.stdout, log_file)
        stderr = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            run(args)


if __name__ == "__main__":
    main()
