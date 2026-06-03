#!/usr/bin/env python3
"""Local multi-client FL simulation on image datasets.

This is step 1 before wiring the same payloads into Fabric PDC:

1. Split the selected dataset into non-IID client datasets.
2. Run either prototype distillation or standard FedAvg.
3. Compress prototype payloads when running prototype mode.
4. Report round accuracy and communication bytes for comparison.

Run:
    python3 examples/main.py

Dependencies:
    pip install -r examples/requirements.txt
"""

from __future__ import annotations

import argparse
import random
import struct
import sys

try:
    import torch
    from data import (
        DATASET_SPECS,
        class_histogram,
        load_image_dataset,
        make_client_loaders,
        make_iid_client_subsets,
        make_noniid_client_subsets,
        make_test_loader,
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def aggregate_prototypes(
    payloads: list[ClientUpdate],
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    if not payloads:
        raise ValueError("No client payloads to aggregate")

    embed_dim = payloads[0].prototypes.shape[1]
    sums = torch.zeros(num_classes, embed_dim, device=device)
    counts = torch.zeros(num_classes, device=device)

    for payload in payloads:
        client_counts = payload.counts.to(device)
        sums += payload.prototypes.to(device) * client_counts.unsqueeze(1)
        counts += client_counts

    global_prototypes = torch.zeros_like(sums)
    present = counts > 0
    global_prototypes[present] = sums[present] / counts[present].unsqueeze(1)
    return global_prototypes


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


def average_accuracy(clients: list[FederatedClient], loader) -> float:
    return sum(client.evaluate(loader) for client in clients) / len(clients)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local multi-client FL simulation")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="mnist")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--mode", choices=["prototype", "fedavg"], default="prototype")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--samples-per-client", type=int, default=600)
    parser.add_argument("--partition", choices=["iid", "noniid"], default="iid")
    parser.add_argument("--classes-per-client", type=int, default=2, help="Only used when --partition noniid")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--proto-weight", type=float, default=0.2)
    parser.add_argument("--compression", choices=["fp32", "fp16", "int8"], default="int8")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compressor = PrototypeCompressor(args.compression)

    try:
        train_data, test_data, dataset_spec = load_image_dataset(
            args.dataset,
            args.data_dir,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    if args.partition == "iid":
        client_subsets = make_iid_client_subsets(
            train_data,
            num_classes=dataset_spec.num_classes,
            num_clients=args.num_clients,
            samples_per_client=args.samples_per_client,
            seed=args.seed + 1,
        )
    else:
        client_subsets = make_noniid_client_subsets(
            train_data,
            num_classes=dataset_spec.num_classes,
            num_clients=args.num_clients,
            samples_per_client=args.samples_per_client,
            classes_per_client=args.classes_per_client,
            seed=args.seed + 1,
        )
    client_loaders, proto_loaders = make_client_loaders(client_subsets, args.batch_size)
    test_loader = make_test_loader(test_data, args.batch_size, args.test_limit)

    clients = [
        FederatedClient(
            client_id=client_id,
            train_loader=client_loaders[client_id],
            prototype_loader=proto_loaders[client_id],
            device=device,
            lr=args.lr,
            input_shape=dataset_spec.input_shape,
            num_classes=dataset_spec.num_classes,
        )
        for client_id in range(args.num_clients)
    ]
    global_prototypes: torch.Tensor | None = None
    total_wire_bytes = 0
    total_raw_bytes = 0

    print("Local FL simulation")
    print("===================")
    print(f"Dataset: {dataset_spec.name}")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Clients: {args.num_clients}")
    print(f"Partition: {args.partition}")
    print(f"Rounds: {args.rounds}")
    if args.test_limit is not None:
        print(f"Test limit: {args.test_limit}")
    if args.mode == "prototype":
        print(f"Compression: {args.compression}")
        print(f"Prototype loss weight: {args.proto_weight}")
    print()
    print("Client label histograms:")
    for client_id, subset in enumerate(client_subsets):
        print(f"  client {client_id}: {class_histogram(subset, train_data, dataset_spec.num_classes)}")

    if args.mode == "fedavg":
        global_model_state = clients[0].get_model_state()
        for client in clients:
            client.load_model_state(global_model_state)

    for round_id in range(1, args.rounds + 1):
        print(f"\nRound {round_id}")
        round_wire_bytes = 0
        round_raw_bytes = 0

        if args.mode == "prototype":
            payloads: list[ClientUpdate] = []

            for client in clients:
                metrics = client.train_round(
                    local_epochs=args.local_epochs,
                    global_prototypes=global_prototypes,
                    proto_weight=args.proto_weight,
                )
                payload = client.build_update(round_id=round_id, compressor=compressor)
                payloads.append(payload)
                round_wire_bytes += payload.compressed_bytes
                round_raw_bytes += payload.raw_bytes

                acc = client.evaluate(test_loader)
                print(
                    f"  client {client.client_id}: loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f} "
                    f"test_acc={acc * 100:5.2f}% payload={payload.compressed_bytes}B"
                )

            global_prototypes = aggregate_prototypes(payloads, device, dataset_spec.num_classes)
            avg_acc = average_accuracy(clients, test_loader)
            ratio = round_wire_bytes / round_raw_bytes if round_raw_bytes else 0.0
            print(
                f"  aggregator: avg_acc={avg_acc * 100:5.2f}% "
                f"round_payload={round_wire_bytes}B raw={round_raw_bytes}B ratio={ratio:.3f}"
            )
        else:
            model_updates: list[ModelUpdate] = []
            for client in clients:
                client.load_model_state(global_model_state)
                metrics = client.train_round(
                    local_epochs=args.local_epochs,
                    global_prototypes=None,
                    proto_weight=0.0,
                )
                update = client.build_model_update(round_id=round_id)
                model_updates.append(update)
                round_wire_bytes += update.raw_bytes
                round_raw_bytes += update.raw_bytes
                acc = client.evaluate(test_loader)
                print(
                    f"  client {client.client_id}: loss={metrics.loss:.4f} ce={metrics.ce_loss:.4f} "
                    f"local_test_acc={acc * 100:5.2f}% payload={update.raw_bytes}B"
                )

            global_model_state = aggregate_model_updates(model_updates)
            for client in clients:
                client.load_model_state(global_model_state)
            avg_acc = average_accuracy(clients, test_loader)
            print(
                f"  aggregator: global_acc={avg_acc * 100:5.2f}% "
                f"round_payload={round_wire_bytes}B raw={round_raw_bytes}B ratio=1.000"
            )

        total_wire_bytes += round_wire_bytes
        total_raw_bytes += round_raw_bytes

    total_ratio = total_wire_bytes / total_raw_bytes if total_raw_bytes else 0.0
    print("\nFinal communication summary")
    print("===========================")
    payload_name = "prototype" if args.mode == "prototype" else "model"
    print(f"Raw {payload_name} bytes:        {total_raw_bytes}")
    print(f"Compressed/wire bytes:      {total_wire_bytes}")
    print(f"Compression ratio:          {total_ratio:.3f}")
    if args.mode == "prototype":
        print("These bytes are the local stand-in for the future Fabric PDC payloads.")


if __name__ == "__main__":
    main()
