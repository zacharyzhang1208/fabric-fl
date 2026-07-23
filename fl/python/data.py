"""Dataset loading and client partitioning utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    input_shape: tuple[int, int, int]


DATASET_SPECS = {
    "mnist": DatasetSpec(name="mnist", num_classes=10, input_shape=(1, 28, 28)),
    "cifar10": DatasetSpec(name="cifar10", num_classes=10, input_shape=(3, 32, 32)),
    "cifar100": DatasetSpec(name="cifar100", num_classes=100, input_shape=(3, 32, 32)),
}


DATASET_DOWNLOAD_URLS = {
    "mnist": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
    ],
    "cifar10": ["https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"],
    "cifar100": ["https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"],
}


DATASET_EXPECTED_PATHS = {
    "mnist": "data/MNIST/raw/*.gz",
    "cifar10": "data/cifar-10-batches-py/",
    "cifar100": "data/cifar-100-python/",
}


def load_image_dataset(dataset_name: str, data_dir: str):
    name = dataset_name.lower()
    if name not in DATASET_SPECS:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported dataset {dataset_name!r}. Choose one of: {supported}")

    if name == "mnist":
        dataset_cls = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    elif name == "cifar10":
        dataset_cls = datasets.CIFAR10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    elif name == "cifar100":
        dataset_cls = datasets.CIFAR100
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
    else:
        raise AssertionError(f"Unhandled dataset after validation: {name}")

    try:
        train_data = dataset_cls(
            root=str(Path(data_dir)),
            train=True,
            transform=transform,
            download=False,
        )
        test_data = dataset_cls(
            root=str(Path(data_dir)),
            train=False,
            transform=transform,
            download=False,
        )
    except RuntimeError as exc:
        raise FileNotFoundError(dataset_missing_message(name, data_dir)) from exc
    return train_data, test_data, DATASET_SPECS[name]


def dataset_missing_message(dataset_name: str, data_dir: str) -> str:
    expected = DATASET_EXPECTED_PATHS[dataset_name].replace("data/", f"{data_dir.rstrip('/')}/")
    urls = "\n  ".join(DATASET_DOWNLOAD_URLS[dataset_name])
    return (
        f"Dataset {dataset_name!r} was not found in {data_dir!r}.\n"
        f"Expected local path: {expected}\n"
        "This demo does not auto-download or auto-extract datasets. "
        "Please download/extract it manually.\n"
        f"Official URL(s):\n  {urls}"
    )


def dataset_labels(dataset) -> list[int]:
    targets = dataset.targets
    if hasattr(targets, "tolist"):
        return targets.tolist()
    return [int(label) for label in targets]


def make_kn_client_subsets(
    dataset,
    num_classes: int,
    num_clients: int,
    ways: int,
    shots: int,
    stdev: int,
    train_shots_max: int,
    seed: int,
) -> list[Subset]:
    if ways < 2 or ways > num_classes:
        raise ValueError(f"--ways must be between 2 and {num_classes}")
    if shots <= 0:
        raise ValueError("--shots must be positive")
    if stdev <= 1:
        raise ValueError("K/N partition requires --stdev greater than 1")
    if train_shots_max < shots + stdev - 2:
        raise ValueError("--train-shots-max is too small for the requested shots and stdev")

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    labels = np.array(dataset_labels(dataset))
    idxs = np.arange(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort(kind="stable")]
    sorted_idxs = idxs_labels[0, :]

    label_begin: dict[int, int] = {}
    for position, label in enumerate(idxs_labels[1, :]):
        label_begin.setdefault(int(label), position)

    n_low = max(2, ways - stdev)
    n_high = min(num_classes, ways + stdev + 1)
    k_low = shots - stdev + 1
    k_high = shots + stdev - 1
    n_list = np_rng.randint(n_low, n_high, num_clients)
    k_list = np_rng.randint(k_low, k_high, num_clients)

    subsets: list[Subset] = []
    for client_id in range(num_clients):
        classes = sorted(rng.sample(range(num_classes), int(n_list[client_id])))
        chosen: list[int] = []
        for label in classes:
            begin = client_id * train_shots_max + label_begin[label]
            end = begin + int(k_list[client_id])
            label_end = label_begin.get(label + 1, len(sorted_idxs))
            if end > label_end:
                raise ValueError(
                    "K/N partition does not have enough class samples; "
                    "reduce --num-clients or --train-shots-max"
                )
            chosen.extend(int(idx) for idx in sorted_idxs[begin:end])
        rng.shuffle(chosen)
        subsets.append(Subset(dataset, chosen))
    return subsets


def make_dirichlet_client_subsets(
    dataset,
    num_classes: int,
    num_clients: int,
    samples_per_client: int,
    alpha: float,
    seed: int,
) -> list[Subset]:
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    if num_clients <= 0:
        raise ValueError("--num-clients must be positive")
    if samples_per_client <= 0:
        raise ValueError("--samples-per-client must be positive")

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    labels = dataset_labels(dataset)
    buckets = {label: [] for label in range(num_classes)}
    for idx, label in enumerate(labels):
        buckets[int(label)].append(idx)
    for indices in buckets.values():
        rng.shuffle(indices)

    total_samples = num_clients * samples_per_client
    samples_per_class, remainder = divmod(total_samples, num_classes)
    class_targets = np.array(
        [
            samples_per_class + (1 if label < remainder else 0)
            for label in range(num_classes)
        ],
        dtype=np.int64,
    )
    unavailable = [
        (label, int(class_targets[label]), len(buckets[label]))
        for label in range(num_classes)
        if len(buckets[label]) < class_targets[label]
    ]
    if unavailable:
        details = ", ".join(
            f"class {label}: need {needed}, have {available}"
            for label, needed, available in unavailable
        )
        raise ValueError(
            "Balanced Dirichlet sample pool exceeds dataset capacity; "
            f"reduce --num-clients or --samples-per-client ({details})"
        )

    concentration = np.full(num_classes, float(alpha), dtype=np.float64)
    preferences = np_rng.dirichlet(concentration, size=num_clients)
    remaining = class_targets.copy()
    allocations = np.zeros((num_clients, num_classes), dtype=np.int64)

    # Fill clients in shuffled round-robin order. Remaining class capacity is
    # included in the sampling weight so no client is systematically left with
    # the final labels while row and column totals remain exact.
    for _ in range(samples_per_client):
        for client_id in np_rng.permutation(num_clients):
            available = remaining > 0
            weights = preferences[client_id] * available
            capacity_fraction = np.divide(
                remaining,
                class_targets,
                out=np.zeros(num_classes, dtype=np.float64),
                where=class_targets > 0,
            )
            weights *= capacity_fraction
            weight_sum = weights.sum()
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                weights = remaining.astype(np.float64)
                weight_sum = weights.sum()
            probabilities = weights / weight_sum
            label = int(np_rng.choice(num_classes, p=probabilities))
            allocations[client_id, label] += 1
            remaining[label] -= 1

    pointers = np.zeros(num_classes, dtype=np.int64)
    subsets: list[Subset] = []
    for client_id in range(num_clients):
        chosen: list[int] = []
        for label in range(num_classes):
            count = int(allocations[client_id, label])
            begin = int(pointers[label])
            end = begin + count
            chosen.extend(buckets[label][begin:end])
            pointers[label] = end
        rng.shuffle(chosen)
        subsets.append(Subset(dataset, chosen))
    return subsets


def make_client_loaders(
    subsets: list[Subset],
    batch_size: int,
) -> tuple[list[DataLoader], list[DataLoader]]:
    train_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
        for subset in subsets
    ]
    prototype_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=False)
        for subset in subsets
    ]
    return train_loaders, prototype_loaders


def subset_label_set(subset: Subset, dataset) -> set[int]:
    labels = dataset_labels(dataset)
    return {int(labels[idx]) for idx in subset.indices}


def make_client_test_loaders(
    train_subsets: list[Subset],
    train_dataset,
    test_dataset,
    batch_size: int,
    test_limit: int | None = None,
    seed: int = 0,
) -> list[DataLoader]:
    if test_limit is not None and test_limit <= 0:
        raise ValueError("--test-limit must be positive")

    train_labels = dataset_labels(train_dataset)
    test_labels = dataset_labels(test_dataset)
    num_classes = len(set(train_labels))
    test_buckets = {label: [] for label in range(num_classes)}
    for idx, label in enumerate(test_labels):
        test_buckets[int(label)].append(idx)

    loaders: list[DataLoader] = []
    for client_id, train_subset in enumerate(train_subsets):
        rng = random.Random(seed + client_id)
        shuffled_buckets = {label: list(indices) for label, indices in test_buckets.items()}
        for indices in shuffled_buckets.values():
            rng.shuffle(indices)

        train_counts = [0 for _ in range(num_classes)]
        for idx in train_subset.indices:
            train_counts[int(train_labels[idx])] += 1

        target_size = test_limit if test_limit is not None else len(train_subset)
        caps = [len(shuffled_buckets[label]) for label in range(num_classes)]
        quotas = distribution_matched_quotas(train_counts, caps, target_size)

        indices = []
        for label, quota in enumerate(quotas):
            indices.extend(shuffled_buckets[label][:quota])
        rng.shuffle(indices)
        loaders.append(DataLoader(Subset(test_dataset, indices), batch_size=batch_size, shuffle=False))
    return loaders


def make_kn_client_test_loaders(
    train_subsets: list[Subset],
    train_dataset,
    test_dataset,
    batch_size: int,
    test_shots_per_class: int,
    test_limit: int | None = None,
) -> list[DataLoader]:
    if test_shots_per_class <= 0:
        raise ValueError("--test-shots-per-class must be positive")
    if test_limit is not None and test_limit <= 0:
        raise ValueError("--test-limit must be positive")

    labels = np.array(dataset_labels(test_dataset))
    idxs = np.arange(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort(kind="stable")]
    sorted_idxs = idxs_labels[0, :]

    label_begin: dict[int, int] = {}
    for position, label in enumerate(idxs_labels[1, :]):
        label_begin.setdefault(int(label), position)

    loaders: list[DataLoader] = []
    for client_id, train_subset in enumerate(train_subsets):
        classes = sorted(subset_label_set(train_subset, train_dataset))
        chosen: list[int] = []
        for label in classes:
            begin = client_id * test_shots_per_class + label_begin[label]
            end = begin + test_shots_per_class
            label_end = label_begin.get(label + 1, len(sorted_idxs))
            if end > label_end:
                raise ValueError(
                    "K/N test partition does not have enough class samples; "
                    "reduce --num-clients or --test-shots-per-class"
                )
            chosen.extend(int(idx) for idx in sorted_idxs[begin:end])
        if test_limit is not None:
            chosen = chosen[:test_limit]
        loaders.append(DataLoader(Subset(test_dataset, chosen), batch_size=batch_size, shuffle=False))
    return loaders


def distribution_matched_quotas(
    train_counts: list[int],
    caps: list[int],
    target_size: int,
) -> list[int]:
    if len(train_counts) != len(caps):
        raise ValueError("train_counts and caps must have the same length")
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    eligible = [
        label
        for label, count in enumerate(train_counts)
        if count > 0 and caps[label] > 0
    ]
    if not eligible:
        raise ValueError("No matching test labels for client training distribution")

    target_size = min(target_size, sum(caps[label] for label in eligible))
    quotas = [0 for _ in train_counts]
    remaining = target_size
    active = set(eligible)

    while active and remaining > 0:
        weight_sum = sum(train_counts[label] for label in active)
        planned: dict[int, int] = {}
        remainders: list[tuple[float, int]] = []
        planned_total = 0

        for label in active:
            exact = remaining * train_counts[label] / weight_sum
            quota = min(caps[label] - quotas[label], int(exact))
            planned[label] = quota
            planned_total += quota
            remainders.append((exact - int(exact), label))

        leftover = remaining - planned_total
        for _, label in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if leftover <= 0:
                break
            if planned[label] < caps[label] - quotas[label]:
                planned[label] += 1
                leftover -= 1

        progress = 0
        for label, quota in planned.items():
            quotas[label] += quota
            progress += quota

        if progress == 0:
            break
        remaining -= progress
        active = {
            label
            for label in active
            if quotas[label] < caps[label]
        }

    return quotas


def make_global_test_loaders(
    test_dataset,
    num_classes: int,
    num_clients: int,
    batch_size: int,
    test_limit: int | None = None,
) -> list[DataLoader]:
    if test_limit is None:
        indices = list(range(len(test_dataset)))
    else:
        if test_limit < num_classes:
            raise ValueError("--test-limit must be at least the number of classes for global evaluation")
        labels = dataset_labels(test_dataset)
        buckets = {label: [] for label in range(num_classes)}
        for idx, label in enumerate(labels):
            buckets[int(label)].append(idx)
        per_class = test_limit // num_classes
        remainder = test_limit % num_classes
        indices = []
        for label in range(num_classes):
            quota = per_class + (1 if label < remainder else 0)
            indices.extend(buckets[label][:quota])
    subset = Subset(test_dataset, indices)
    return [
        DataLoader(subset, batch_size=batch_size, shuffle=False)
        for _ in range(num_clients)
    ]




def class_histogram(subset: Subset, dataset, num_classes: int) -> list[int]:
    counts = [0 for _ in range(num_classes)]
    labels = dataset_labels(dataset)
    for idx in subset.indices:
        counts[int(labels[idx])] += 1
    return counts
