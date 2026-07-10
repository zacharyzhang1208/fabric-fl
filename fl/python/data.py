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
    if shots - stdev + 1 >= shots + stdev - 1:
        raise ValueError("K/N sampling requires --stdev greater than 1")

    random.seed(seed)
    np.random.seed(seed)
    labels = np.array(dataset_labels(dataset))
    idxs = np.arange(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    sorted_idxs = idxs_labels[0, :]

    label_begin: dict[int, int] = {}
    for position, label in enumerate(idxs_labels[1, :]):
        label_begin.setdefault(int(label), position)

    n_low = max(2, ways - stdev)
    n_high = min(num_classes, ways + stdev + 1)
    n_list = np.random.randint(n_low, n_high, num_clients)
    k_list = np.random.randint(shots - stdev + 1, shots + stdev - 1, num_clients)

    subsets: list[Subset] = []
    for client_id in range(num_clients):
        classes = sorted(random.sample(range(num_classes), int(n_list[client_id])))
        chosen: list[int] = []
        for label in classes:
            begin = client_id * train_shots_max + label_begin[label]
            end = begin + int(k_list[client_id])
            chosen.extend(int(idx) for idx in sorted_idxs[begin:end])
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

    rng = random.Random(seed)
    torch.manual_seed(seed)
    labels = dataset_labels(dataset)
    buckets = {label: [] for label in range(num_classes)}
    for idx, label in enumerate(labels):
        buckets[int(label)].append(idx)
    for indices in buckets.values():
        rng.shuffle(indices)

    pointers = {label: 0 for label in range(num_classes)}
    concentration = torch.full((num_classes,), float(alpha), dtype=torch.float32)
    subsets: list[Subset] = []

    for _ in range(num_clients):
        class_probs = torch.distributions.Dirichlet(concentration).sample()
        chosen: list[int] = []

        while len(chosen) < samples_per_client:
            available_labels = [
                label
                for label in range(num_classes)
                if pointers[label] < len(buckets[label])
            ]
            if not available_labels:
                break

            masked_probs = torch.zeros(num_classes, dtype=torch.float32)
            masked_probs[available_labels] = class_probs[available_labels]
            if masked_probs.sum() <= 0:
                masked_probs[available_labels] = 1.0
            masked_probs = masked_probs / masked_probs.sum()
            label = int(torch.multinomial(masked_probs, 1).item())
            chosen.append(buckets[label][pointers[label]])
            pointers[label] += 1

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
) -> list[DataLoader]:
    test_labels = dataset_labels(test_dataset)
    loaders: list[DataLoader] = []
    for train_subset in train_subsets:
        allowed_labels = subset_label_set(train_subset, train_dataset)
        indices = [
            idx
            for idx, label in enumerate(test_labels)
            if int(label) in allowed_labels
        ]
        if test_limit is not None:
            indices = indices[:test_limit]
        loaders.append(DataLoader(Subset(test_dataset, indices), batch_size=batch_size, shuffle=False))
    return loaders


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


def make_kn_client_test_loaders(
    train_subsets: list[Subset],
    train_dataset,
    test_dataset,
    batch_size: int,
    test_shots_per_class: int,
    test_limit: int | None = None,
) -> list[DataLoader]:
    labels = np.array(dataset_labels(test_dataset))
    idxs = np.arange(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
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
            chosen.extend(int(idx) for idx in sorted_idxs[begin:end])
        if test_limit is not None:
            chosen = chosen[:test_limit]
        loaders.append(DataLoader(Subset(test_dataset, chosen), batch_size=batch_size, shuffle=False))
    return loaders


def class_histogram(subset: Subset, dataset, num_classes: int) -> list[int]:
    counts = [0 for _ in range(num_classes)]
    labels = dataset_labels(dataset)
    for idx in subset.indices:
        counts[int(labels[idx])] += 1
    return counts
