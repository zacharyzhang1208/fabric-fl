"""Dataset loading and non-IID client partitioning utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

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


def make_noniid_client_subsets(
    dataset,
    num_classes: int,
    num_clients: int,
    samples_per_client: int,
    classes_per_client: int,
    seed: int,
) -> list[Subset]:
    rng = random.Random(seed)
    labels = dataset_labels(dataset)
    buckets = {label: [] for label in range(num_classes)}
    for idx, label in enumerate(labels):
        buckets[int(label)].append(idx)
    for indices in buckets.values():
        rng.shuffle(indices)

    subsets: list[Subset] = []
    pointers = {label: 0 for label in range(num_classes)}
    per_class = max(1, samples_per_client // classes_per_client)

    for client_id in range(num_clients):
        primary_labels = [
            (client_id * classes_per_client + offset) % num_classes
            for offset in range(classes_per_client)
        ]
        chosen: list[int] = []
        for label in primary_labels:
            start = pointers[label]
            end = min(start + per_class, len(buckets[label]))
            chosen.extend(buckets[label][start:end])
            pointers[label] = end

        while len(chosen) < samples_per_client:
            label = rng.randrange(num_classes)
            if pointers[label] < len(buckets[label]):
                chosen.append(buckets[label][pointers[label]])
                pointers[label] += 1

        rng.shuffle(chosen)
        subsets.append(Subset(dataset, chosen[:samples_per_client]))
    return subsets


def make_iid_client_subsets(
    dataset,
    num_classes: int,
    num_clients: int,
    samples_per_client: int,
    seed: int,
) -> list[Subset]:
    rng = random.Random(seed)
    labels = dataset_labels(dataset)
    buckets = {label: [] for label in range(num_classes)}
    for idx, label in enumerate(labels):
        buckets[int(label)].append(idx)
    for indices in buckets.values():
        rng.shuffle(indices)

    chosen = [[] for _ in range(num_clients)]
    pointers = {label: 0 for label in range(num_classes)}
    base_per_class = samples_per_client // num_classes
    remainder = samples_per_client % num_classes

    for client_id in range(num_clients):
        labels_for_extra = list(range(num_classes))
        rng.shuffle(labels_for_extra)
        targets = {label: base_per_class for label in range(num_classes)}
        for label in labels_for_extra[:remainder]:
            targets[label] += 1

        for label, target_count in targets.items():
            start = pointers[label]
            end = min(start + target_count, len(buckets[label]))
            chosen[client_id].extend(buckets[label][start:end])
            pointers[label] = end

    leftovers = [idx for indices in buckets.values() for idx in indices]
    rng.shuffle(leftovers)
    pointer = 0
    for client_id in range(num_clients):
        seen = set(chosen[client_id])
        while len(chosen[client_id]) < samples_per_client and pointer < len(leftovers):
            idx = leftovers[pointer]
            pointer += 1
            if idx not in seen:
                chosen[client_id].append(idx)
                seen.add(idx)

    subsets: list[Subset] = []
    for indices in chosen:
        rng.shuffle(indices)
        subsets.append(Subset(dataset, indices[:samples_per_client]))
    return subsets


def make_client_loaders(
    subsets: list[Subset],
    batch_size: int,
) -> tuple[list[DataLoader], list[DataLoader]]:
    train_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True)
        for subset in subsets
    ]
    prototype_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=False)
        for subset in subsets
    ]
    return train_loaders, prototype_loaders


def make_test_loader(dataset, batch_size: int, test_limit: int | None = None) -> DataLoader:
    if test_limit is not None:
        dataset = Subset(dataset, list(range(min(test_limit, len(dataset)))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def class_histogram(subset: Subset, dataset, num_classes: int) -> list[int]:
    counts = [0 for _ in range(num_classes)]
    labels = dataset_labels(dataset)
    for idx in subset.indices:
        counts[int(labels[idx])] += 1
    return counts
