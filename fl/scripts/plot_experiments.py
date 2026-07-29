#!/usr/bin/env python3
"""Generate publication-ready charts from experiment manifests and logs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
from typing import Iterable


ACCURACY_PATTERN = re.compile(
    r"(?<![A-Za-z_])(?:benign_)?avg_acc=\s*([0-9]+(?:\.[0-9]+)?)%"
)
COMMUNICATION_PATTERN = re.compile(r"communication:\s+round=(\d+)\s+B")
ALGORITHM_ORDER = (
    "local",
    "fedavg",
    "fedprox",
    "trimmed_mean",
    "multi_krum",
    "prototype",
)
ALGORITHM_LABELS = {
    "local": "Local",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "trimmed_mean": "Trimmed Mean",
    "multi_krum": "Multi-Krum",
    "prototype": "FedProto",
}
ALGORITHM_COLORS = {
    "local": "#666666",
    "fedavg": "#377eb8",
    "fedprox": "#4daf4a",
    "trimmed_mean": "#984ea3",
    "multi_krum": "#a65628",
    "prototype": "#e64b35",
}
ALGORITHM_MARKERS = {
    "local": "o",
    "fedavg": "s",
    "fedprox": "^",
    "trimmed_mean": "D",
    "multi_krum": "P",
    "prototype": "X",
}


@dataclass(frozen=True)
class ExperimentRun:
    experiment_dir: Path
    task_id: str
    dataset: str
    model_config: str
    partition: str
    partition_config: str
    beta: str
    samples_per_client: str
    ways: str
    shots: str
    stdev: str
    train_shots_max: str
    test_shots_per_class: str
    algorithm: str
    seed: str
    log_path: Path
    accuracies: tuple[float, ...]
    round_communication_bytes: tuple[int, ...]

    @property
    def last10_accuracy(self) -> float:
        return statistics.fmean(self.accuracies[-10:])

    @property
    def final_accuracy(self) -> float:
        return self.accuracies[-1]

    @property
    def best_accuracy(self) -> float:
        return max(self.accuracies)

    @property
    def total_communication_bytes(self) -> int:
        return sum(self.round_communication_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate accuracy, convergence, delta, and communication charts"
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        type=Path,
        help="Experiment directories containing manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Chart directory; defaults to EXPERIMENT_DIR/figures for one input",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if len(set(args.formats)) != len(args.formats):
        parser.error("--formats contains duplicate values")
    return args


def parse_log(path: Path) -> tuple[tuple[float, ...], tuple[int, ...]]:
    accuracies: list[float] = []
    communication: list[int] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "aggregator:" in line:
                match = ACCURACY_PATTERN.search(line)
                if match:
                    accuracies.append(float(match.group(1)))
            if "communication:" in line:
                match = COMMUNICATION_PATTERN.search(line)
                if match:
                    communication.append(int(match.group(1)))
    return tuple(accuracies), tuple(communication)


def resolve_log_path(experiment_dir: Path, task: dict[str, str]) -> Path | None:
    recorded = task.get("log_path", "")
    if recorded:
        path = Path(recorded).expanduser()
        if path.is_file():
            return path.resolve()

    run_dir = experiment_dir / "runs" / task["task_id"]
    logs = sorted(
        run_dir.glob("*.log"),
        key=lambda path: path.stat().st_mtime,
    )
    return logs[-1].resolve() if logs else None


def normalized_partition_fields(task: dict[str, str]) -> dict[str, str]:
    partition = task.get("partition") or "beta"
    beta = task.get("beta", "")
    samples = task.get("samples_per_client", "")
    partition_config = task.get("partition_config", "")
    if not partition_config:
        suffix = f"-samples-{samples}" if samples else ""
        partition_config = f"beta-{beta}{suffix}"
    return {
        "partition": partition,
        "partition_config": partition_config,
        "beta": beta,
        "samples_per_client": samples,
        "ways": task.get("ways", ""),
        "shots": task.get("shots", ""),
        "stdev": task.get("stdev", ""),
        "train_shots_max": task.get("train_shots_max", ""),
        "test_shots_per_class": task.get("test_shots_per_class", ""),
    }


def load_runs(experiment_dirs: Iterable[Path]) -> tuple[list[ExperimentRun], list[str]]:
    runs: list[ExperimentRun] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    datasets: set[str] = set()
    model_configs: set[str] = set()
    for raw_dir in experiment_dirs:
        experiment_dir = raw_dir.expanduser().resolve()
        manifest_path = experiment_dir / "manifest.csv"
        if not manifest_path.is_file():
            raise ValueError(f"Missing manifest: {manifest_path}")
        metadata_path = experiment_dir / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        dataset = str(metadata.get("dataset", ""))
        model_config = str(metadata.get("model_config", ""))
        if dataset:
            datasets.add(dataset)
        if model_config:
            model_configs.add(model_config)
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            tasks = list(csv.DictReader(handle))
        for task in tasks:
            if task.get("status") != "completed":
                continue
            fields = normalized_partition_fields(task)
            identity = (
                fields["partition_config"],
                task["algorithm"],
                task["seed"],
            )
            if identity in seen:
                warnings.append(
                    "Duplicate completed run skipped: "
                    f"{fields['partition_config']} {task['algorithm']} seed={task['seed']}"
                )
                continue
            log_path = resolve_log_path(experiment_dir, task)
            if log_path is None:
                warnings.append(f"Log not found for completed task {task['task_id']}")
                continue
            accuracies, communication = parse_log(log_path)
            if not accuracies:
                warnings.append(f"No aggregator accuracy in {log_path}")
                continue
            runs.append(
                ExperimentRun(
                    experiment_dir=experiment_dir,
                    task_id=task["task_id"],
                    dataset=dataset,
                    model_config=model_config,
                    algorithm=task["algorithm"],
                    seed=task["seed"],
                    log_path=log_path,
                    accuracies=accuracies,
                    round_communication_bytes=communication,
                    **fields,
                )
            )
            seen.add(identity)
    if len(datasets) > 1:
        raise ValueError(
            "Cannot combine different datasets in one chart set: "
            + ", ".join(sorted(datasets))
        )
    if len(model_configs) > 1:
        raise ValueError(
            "Cannot combine different model configurations in one chart set: "
            + ", ".join(sorted(model_configs))
        )
    if not runs:
        raise ValueError("No completed runs with readable accuracy logs were found")
    return runs, warnings


def algorithm_sort_key(algorithm: str) -> tuple[int, str]:
    try:
        return ALGORITHM_ORDER.index(algorithm), algorithm
    except ValueError:
        return len(ALGORITHM_ORDER), algorithm


def mean_stdev(values: Iterable[float]) -> tuple[float, float]:
    materialized = list(values)
    if not materialized:
        return float("nan"), 0.0
    return (
        statistics.fmean(materialized),
        statistics.stdev(materialized) if len(materialized) > 1 else 0.0,
    )


def group_runs(
    runs: Iterable[ExperimentRun],
) -> dict[tuple[str, str], list[ExperimentRun]]:
    groups: dict[tuple[str, str], list[ExperimentRun]] = {}
    for run in runs:
        groups.setdefault((run.partition_config, run.algorithm), []).append(run)
    return groups


def paired_deltas(runs: Iterable[ExperimentRun]) -> dict[tuple[str, str], list[float]]:
    materialized = list(runs)
    local = {
        (run.partition_config, run.seed): run.last10_accuracy
        for run in materialized
        if run.algorithm == "local"
    }
    deltas: dict[tuple[str, str], list[float]] = {}
    for run in materialized:
        baseline = local.get((run.partition_config, run.seed))
        if baseline is None:
            continue
        deltas.setdefault((run.partition_config, run.algorithm), []).append(
            run.last10_accuracy - baseline
        )
    return deltas


def write_plot_data(path: Path, runs: list[ExperimentRun]) -> None:
    fields = (
        "dataset",
        "model_config",
        "partition",
        "partition_config",
        "beta",
        "samples_per_client",
        "ways",
        "shots",
        "stdev",
        "train_shots_max",
        "test_shots_per_class",
        "algorithm",
        "seed",
        "rounds",
        "last10_avg_acc",
        "final_acc",
        "best_acc",
        "total_communication_bytes",
        "log_path",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(
            runs,
            key=lambda item: (
                item.partition_config,
                algorithm_sort_key(item.algorithm),
                item.seed,
            ),
        ):
            writer.writerow(
                {
                    "dataset": run.dataset,
                    "model_config": run.model_config,
                    "partition": run.partition,
                    "partition_config": run.partition_config,
                    "beta": run.beta,
                    "samples_per_client": run.samples_per_client,
                    "ways": run.ways,
                    "shots": run.shots,
                    "stdev": run.stdev,
                    "train_shots_max": run.train_shots_max,
                    "test_shots_per_class": run.test_shots_per_class,
                    "algorithm": run.algorithm,
                    "seed": run.seed,
                    "rounds": len(run.accuracies),
                    "last10_avg_acc": f"{run.last10_accuracy:.6f}",
                    "final_acc": f"{run.final_accuracy:.6f}",
                    "best_acc": f"{run.best_accuracy:.6f}",
                    "total_communication_bytes": run.total_communication_bytes,
                    "log_path": str(run.log_path),
                }
            )


def import_matplotlib():
    config_dir = Path(tempfile.gettempdir()) / "fabric-fl-matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Matplotlib is required. Install dependencies with: "
            "python -m pip install -r fl/python/requirements.txt"
        ) from exc
    return plt


def style_axis(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def bar_value_label(value: float, metric_name: str) -> str:
    if not math.isfinite(value):
        return ""
    if metric_name == "communication":
        return f"{value:.3f}" if abs(value) < 0.1 else f"{value:.2f}"
    return f"{value:.2f}"


def add_bar_labels(ax, bars, values: list[float], metric_name: str) -> None:
    ax.bar_label(
        bars,
        labels=[bar_value_label(value, metric_name) for value in values],
        padding=4,
        fontsize=8,
    )


def expand_y_axis_for_labels(ax) -> None:
    lower, upper = ax.get_ylim()
    span = upper - lower
    if span > 0:
        bottom = lower - span * 0.08 if lower < 0 else lower
        top = upper + span * 0.1 if upper > 0 else upper
        ax.set_ylim(bottom, top)


def save_figure(fig, output_stem: Path, formats: Iterable[str], dpi: int) -> list[Path]:
    paths = []
    fig.tight_layout()
    for extension in formats:
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths


def plot_categorical_metric(
    plt,
    runs: list[ExperimentRun],
    partition: str,
    metric_name: str,
    value_getter,
    ylabel: str,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    selected = [run for run in runs if run.partition == partition]
    configs = sorted({run.partition_config for run in selected})
    algorithms = sorted(
        {run.algorithm for run in selected},
        key=algorithm_sort_key,
    )
    groups = group_runs(selected)
    if not configs or not algorithms:
        return []

    width = min(0.8 / len(algorithms), 0.22)
    x_positions = list(range(len(configs)))
    fig, ax = plt.subplots(figsize=(max(6.4, len(configs) * 1.8), 4.4))
    for index, algorithm in enumerate(algorithms):
        means = []
        errors = []
        for config in configs:
            mean, stdev = mean_stdev(
                value_getter(run)
                for run in groups.get((config, algorithm), [])
            )
            means.append(mean)
            errors.append(stdev)
        offsets = [
            x + (index - (len(algorithms) - 1) / 2) * width
            for x in x_positions
        ]
        bars = ax.bar(
            offsets,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            color=ALGORITHM_COLORS.get(algorithm, "#333333"),
            label=ALGORITHM_LABELS.get(algorithm, algorithm),
        )
        add_bar_labels(ax, bars, means, metric_name)
    expand_y_axis_for_labels(ax)
    ax.set_xticks(x_positions, [config_label(config) for config in configs])
    ax.set_xlabel("Partition configuration")
    style_axis(ax, ylabel)
    ax.legend(frameon=False, ncol=min(3, len(algorithms)))
    return save_figure(
        fig,
        output_dir / f"{metric_name}_{partition}",
        formats,
        dpi,
    )


def plot_beta_metric(
    plt,
    runs: list[ExperimentRun],
    metric_name: str,
    value_getter,
    ylabel: str,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    selected = [run for run in runs if run.partition == "beta" and run.beta]
    beta_values = sorted({float(run.beta) for run in selected}, reverse=True)
    algorithms = sorted(
        {run.algorithm for run in selected},
        key=algorithm_sort_key,
    )
    if not beta_values or not algorithms:
        return []
    by_beta_algorithm: dict[tuple[float, str], list[ExperimentRun]] = {}
    for run in selected:
        by_beta_algorithm.setdefault((float(run.beta), run.algorithm), []).append(run)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for algorithm in algorithms:
        means = []
        errors = []
        for beta in beta_values:
            mean, stdev = mean_stdev(
                value_getter(run)
                for run in by_beta_algorithm.get((beta, algorithm), [])
            )
            means.append(mean)
            errors.append(stdev)
        ax.errorbar(
            beta_values,
            means,
            yerr=errors,
            marker=ALGORITHM_MARKERS.get(algorithm, "o"),
            markersize=5,
            linewidth=1.8,
            capsize=3,
            color=ALGORITHM_COLORS.get(algorithm, "#333333"),
            label=ALGORITHM_LABELS.get(algorithm, algorithm),
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(beta_values, [f"{value:g}" for value in beta_values])
    ax.minorticks_off()
    ax.set_xlabel("Dirichlet beta (more non-IID to the right)")
    style_axis(ax, ylabel)
    ax.legend(frameon=False, ncol=min(3, len(algorithms)))
    return save_figure(
        fig,
        output_dir / f"{metric_name}_beta",
        formats,
        dpi,
    )


def plot_delta(
    plt,
    runs: list[ExperimentRun],
    partition: str,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    selected = [run for run in runs if run.partition == partition]
    deltas = paired_deltas(selected)
    algorithms = sorted(
        {algorithm for _, algorithm in deltas if algorithm != "local"},
        key=algorithm_sort_key,
    )
    if not algorithms:
        return []

    if partition == "beta":
        beta_by_config = {
            run.partition_config: float(run.beta)
            for run in selected
            if run.beta
        }
        beta_values = sorted(set(beta_by_config.values()), reverse=True)
        fig, ax = plt.subplots(figsize=(6.8, 4.5))
        for algorithm in algorithms:
            means = []
            errors = []
            for beta in beta_values:
                values = [
                    value
                    for (config, candidate), group in deltas.items()
                    if candidate == algorithm and beta_by_config.get(config) == beta
                    for value in group
                ]
                mean, stdev = mean_stdev(values)
                means.append(mean)
                errors.append(stdev)
            ax.errorbar(
                beta_values,
                means,
                yerr=errors,
                marker=ALGORITHM_MARKERS.get(algorithm, "o"),
                linewidth=1.8,
                capsize=3,
                color=ALGORITHM_COLORS.get(algorithm, "#333333"),
                label=ALGORITHM_LABELS.get(algorithm, algorithm),
            )
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xticks(beta_values, [f"{value:g}" for value in beta_values])
        ax.minorticks_off()
        ax.set_xlabel("Dirichlet beta (more non-IID to the right)")
    else:
        configs = sorted({config for config, _ in deltas})
        width = min(0.8 / len(algorithms), 0.24)
        x_positions = list(range(len(configs)))
        fig, ax = plt.subplots(figsize=(max(6.4, len(configs) * 1.8), 4.4))
        for index, algorithm in enumerate(algorithms):
            means, errors = [], []
            for config in configs:
                mean, stdev = mean_stdev(deltas.get((config, algorithm), []))
                means.append(mean)
                errors.append(stdev)
            offsets = [
                x + (index - (len(algorithms) - 1) / 2) * width
                for x in x_positions
            ]
            bars = ax.bar(
                offsets,
                means,
                width=width,
                yerr=errors,
                capsize=3,
                color=ALGORITHM_COLORS.get(algorithm, "#333333"),
                label=ALGORITHM_LABELS.get(algorithm, algorithm),
            )
            add_bar_labels(ax, bars, means, "delta")
        expand_y_axis_for_labels(ax)
        ax.set_xticks(x_positions, [config_label(config) for config in configs])
        ax.set_xlabel("Partition configuration")
    ax.axhline(0, color="#222222", linewidth=0.8)
    style_axis(ax, "Last-10 accuracy delta vs Local (percentage points)")
    ax.legend(frameon=False, ncol=min(3, len(algorithms)))
    return save_figure(fig, output_dir / f"delta_vs_local_{partition}", formats, dpi)


def plot_convergence(
    plt,
    runs: list[ExperimentRun],
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    paths: list[Path] = []
    configs = sorted({run.partition_config for run in runs})
    for config in configs:
        selected = [run for run in runs if run.partition_config == config]
        algorithms = sorted(
            {run.algorithm for run in selected},
            key=algorithm_sort_key,
        )
        fig, ax = plt.subplots(figsize=(6.8, 4.5))
        for algorithm in algorithms:
            algorithm_runs = [run for run in selected if run.algorithm == algorithm]
            max_rounds = max(len(run.accuracies) for run in algorithm_runs)
            means, errors, rounds = [], [], []
            for round_index in range(max_rounds):
                values = [
                    run.accuracies[round_index]
                    for run in algorithm_runs
                    if round_index < len(run.accuracies)
                ]
                if not values:
                    continue
                mean, stdev = mean_stdev(values)
                rounds.append(round_index + 1)
                means.append(mean)
                errors.append(stdev)
            color = ALGORITHM_COLORS.get(algorithm, "#333333")
            ax.plot(
                rounds,
                means,
                color=color,
                linewidth=1.8,
                label=ALGORITHM_LABELS.get(algorithm, algorithm),
            )
            if len(algorithm_runs) > 1:
                lower = [mean - error for mean, error in zip(means, errors)]
                upper = [mean + error for mean, error in zip(means, errors)]
                ax.fill_between(rounds, lower, upper, color=color, alpha=0.14)
        ax.set_xlabel("Communication round")
        style_axis(ax, "Average local accuracy (%)")
        ax.legend(frameon=False, ncol=min(3, len(algorithms)))
        paths.extend(
            save_figure(
                fig,
                output_dir / f"convergence_{safe_filename(config)}",
                formats,
                dpi,
            )
        )
    return paths


def config_label(config: str) -> str:
    if config.startswith("beta-"):
        match = re.match(r"beta-([^-]+)", config)
        return f"beta={match.group(1)}" if match else config
    match = re.match(r"kn-ways-(\d+)-shots-(\d+)-stdev-(\d+)", config)
    if match:
        return f"{match.group(1)}-way, {match.group(2)}-shot"
    return config


def safe_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value
    )


def generate_charts(
    runs: list[ExperimentRun],
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    plt = import_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    partitions = sorted({run.partition for run in runs})
    for partition in partitions:
        if partition == "beta":
            written.extend(
                plot_beta_metric(
                    plt,
                    runs,
                    "accuracy",
                    lambda run: run.last10_accuracy,
                    "Last-10 average local accuracy (%)",
                    output_dir,
                    formats,
                    dpi,
                )
            )
            written.extend(
                plot_beta_metric(
                    plt,
                    runs,
                    "communication",
                    lambda run: run.total_communication_bytes / (1024 * 1024),
                    "Total logical communication (MiB)",
                    output_dir,
                    formats,
                    dpi,
                )
            )
        else:
            written.extend(
                plot_categorical_metric(
                    plt,
                    runs,
                    partition,
                    "accuracy",
                    lambda run: run.last10_accuracy,
                    "Last-10 average local accuracy (%)",
                    output_dir,
                    formats,
                    dpi,
                )
            )
            written.extend(
                plot_categorical_metric(
                    plt,
                    runs,
                    partition,
                    "communication",
                    lambda run: run.total_communication_bytes / (1024 * 1024),
                    "Total logical communication (MiB)",
                    output_dir,
                    formats,
                    dpi,
                )
            )
        written.extend(plot_delta(plt, runs, partition, output_dir, formats, dpi))
    written.extend(plot_convergence(plt, runs, output_dir, formats, dpi))
    plt.close("all")
    return written


def main() -> int:
    args = parse_args()
    try:
        runs, warnings = load_runs(args.experiment_dirs)
        if args.output_dir:
            output_dir = args.output_dir.expanduser().resolve()
        elif len(args.experiment_dirs) == 1:
            output_dir = args.experiment_dirs[0].expanduser().resolve() / "figures"
        else:
            output_dir = Path.cwd() / "fl" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_data_path = output_dir / "plot_data.csv"
        write_plot_data(plot_data_path, runs)
        paths = generate_charts(runs, output_dir, args.formats, args.dpi)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"Loaded runs: {len(runs)}")
    print(f"Plot data: {plot_data_path}")
    print(f"Charts: {len(paths)}")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
