#!/usr/bin/env python3
"""Run and summarize a sequence of federated-learning experiments."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PATH = REPO_ROOT / "fl" / "python" / "main.py"
DEFAULT_ALGORITHMS = ("local", "fedavg", "fedprox", "prototype")
DEFAULT_BETAS = (10.0, 1.0, 0.5, 0.2, 0.1)
DEFAULT_SEEDS = (1234,)
MANIFEST_FIELDS = (
    "task_id",
    "beta",
    "algorithm",
    "seed",
    "status",
    "exit_code",
    "log_path",
    "last10_avg_acc",
    "final_avg_acc",
    "best_avg_acc",
    "delta_vs_local",
    "command",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run clean beta x algorithm experiments and compare every method "
            "with Local using matched seeds."
        )
    )
    parser.add_argument("--dataset", choices=("mnist", "cifar10", "cifar100"))
    parser.add_argument("--betas", nargs="+", type=float, default=list(DEFAULT_BETAS))
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=(
            "local",
            "fedavg",
            "fedprox",
            "prototype",
            "trimmed_mean",
            "multi_krum",
        ),
        default=list(DEFAULT_ALGORITHMS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "fl" / "data"))
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument(
        "--model-config",
        choices=("homogeneous", "heterogeneous"),
        default="homogeneous",
    )
    parser.add_argument("--samples-per-client", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument(
        "--test-limit",
        type=int,
        default=300,
        help="Per-client distribution-matched local test samples",
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--proto-weight", type=float, default=0.5)
    parser.add_argument("--proto-temperature", type=float, default=0.1)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--eval-scope", choices=("local",), default="local")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New experiment directory; defaults to fl/experiments/experiment_TIMESTAMP",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume unfinished/failed tasks from an existing experiment directory",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for fl/python/main.py (default: current interpreter)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the plan without training")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed run")
    args = parser.parse_args()

    if args.resume and args.output_dir:
        parser.error("--resume and --output-dir cannot be used together")
    if not args.resume and not args.dataset:
        parser.error("--dataset is required when creating an experiment")
    if any(beta <= 0 for beta in args.betas):
        parser.error("all --betas values must be positive")
    if args.num_clients <= 0:
        parser.error("--num-clients must be positive")
    if args.samples_per_client <= 0:
        parser.error("--samples-per-client must be positive")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.test_limit <= 0:
        parser.error("--test-limit must be positive")
    if args.proto_temperature <= 0:
        parser.error("--proto-temperature must be positive")
    if not args.seeds:
        parser.error("--seeds must contain at least one seed")
    if len(set(args.algorithms)) != len(args.algorithms):
        parser.error("--algorithms contains duplicate values")
    if "local" not in args.algorithms:
        parser.error("--algorithms must include local to calculate delta_vs_local")
    if args.model_config == "heterogeneous":
        unsupported = set(args.algorithms) - {"local", "prototype"}
        if unsupported:
            parser.error(
                "--model-config heterogeneous only supports algorithms: "
                "local prototype"
            )
        if args.dataset != "mnist":
            parser.error("--model-config heterogeneous currently requires --dataset mnist")
        if args.num_clients < 3:
            parser.error("--model-config heterogeneous requires at least 3 clients")
    return args


def beta_text(beta: float) -> str:
    return f"{beta:g}"


def task_id(beta: float, algorithm: str, seed: int) -> str:
    return f"beta-{beta_text(beta)}__algorithm-{algorithm}__seed-{seed}"


def build_command(
    args: argparse.Namespace,
    beta: float,
    algorithm: str,
    seed: int,
    log_dir: Path,
) -> list[str]:
    return [
        args.python,
        str(MAIN_PATH),
        "--dataset",
        args.dataset,
        "--data-dir",
        str(Path(args.data_dir).resolve()),
        "--algorithm",
        algorithm,
        "--backend",
        "memory",
        "--partition",
        "beta",
        "--beta",
        beta_text(beta),
        "--samples-per-client",
        str(args.samples_per_client),
        "--num-clients",
        str(args.num_clients),
        "--model-config",
        args.model_config,
        "--rounds",
        str(args.rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--test-limit",
        str(args.test_limit),
        "--eval-scope",
        args.eval_scope,
        "--lr",
        str(args.lr),
        "--optimizer",
        args.optimizer,
        "--proto-weight",
        str(args.proto_weight),
        "--proto-temperature",
        str(args.proto_temperature),
        "--fedprox-mu",
        str(args.fedprox_mu),
        "--seed",
        str(seed),
        "--attack",
        "none",
        "--log-dir",
        str(log_dir),
    ]


def make_tasks(args: argparse.Namespace, experiment_dir: Path) -> list[dict[str, str]]:
    tasks = []
    for beta in args.betas:
        for seed in args.seeds:
            for algorithm in args.algorithms:
                identifier = task_id(beta, algorithm, seed)
                log_dir = experiment_dir / "runs" / identifier
                command = build_command(args, beta, algorithm, seed, log_dir)
                tasks.append(
                    {
                        "task_id": identifier,
                        "beta": beta_text(beta),
                        "algorithm": algorithm,
                        "seed": str(seed),
                        "status": "pending",
                        "exit_code": "",
                        "log_path": "",
                        "last10_avg_acc": "",
                        "final_avg_acc": "",
                        "best_avg_acc": "",
                        "delta_vs_local": "",
                        "command": json.dumps(command),
                    }
                )
    return tasks


def write_manifest(path: Path, tasks: Iterable[dict[str, str]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(tasks)
    temporary.replace(path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def accuracy_pattern() -> re.Pattern[str]:
    metric = r"(?:benign_)?avg_acc"
    return re.compile(rf"(?<![A-Za-z_]){metric}=([0-9]+(?:\.[0-9]+)?)%")


def parse_accuracies(log_path: Path) -> list[float]:
    pattern = accuracy_pattern()
    accuracies = []
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "aggregator:" not in line:
                continue
            match = pattern.search(line)
            if match:
                accuracies.append(float(match.group(1)))
    return accuracies


def fill_accuracy_metrics(task: dict[str, str], eval_scope: str) -> None:
    log_path = Path(task["log_path"])
    accuracies = parse_accuracies(log_path)
    if not accuracies:
        raise ValueError(f"No aggregator accuracy found in {log_path}")
    task["last10_avg_acc"] = f"{statistics.fmean(accuracies[-10:]):.6f}"
    task["final_avg_acc"] = f"{accuracies[-1]:.6f}"
    task["best_avg_acc"] = f"{max(accuracies):.6f}"


def update_local_deltas(tasks: list[dict[str, str]]) -> None:
    local_scores = {
        (task["beta"], task["seed"]): float(task["last10_avg_acc"])
        for task in tasks
        if task["status"] == "completed"
        and task["algorithm"] == "local"
        and task["last10_avg_acc"]
    }
    for task in tasks:
        if task["status"] != "completed" or not task["last10_avg_acc"]:
            task["delta_vs_local"] = ""
            continue
        local_score = local_scores.get((task["beta"], task["seed"]))
        task["delta_vs_local"] = (
            f"{float(task['last10_avg_acc']) - local_score:.6f}"
            if local_score is not None
            else ""
        )

def mean_and_stdev(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.6f}", f"{stdev:.6f}"


def write_summary(path: Path, tasks: list[dict[str, str]]) -> None:
    fields = (
        "beta",
        "algorithm",
        "expected_runs",
        "completed_runs",
        "last10_acc_mean",
        "last10_acc_stdev",
        "final_acc_mean",
        "best_acc_mean",
        "delta_vs_local_mean",
        "delta_vs_local_stdev",
    )
    groups: dict[tuple[float, str], list[dict[str, str]]] = {}
    for task in tasks:
        groups.setdefault((float(task["beta"]), task["algorithm"]), []).append(task)

    rows = []
    for (beta, algorithm), group in sorted(groups.items()):
        completed = [
            task
            for task in group
            if task["status"] == "completed" and task["last10_avg_acc"]
        ]
        last10 = [float(task["last10_avg_acc"]) for task in completed]
        final = [float(task["final_avg_acc"]) for task in completed]
        best = [float(task["best_avg_acc"]) for task in completed]
        deltas = [float(task["delta_vs_local"]) for task in completed if task["delta_vs_local"]]
        last10_mean, last10_stdev = mean_and_stdev(last10)
        delta_mean, delta_stdev = mean_and_stdev(deltas)
        rows.append(
            {
                "beta": beta_text(beta),
                "algorithm": algorithm,
                "expected_runs": len(group),
                "completed_runs": len(completed),
                "last10_acc_mean": last10_mean,
                "last10_acc_stdev": last10_stdev,
                "final_acc_mean": mean_and_stdev(final)[0],
                "best_acc_mean": mean_and_stdev(best)[0],
                "delta_vs_local_mean": delta_mean,
                "delta_vs_local_stdev": delta_stdev,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_log(run_dir: Path) -> Path | None:
    logs = sorted(run_dir.glob("*.log"), key=lambda path: path.stat().st_mtime)
    return logs[-1].resolve() if logs else None


def git_metadata() -> dict[str, object]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }


def create_experiment(args: argparse.Namespace) -> tuple[Path, list[dict[str, str]]]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (REPO_ROOT / "fl" / "experiments" / f"experiment_{timestamp}").resolve()
    )
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks(args, experiment_dir)
    metadata = {
        **git_metadata(),
        "dataset": args.dataset,
        "betas": args.betas,
        "algorithms": args.algorithms,
        "seeds": args.seeds,
        "num_clients": args.num_clients,
        "model_config": args.model_config,
        "samples_per_client": args.samples_per_client,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "test_limit": args.test_limit,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "proto_weight": args.proto_weight,
        "fedprox_mu": args.fedprox_mu,
        "eval_scope": args.eval_scope,
    }
    (experiment_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return experiment_dir, tasks


def load_experiment(args: argparse.Namespace) -> tuple[Path, list[dict[str, str]]]:
    experiment_dir = args.resume.resolve()
    manifest_path = experiment_dir / "manifest.csv"
    metadata_path = experiment_dir / "metadata.json"
    if not manifest_path.exists() or not metadata_path.exists():
        raise ValueError(f"Not an experiment directory: {experiment_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    args.dataset = metadata["dataset"]
    args.eval_scope = metadata["eval_scope"]
    tasks = read_manifest(manifest_path)
    return experiment_dir, tasks


def run_tasks(
    args: argparse.Namespace,
    experiment_dir: Path,
    tasks: list[dict[str, str]],
) -> int:
    manifest_path = experiment_dir / "manifest.csv"
    summary_path = experiment_dir / "summary.csv"
    write_manifest(manifest_path, tasks)
    write_summary(summary_path, tasks)

    runnable = [task for task in tasks if task["status"] != "completed"]
    print(f"Experiment directory: {experiment_dir}")
    print(f"Tasks: {len(tasks)} total, {len(runnable)} to run")
    if args.dry_run:
        for task in runnable:
            print(f"[dry-run] {task['task_id']}")
            print(f"  {shlex.join(json.loads(task['command']))}")
        return 0

    failures = 0
    for index, task in enumerate(runnable, start=1):
        command = json.loads(task["command"])
        run_dir = Path(command[command.index("--log-dir") + 1])
        run_dir.mkdir(parents=True, exist_ok=True)
        task["status"] = "running"
        task["exit_code"] = ""
        write_manifest(manifest_path, tasks)

        print()
        print(f"[{index}/{len(runnable)}] {task['task_id']}")
        try:
            result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        except KeyboardInterrupt:
            task["status"] = "interrupted"
            write_manifest(manifest_path, tasks)
            update_local_deltas(tasks)
            write_summary(summary_path, tasks)
            print("\nInterrupted. Resume with:")
            print(
                "  "
                + shlex.join(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--resume",
                        str(experiment_dir),
                    ]
                )
            )
            return 130

        task["exit_code"] = str(result.returncode)
        log_path = find_log(run_dir)
        task["log_path"] = str(log_path) if log_path else ""
        if result.returncode == 0 and log_path:
            try:
                fill_accuracy_metrics(task, args.eval_scope)
                task["status"] = "completed"
            except ValueError as exc:
                task["status"] = "failed"
                print(f"Result parsing failed: {exc}", file=sys.stderr)
        else:
            task["status"] = "failed"

        if task["status"] == "failed":
            failures += 1
        update_local_deltas(tasks)
        write_manifest(manifest_path, tasks)
        write_summary(summary_path, tasks)
        print(f"Status: {task['status']}")
        if args.fail_fast and task["status"] == "failed":
            break

    completed = sum(task["status"] == "completed" for task in tasks)
    print()
    print(f"Completed: {completed}/{len(tasks)}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    try:
        if args.resume:
            experiment_dir, tasks = load_experiment(args)
        else:
            experiment_dir, tasks = create_experiment(args)
        return run_tasks(args, experiment_dir, tasks)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
