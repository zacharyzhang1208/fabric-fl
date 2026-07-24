"""Logging helpers for local FL simulations."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterator


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


def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes} B ({num_bytes / 1024:.2f} KiB)"
    return f"{num_bytes} B ({num_bytes / (1024 * 1024):.2f} MiB)"


def make_log_path(args) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    backend = getattr(args, "backend", "memory")
    attack = "clean" if args.attack == "none" else f"attack-{args.attack}"
    partition = getattr(args, "partition", "beta")
    model_config = getattr(args, "model_config", "homogeneous")
    parts = [
        timestamp,
        safe_filename_part(args.dataset),
        safe_filename_part(args.algorithm),
        safe_filename_part(backend),
        safe_filename_part(attack),
        f"model-config-{safe_filename_part(model_config)}",
        f"partition-{partition}",
    ]
    if partition == "beta":
        parts.extend([f"beta-{args.beta:g}", f"samples-{args.samples_per_client}"])
    else:
        parts.extend(
            [
                f"ways-{args.ways}",
                f"shots-{args.shots}",
                f"stdev-{args.stdev}",
            ]
        )
    if args.algorithm == "prototype":
        parts.extend(
            [
                f"proto-weight-{args.proto_weight:g}",
                f"proto-temperature-{args.proto_temperature:g}",
                f"prototypes-per-class-{args.prototypes_per_class}",
            ]
        )
        if getattr(args, "prototype_synthesis", False):
            parts.append("prototype-synthesis")
    parts.extend([f"clients-{args.num_clients}", f"rounds-{args.rounds}"])
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return unique_log_path(log_dir, "_".join(parts), ".log")


def safe_filename_part(value: object) -> str:
    text = str(value).strip().replace(" ", "-")
    return "".join(char if char.isalnum() or char in {"-", "."} else "-" for char in text)


def unique_log_path(log_dir: Path, stem: str, suffix: str) -> Path:
    path = log_dir / f"{stem}{suffix}"
    if not path.exists():
        return path

    index = 2
    while True:
        candidate = log_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


@contextmanager
def redirect_output_to_log(log_path: Path) -> Iterator[None]:
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout = Tee(sys.stdout, log_file)
        stderr = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield
