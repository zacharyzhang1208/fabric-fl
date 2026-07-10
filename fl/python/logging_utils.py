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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if args.mode == "task_heter":
        mode_text = f"task_heter_w{args.ways}_s{args.shots}_sd{args.stdev}"
    elif args.mode == "dirichlet":
        mode_text = f"dirichlet_a{args.dirichlet_alpha}_samples{args.samples_per_client}"
    else:
        mode_text = args.mode
    filename = f"{timestamp}_{args.dataset}_{args.algorithm}_{mode_text}_clients{args.num_clients}_rounds{args.rounds}.log"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / filename


@contextmanager
def redirect_output_to_log(log_path: Path) -> Iterator[None]:
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout = Tee(sys.stdout, log_file)
        stderr = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield
