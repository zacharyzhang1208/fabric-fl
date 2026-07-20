"""Docker container traffic counters for Fabric experiments."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess


DEFAULT_FABRIC_CONTAINERS = (
    "orderer.org1.example.com",
    "orderer.org2.example.com",
    "orderer.org3.example.com",
    "orderer.org4.example.com",
    "orderer.org5.example.com",
    "peer0.org1.example.com",
    "peer0.org2.example.com",
    "peer0.org3.example.com",
    "peer0.org4.example.com",
    "peer0.org5.example.com",
)


@dataclass(frozen=True)
class TrafficSnapshot:
    rx_bytes: int
    tx_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.rx_bytes + self.tx_bytes

    def delta(self, previous: "TrafficSnapshot") -> "TrafficSnapshot":
        return TrafficSnapshot(
            rx_bytes=self.rx_bytes - previous.rx_bytes,
            tx_bytes=self.tx_bytes - previous.tx_bytes,
        )


class FabricTrafficMonitor:
    def __init__(self, containers: tuple[str, ...] = DEFAULT_FABRIC_CONTAINERS) -> None:
        if not containers:
            raise ValueError("At least one Fabric container is required")
        self.containers = containers
        self.baseline = self.snapshot()
        self.previous = self.baseline

    def snapshot(self) -> TrafficSnapshot:
        rx_total = 0
        tx_total = 0
        for container in self.containers:
            rx_total += read_container_counter(container, "rx_bytes")
            tx_total += read_container_counter(container, "tx_bytes")
        return TrafficSnapshot(rx_bytes=rx_total, tx_bytes=tx_total)

    def round_delta(self) -> tuple[TrafficSnapshot, TrafficSnapshot]:
        current = self.snapshot()
        round_delta = current.delta(self.previous)
        total_delta = current.delta(self.baseline)
        self.previous = current
        return round_delta, total_delta


def read_container_counter(container: str, counter: str) -> int:
    path = f"/sys/class/net/eth0/statistics/{counter}"
    try:
        result = subprocess.run(
            ["docker", "exec", container, "cat", path],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("docker command was not found") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"read traffic counter from {container}: {message}") from exc
    return int(result.stdout.strip())
