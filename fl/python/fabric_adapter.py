"""HTTP client and wire format for exchanging prototypes with Fabric."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch


DEFAULT_ADAPTER_URL = "http://127.0.0.1:18080"
DEFAULT_PROTOTYPE_SCALE = 1_000_000


class FabricAdapterError(RuntimeError):
    """Raised when the HTTP adapter rejects a request or cannot be reached."""


@dataclass(frozen=True)
class PrototypeBatchStatus:
    round_id: int
    expected_clients: int
    received_clients: int
    status: str
    round_result: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PrototypeBatchStatus":
        status = cls(
            round_id=int(value["round_id"]),
            expected_clients=int(value["expected_clients"]),
            received_clients=int(value["received_clients"]),
            status=str(value["status"]),
            round_result=value.get("round_result"),
        )
        if status.status not in {"COLLECTING", "READY", "SUBMITTING", "PROCESSED"}:
            raise ValueError(f"unsupported prototype batch status {status.status!r}")
        if status.round_result is not None and not isinstance(status.round_result, dict):
            raise ValueError("prototype batch round_result must be an object")
        return status


@dataclass(frozen=True)
class PrototypePayload:
    round_id: int
    client_id: int
    shape: tuple[int, int]
    scale: int
    values: tuple[int, ...]
    counts: tuple[int, ...]
    encoding: str = "fixed-point-int64"

    @classmethod
    def from_tensors(
        cls,
        round_id: int,
        client_id: int,
        prototypes: torch.Tensor,
        counts: torch.Tensor,
        scale: int = DEFAULT_PROTOTYPE_SCALE,
    ) -> "PrototypePayload":
        if round_id < 1:
            raise ValueError("round_id must be positive")
        if client_id < 0:
            raise ValueError("client_id must be non-negative")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if prototypes.ndim != 2:
            raise ValueError("prototypes must have shape [num_classes, dimension]")
        if counts.ndim != 1 or counts.shape[0] != prototypes.shape[0]:
            raise ValueError("counts must have one value per prototype class")

        prototypes_cpu = prototypes.detach().to(device="cpu", dtype=torch.float64)
        counts_cpu = counts.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(prototypes_cpu).all():
            raise ValueError("prototypes must contain only finite values")
        if not torch.isfinite(counts_cpu).all() or (counts_cpu < 0).any():
            raise ValueError("counts must contain finite non-negative values")
        if not torch.equal(counts_cpu, counts_cpu.round()):
            raise ValueError("counts must contain integer values")

        max_value = torch.iinfo(torch.int64).max / scale
        if prototypes_cpu.numel() and prototypes_cpu.abs().max().item() > max_value:
            raise ValueError("scaled prototype value exceeds int64 range")

        quantized = torch.round(prototypes_cpu * scale).to(torch.int64)
        integer_counts = counts_cpu.to(torch.int64)
        return cls(
            round_id=round_id,
            client_id=client_id,
            shape=(int(prototypes.shape[0]), int(prototypes.shape[1])),
            scale=scale,
            values=tuple(int(value) for value in quantized.reshape(-1).tolist()),
            counts=tuple(int(value) for value in integer_counts.tolist()),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PrototypePayload":
        if value.get("encoding") != "fixed-point-int64":
            raise ValueError("unsupported prototype encoding")

        shape = value.get("shape")
        values = value.get("values")
        counts = value.get("counts")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("prototype shape must contain two dimensions")
        if not isinstance(values, list) or not isinstance(counts, list):
            raise ValueError("prototype values and counts must be arrays")

        payload = cls(
            round_id=int(value["round_id"]),
            client_id=int(value["client_id"]),
            shape=(int(shape[0]), int(shape[1])),
            scale=int(value["scale"]),
            values=tuple(int(item) for item in values),
            counts=tuple(int(item) for item in counts),
        )
        payload.validate()
        return payload

    def validate(self) -> None:
        num_classes, dimension = self.shape
        if self.round_id < 1 or self.client_id < 0:
            raise ValueError("invalid round or client id")
        if num_classes < 1 or dimension < 1 or self.scale <= 0:
            raise ValueError("invalid prototype shape or scale")
        if len(self.values) != num_classes * dimension:
            raise ValueError("prototype value count does not match shape")
        if len(self.counts) != num_classes or any(count < 0 for count in self.counts):
            raise ValueError("prototype counts do not match shape")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "encoding": self.encoding,
            "round_id": self.round_id,
            "client_id": self.client_id,
            "shape": list(self.shape),
            "scale": self.scale,
            "values": list(self.values),
            "counts": list(self.counts),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    def to_tensors(self, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        self.validate()
        prototypes = torch.tensor(self.values, dtype=torch.float64).reshape(self.shape)
        prototypes = (prototypes / self.scale).to(dtype=torch.float32, device=device)
        counts = torch.tensor(self.counts, dtype=torch.float32, device=device)
        return prototypes, counts

    @property
    def state_key(self) -> str:
        return f"prototype:{self.round_id}:{self.client_id}"


@dataclass(frozen=True)
class GlobalPrototypePayload:
    round_id: int
    shape: tuple[int, int]
    scale: int
    values: tuple[int, ...]
    counts: tuple[int, ...]
    encoding: str = "fixed-point-int64"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GlobalPrototypePayload":
        if value.get("encoding") != "fixed-point-int64":
            raise ValueError("unsupported global prototype encoding")
        shape = value.get("shape")
        values = value.get("values")
        counts = value.get("counts")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("global prototype shape must contain two dimensions")
        if not isinstance(values, list) or not isinstance(counts, list):
            raise ValueError("global prototype values and counts must be arrays")

        payload = cls(
            round_id=int(value["round_id"]),
            shape=(int(shape[0]), int(shape[1])),
            scale=int(value["scale"]),
            values=tuple(int(item) for item in values),
            counts=tuple(int(item) for item in counts),
        )
        payload.validate()
        return payload

    def validate(self) -> None:
        num_classes, dimension = self.shape
        if self.round_id < 1 or num_classes < 1 or dimension < 1 or self.scale <= 0:
            raise ValueError("invalid global prototype metadata")
        if len(self.values) != num_classes * dimension:
            raise ValueError("global prototype value count does not match shape")
        if len(self.counts) != num_classes or any(count < 0 for count in self.counts):
            raise ValueError("global prototype counts do not match shape")

    def to_tensors(self, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        self.validate()
        prototypes = torch.tensor(self.values, dtype=torch.float64).reshape(self.shape)
        prototypes = (prototypes / self.scale).to(dtype=torch.float32, device=device)
        counts = torch.tensor(self.counts, dtype=torch.float32, device=device)
        return prototypes, counts


@dataclass(frozen=True)
class ClientAssessment:
    client_id: int
    distance: int
    threshold: int
    assessed: bool
    anomalous: bool
    included: bool
    previous_score: int
    new_score: int
    status: str
    consecutive_anomalies: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClientAssessment":
        return cls(
            client_id=int(value["client_id"]),
            distance=int(value["distance"]),
            threshold=int(value["threshold"]),
            assessed=bool(value["assessed"]),
            anomalous=bool(value["anomalous"]),
            included=bool(value["included"]),
            previous_score=int(value["previous_score"]),
            new_score=int(value["new_score"]),
            status=str(value["status"]),
            consecutive_anomalies=int(value["consecutive_anomalies"]),
        )


@dataclass(frozen=True)
class ClientReputation:
    experiment_id: int
    client_id: int
    score: int
    status: str
    assessments: int
    anomalies: int
    consecutive_anomalies: int
    last_round_id: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClientReputation":
        return cls(
            experiment_id=int(value["experiment_id"]),
            client_id=int(value["client_id"]),
            score=int(value["score"]),
            status=str(value["status"]),
            assessments=int(value["assessments"]),
            anomalies=int(value["anomalies"]),
            consecutive_anomalies=int(value["consecutive_anomalies"]),
            last_round_id=int(value["last_round_id"]),
        )


@dataclass(frozen=True)
class ReputationReport:
    round_id: int
    experiment_id: int
    sequence: int
    warmup: bool
    detection_used: bool
    median_distance: int
    mad: int
    threshold: int
    assessments: tuple[ClientAssessment, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReputationReport":
        assessments = value.get("assessments")
        if not isinstance(assessments, list):
            raise ValueError("reputation assessments must be an array")
        return cls(
            round_id=int(value["round_id"]),
            experiment_id=int(value["experiment_id"]),
            sequence=int(value["sequence"]),
            warmup=bool(value["warmup"]),
            detection_used=bool(value["detection_used"]),
            median_distance=int(value["median_distance"]),
            mad=int(value["mad"]),
            threshold=int(value["threshold"]),
            assessments=tuple(ClientAssessment.from_dict(item) for item in assessments),
        )


class FabricAdapterClient:
    def __init__(self, base_url: str = DEFAULT_ADAPTER_URL, timeout: float = 15.0) -> None:
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be a positive finite number")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def submit(self, transaction: str, *args: str) -> Any:
        return self._post("/submit", transaction, args)

    def evaluate(self, transaction: str, *args: str) -> Any:
        return self._post("/evaluate", transaction, args)

    def create_round(
        self,
        round_id: int,
        experiment_id: int,
        sequence: int,
        expected_clients: int,
        num_classes: int,
        dimension: int,
        scale: int = DEFAULT_PROTOTYPE_SCALE,
    ) -> None:
        self.submit(
            "CreateRound",
            str(round_id),
            str(experiment_id),
            str(sequence),
            str(expected_clients),
            str(num_classes),
            str(dimension),
            str(scale),
        )

    def upload_prototype(self, payload: PrototypePayload) -> None:
        payload.validate()
        self.submit(
            "SubmitPrototype",
            str(payload.round_id),
            str(payload.client_id),
            payload.to_json(),
        )

    def open_prototype_batch(
        self,
        round_id: int,
        experiment_id: int,
        sequence: int,
        expected_clients: int,
        num_classes: int,
        dimension: int,
        scale: int,
    ) -> PrototypeBatchStatus:
        config_values = (
            round_id,
            experiment_id,
            sequence,
            expected_clients,
            num_classes,
            dimension,
            scale,
        )
        if min(config_values) < 1:
            raise ValueError("all prototype batch configuration values must be positive")
        value = self._post_json(
            "/prototype-batches/open",
            {
                "round_id": round_id,
                "experiment_id": experiment_id,
                "sequence": sequence,
                "expected_clients": expected_clients,
                "num_classes": num_classes,
                "dimension": dimension,
                "scale": scale,
            },
        )
        if not isinstance(value, dict):
            raise FabricAdapterError("prototype batch response is not a JSON object")
        return PrototypeBatchStatus.from_dict(value)

    def collect_prototype(self, payload: PrototypePayload) -> PrototypeBatchStatus:
        payload.validate()
        value = self._post_json("/prototype-batches/submit", payload.to_dict())
        if not isinstance(value, dict):
            raise FabricAdapterError("prototype submission response is not a JSON object")
        return PrototypeBatchStatus.from_dict(value)

    def upload_prototype_batch(
        self,
        payloads: list[PrototypePayload],
        experiment_id: int,
        sequence: int,
    ) -> PrototypeBatchStatus:
        if not payloads:
            raise ValueError("prototype batch must not be empty")
        round_id = payloads[0].round_id
        client_ids = set()
        for payload in payloads:
            payload.validate()
            if payload.round_id != round_id:
                raise ValueError("all prototype payloads must use the same round_id")
            if payload.client_id in client_ids:
                raise ValueError(f"duplicate prototype client_id {payload.client_id}")
            client_ids.add(payload.client_id)
        expected_client_ids = set(range(len(payloads)))
        if client_ids != expected_client_ids:
            raise ValueError(
                "prototype batch client_ids must exactly cover "
                f"0 through {len(payloads) - 1}"
            )

        num_classes, dimension = payloads[0].shape
        scale = payloads[0].scale
        if any(
            payload.shape != (num_classes, dimension) or payload.scale != scale
            for payload in payloads
        ):
            raise ValueError("all prototype payloads must have the same shape and scale")

        self.open_prototype_batch(
            round_id=round_id,
            experiment_id=experiment_id,
            sequence=sequence,
            expected_clients=len(payloads),
            num_classes=num_classes,
            dimension=dimension,
            scale=scale,
        )
        status = None
        for payload in payloads:
            status = self.collect_prototype(payload)
        assert status is not None
        if status.status != "PROCESSED" or status.round_result is None:
            raise FabricAdapterError(
                "prototype round was not processed after receiving all clients: "
                f"status={status.status} received={status.received_clients}/"
                f"{status.expected_clients}"
            )
        if (
            int(status.round_result.get("round_id", -1)) != round_id
            or status.round_result.get("status") != "FINALIZED"
        ):
            raise FabricAdapterError("ProcessRound returned an invalid completion receipt")
        return status

    def finalize_round(self, round_id: int) -> None:
        self.submit("FinalizeRound", str(round_id))

    def get_global_prototype(self, round_id: int) -> GlobalPrototypePayload:
        value = self.evaluate("GetGlobalPrototype", str(round_id))
        if not isinstance(value, dict):
            raise FabricAdapterError("global prototype response is not a JSON object")
        return GlobalPrototypePayload.from_dict(value)

    def get_round_reputation_report(self, round_id: int) -> ReputationReport:
        value = self.evaluate("GetRoundReputationReport", str(round_id))
        if not isinstance(value, dict):
            raise FabricAdapterError("reputation report response is not a JSON object")
        return ReputationReport.from_dict(value)

    def get_client_reputation(self, experiment_id: int, client_id: int) -> ClientReputation:
        value = self.evaluate("GetClientReputation", str(experiment_id), str(client_id))
        if not isinstance(value, dict):
            raise FabricAdapterError("client reputation response is not a JSON object")
        return ClientReputation.from_dict(value)

    def _post(self, path: str, transaction: str, args: tuple[str, ...]) -> Any:
        if not transaction:
            raise ValueError("transaction is required")
        if any(not isinstance(arg, str) for arg in args):
            raise TypeError("Fabric transaction arguments must be strings")

        return self._post_json(
            path,
            {"transaction": transaction, "args": list(args)},
        )

    def _post_json(self, path: str, value: dict[str, Any]) -> Any:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read()
        except HTTPError as exc:
            message = _error_message(exc.read())
            raise FabricAdapterError(f"adapter returned HTTP {exc.code}: {message}") from exc
        except URLError as exc:
            raise FabricAdapterError(f"cannot reach Fabric adapter: {exc.reason}") from exc

        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise FabricAdapterError("adapter returned invalid JSON") from exc
        if not isinstance(decoded, dict) or "result" not in decoded:
            raise FabricAdapterError("adapter response does not contain result")
        return decoded["result"]


def _error_message(body: bytes) -> str:
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return body.decode("utf-8", errors="replace") or "request failed"
    if isinstance(decoded, dict) and isinstance(decoded.get("error"), str):
        return decoded["error"]
    return "request failed"
