from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_adapter import (
    FabricAdapterClient,
    ClientReputation,
    GlobalPrototypePayload,
    PrototypePayload,
    ReputationReport,
)


class PrototypePayloadTests(unittest.TestCase):
    def test_tensor_round_trip(self) -> None:
        prototypes = torch.tensor([[0.1234567, -0.5], [1.25, 0.0]], dtype=torch.float32)
        counts = torch.tensor([4, 0], dtype=torch.float32)

        payload = PrototypePayload.from_tensors(2, 7, prototypes, counts)
        restored_prototypes, restored_counts = payload.to_tensors()

        self.assertEqual(payload.encoding, "fixed-point-int64")
        self.assertEqual(payload.shape, (2, 2))
        self.assertTrue(torch.allclose(restored_prototypes, prototypes, atol=1e-6))
        self.assertTrue(torch.equal(restored_counts, counts))

    def test_json_round_trip(self) -> None:
        payload = PrototypePayload.from_tensors(
            1,
            3,
            torch.tensor([[0.25, 0.5]], dtype=torch.float32),
            torch.tensor([2]),
        )
        restored = PrototypePayload.from_dict(json.loads(payload.to_json()))
        self.assertEqual(restored, payload)

    def test_rejects_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            PrototypePayload.from_tensors(
                1,
                0,
                torch.tensor([[float("nan")]]),
                torch.tensor([1]),
            )


class FabricAdapterClientTests(unittest.TestCase):
    @patch("fabric_adapter.urlopen")
    def test_create_round_includes_reputation_scope(self, mocked_urlopen: MagicMock) -> None:
        response = MagicMock()
        response.read.return_value = b'{"result":null}'
        mocked_urlopen.return_value.__enter__.return_value = response

        FabricAdapterClient().create_round(101, 100, 2, 5, 10, 50)

        request = mocked_urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["transaction"], "CreateRound")
        self.assertEqual(body["args"], ["101", "100", "2", "5", "10", "50", "1000000"])

    @patch("fabric_adapter.urlopen")
    def test_upload_prototype_uses_dedicated_transaction(self, mocked_urlopen: MagicMock) -> None:
        response = MagicMock()
        response.read.return_value = b'{"result":null}'
        mocked_urlopen.return_value.__enter__.return_value = response

        payload = PrototypePayload.from_tensors(
            1,
            4,
            torch.tensor([[0.75]], dtype=torch.float32),
            torch.tensor([1]),
        )
        FabricAdapterClient().upload_prototype(payload)

        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8080/submit")
        body = json.loads(request.data)
        self.assertEqual(body["transaction"], "SubmitPrototype")
        self.assertEqual(body["args"][:2], ["1", "4"])
        self.assertEqual(json.loads(body["args"][2]), payload.to_dict())

    def test_global_prototype_to_tensors(self) -> None:
        payload = GlobalPrototypePayload.from_dict(
            {
                "encoding": "fixed-point-int64",
                "round_id": 2,
                "shape": [1, 2],
                "scale": 100,
                "values": [25, -50],
                "counts": [3],
            }
        )
        prototypes, counts = payload.to_tensors()
        self.assertTrue(torch.equal(prototypes, torch.tensor([[0.25, -0.5]])))
        self.assertTrue(torch.equal(counts, torch.tensor([3.0])))

    def test_reputation_report_from_dict(self) -> None:
        report = ReputationReport.from_dict(
            {
                "round_id": 102,
                "experiment_id": 100,
                "sequence": 3,
                "warmup": False,
                "detection_used": True,
                "median_distance": 10,
                "mad": 2,
                "threshold": 16,
                "assessments": [
                    {
                        "client_id": 4,
                        "distance": 30,
                        "threshold": 16,
                        "assessed": True,
                        "anomalous": True,
                        "included": False,
                        "previous_score": 6400,
                        "new_score": 5120,
                        "status": "WATCH",
                        "consecutive_anomalies": 2,
                    }
                ],
            }
        )
        self.assertEqual(report.assessments[0].client_id, 4)
        self.assertFalse(report.assessments[0].included)

    def test_client_reputation_from_dict(self) -> None:
        reputation = ClientReputation.from_dict(
            {
                "experiment_id": 100,
                "client_id": 4,
                "score": 5120,
                "status": "WATCH",
                "assessments": 2,
                "anomalies": 2,
                "consecutive_anomalies": 2,
                "last_round_id": 102,
            }
        )
        self.assertEqual(reputation.client_id, 4)
        self.assertEqual(reputation.score, 5120)


if __name__ == "__main__":
    unittest.main()
