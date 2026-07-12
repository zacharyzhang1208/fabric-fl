from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_adapter import FabricAdapterClient, GlobalPrototypePayload, PrototypePayload


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


if __name__ == "__main__":
    unittest.main()
