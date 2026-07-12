from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.prototype import aggregate_prototypes_via_fabric
from fabric_adapter import GlobalPrototypePayload, PrototypePayload
from fl_client import ClientUpdate


class FakeAdapter:
    def __init__(self) -> None:
        self.uploaded: list[PrototypePayload] = []
        self.finalized: int | None = None

    def upload_prototype(self, payload: PrototypePayload) -> None:
        self.uploaded.append(payload)

    def finalize_round(self, round_id: int) -> None:
        self.finalized = round_id

    def get_global_prototype(self, round_id: int) -> GlobalPrototypePayload:
        return GlobalPrototypePayload(
            round_id=round_id,
            shape=(2, 2),
            scale=1_000_000,
            values=(2_000_000, 3_000_000, 5_000_000, 7_000_000),
            counts=(2, 1),
        )


class FabricPrototypeAggregationTests(unittest.TestCase):
    def test_uploads_payloads_and_returns_global_tensors(self) -> None:
        payloads = [
            ClientUpdate(
                round_id=1,
                client_id=0,
                prototypes=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                counts=torch.tensor([5.0, 0.0]),
                payload_bytes=40,
            ),
            ClientUpdate(
                round_id=1,
                client_id=1,
                prototypes=torch.tensor([[3.0, 4.0], [5.0, 7.0]]),
                counts=torch.tensor([2.0, 1.0]),
                payload_bytes=40,
            ),
        ]
        adapter = FakeAdapter()

        prototypes, counts = aggregate_prototypes_via_fabric(
            adapter=adapter,
            ledger_round_id=1001,
            payloads=payloads,
            device=torch.device("cpu"),
            num_classes=2,
            scale=1_000_000,
        )

        self.assertEqual([payload.client_id for payload in adapter.uploaded], [0, 1])
        self.assertTrue(all(payload.round_id == 1001 for payload in adapter.uploaded))
        self.assertEqual(adapter.finalized, 1001)
        self.assertTrue(torch.equal(prototypes, torch.tensor([[2.0, 3.0], [5.0, 7.0]])))
        self.assertTrue(torch.equal(counts, torch.tensor([2.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
