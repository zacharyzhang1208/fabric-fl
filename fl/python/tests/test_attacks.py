from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.attacks import poison_prototype_update
from fl_client import ClientUpdate


class PrototypeAttackTests(unittest.TestCase):
    def test_targeted_label_flip_copies_source_prototype_to_target_class(self) -> None:
        payload = ClientUpdate(
            round_id=1,
            client_id=0,
            prototypes=torch.tensor([[1.0, 2.0], [10.0, 20.0], [30.0, 40.0]]),
            counts=torch.tensor([5.0, 3.0, 0.0]),
            payload_bytes=100,
        )

        poisoned = poison_prototype_update(
            payload,
            attack="targeted_label_flip",
            attack_scale=1.0,
            num_classes=3,
            flip_source_class=0,
            flip_target_class=2,
        )

        self.assertTrue(torch.equal(poisoned.prototypes[2], payload.prototypes[0]))
        self.assertEqual(float(poisoned.counts[2]), 5.0)
        self.assertTrue(torch.equal(poisoned.prototypes[1], payload.prototypes[1]))


if __name__ == "__main__":
    unittest.main()
