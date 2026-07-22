from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.common import aggregate_model_updates
from fl_client import ModelUpdate


class ModelAggregationTests(unittest.TestCase):
    def test_trimmed_mean_drops_coordinate_extremes(self) -> None:
        updates = [
            model_update(0, [0.0, 100.0]),
            model_update(1, [1.0, 2.0]),
            model_update(2, [2.0, 3.0]),
            model_update(3, [3.0, 4.0]),
            model_update(4, [100.0, -100.0]),
        ]

        aggregated = aggregate_model_updates(
            updates,
            aggregation="trimmed_mean",
            trim_count=1,
        )

        self.assertTrue(torch.allclose(aggregated["weight"], torch.tensor([2.0, 3.0])))

    def test_multi_krum_excludes_distant_update(self) -> None:
        updates = [
            model_update(0, [0.0, 0.0]),
            model_update(1, [0.1, 0.0]),
            model_update(2, [0.0, 0.1]),
            model_update(3, [0.1, 0.1]),
            model_update(4, [100.0, 100.0]),
        ]

        aggregated = aggregate_model_updates(
            updates,
            aggregation="multi_krum",
            krum_f=1,
        )

        self.assertTrue(torch.all(aggregated["weight"] < 1.0))


def model_update(client_id: int, values: list[float]) -> ModelUpdate:
    return ModelUpdate(
        round_id=1,
        client_id=client_id,
        state_dict={"weight": torch.tensor(values)},
        num_samples=1,
        payload_bytes=8,
    )


if __name__ == "__main__":
    unittest.main()
