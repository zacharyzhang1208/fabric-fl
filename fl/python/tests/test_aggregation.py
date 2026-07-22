from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.common import aggregate_model_updates, aggregate_prototypes
from fl_client import ClientUpdate, ModelUpdate


class ModelAggregationTests(unittest.TestCase):
    def test_prototype_aggregation_uses_sample_count_weights(self) -> None:
        payloads = [
            client_update(0, [[1.0, 2.0], [3.0, 4.0]], [5.0, 0.0]),
            client_update(1, [[2.0, 4.0], [5.0, 7.0]], [2.0, 1.0]),
        ]

        prototypes, counts = aggregate_prototypes(
            payloads,
            device=torch.device("cpu"),
            num_classes=2,
        )

        self.assertTrue(
            torch.allclose(
                prototypes,
                torch.tensor([[9.0 / 7.0, 18.0 / 7.0], [5.0, 7.0]]),
            )
        )
        self.assertTrue(torch.equal(counts, torch.tensor([7.0, 1.0])))

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


def client_update(
    client_id: int,
    prototypes: list[list[float]],
    counts: list[float],
) -> ClientUpdate:
    return ClientUpdate(
        round_id=1,
        client_id=client_id,
        prototypes=torch.tensor(prototypes),
        counts=torch.tensor(counts),
        payload_bytes=8,
    )


if __name__ == "__main__":
    unittest.main()
