from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import (
    distribution_matched_quotas,
    make_dirichlet_client_subsets,
    make_kn_client_subsets,
)


class FakeDataset:
    def __init__(self, num_classes: int, samples_per_class: int) -> None:
        self.targets = [
            label
            for label in range(num_classes)
            for _ in range(samples_per_class)
        ]

    def __len__(self) -> int:
        return len(self.targets)


class DistributionMatchedQuotasTests(unittest.TestCase):
    def test_matches_training_distribution(self) -> None:
        quotas = distribution_matched_quotas(
            train_counts=[5, 2, 0],
            caps=[100, 100, 100],
            target_size=7,
        )

        self.assertEqual(quotas, [5, 2, 0])

    def test_respects_test_class_caps_and_redistributes(self) -> None:
        quotas = distribution_matched_quotas(
            train_counts=[9, 1],
            caps=[5, 100],
            target_size=10,
        )

        self.assertEqual(quotas, [5, 5])


class KNPartitionTests(unittest.TestCase):
    def test_builds_disjoint_n_way_k_shot_subsets(self) -> None:
        dataset = FakeDataset(num_classes=5, samples_per_class=100)
        subsets = make_kn_client_subsets(
            dataset,
            num_classes=5,
            num_clients=3,
            ways=3,
            shots=10,
            stdev=2,
            train_shots_max=15,
            seed=1234,
        )

        all_indices = [index for subset in subsets for index in subset.indices]
        self.assertEqual(len(all_indices), len(set(all_indices)))
        for subset in subsets:
            labels = {dataset.targets[index] for index in subset.indices}
            self.assertGreaterEqual(len(labels), 2)
            self.assertLessEqual(len(labels), 5)


class DirichletPartitionTests(unittest.TestCase):
    def test_keeps_balanced_sample_pool_fixed_across_alpha(self) -> None:
        dataset = FakeDataset(num_classes=10, samples_per_class=100)
        partitions = [
            make_dirichlet_client_subsets(
                dataset,
                num_classes=10,
                num_clients=10,
                samples_per_client=50,
                alpha=alpha,
                seed=1234,
            )
            for alpha in (10.0, 1.0, 0.2, 0.1)
        ]

        expected_pool = {
            index
            for subset in partitions[0]
            for index in subset.indices
        }
        for subsets in partitions:
            all_indices = [index for subset in subsets for index in subset.indices]
            histogram = Counter(dataset.targets[index] for index in all_indices)

            self.assertTrue(all(len(subset) == 50 for subset in subsets))
            self.assertEqual(len(all_indices), len(set(all_indices)))
            self.assertEqual(set(all_indices), expected_pool)
            self.assertEqual([histogram[label] for label in range(10)], [50] * 10)

    def test_is_deterministic_for_same_seed_and_alpha(self) -> None:
        dataset = FakeDataset(num_classes=5, samples_per_class=100)
        kwargs = {
            "dataset": dataset,
            "num_classes": 5,
            "num_clients": 5,
            "samples_per_client": 40,
            "alpha": 0.2,
            "seed": 2026,
        }

        first = make_dirichlet_client_subsets(**kwargs)
        second = make_dirichlet_client_subsets(**kwargs)

        self.assertEqual(
            [subset.indices for subset in first],
            [subset.indices for subset in second],
        )


if __name__ == "__main__":
    unittest.main()
