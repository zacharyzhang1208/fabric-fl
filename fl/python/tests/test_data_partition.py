from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import distribution_matched_quotas


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


if __name__ == "__main__":
    unittest.main()
