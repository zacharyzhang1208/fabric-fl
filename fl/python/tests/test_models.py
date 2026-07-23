from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import MODEL_NAMES, build_model, model_name_for_client


class HeterogeneousModelTests(unittest.TestCase):
    def test_all_models_share_only_the_output_contract(self) -> None:
        images = torch.randn(4, 1, 28, 28)
        state_shapes = []

        for model_name in MODEL_NAMES:
            model = build_model(
                dataset_name="mnist",
                input_shape=(1, 28, 28),
                num_classes=10,
                model_name=model_name,
            )
            log_probs, prototypes = model(images)

            self.assertEqual(log_probs.shape, (4, 10))
            self.assertEqual(prototypes.shape, (4, 50))
            self.assertEqual(model.prototype_dim, 50)
            state_shapes.append(
                [(name, tuple(tensor.shape)) for name, tensor in model.state_dict().items()]
            )

        self.assertEqual(len({repr(shapes) for shapes in state_shapes}), len(MODEL_NAMES))

    def test_twenty_clients_are_split_seven_seven_six(self) -> None:
        assignments = [
            model_name_for_client(client_id, 20, "heterogeneous")
            for client_id in range(20)
        ]

        self.assertEqual(assignments[:7], ["mlp"] * 7)
        self.assertEqual(assignments[7:14], ["cnn"] * 7)
        self.assertEqual(assignments[14:], ["mini_resnet"] * 6)

    def test_homogeneous_config_uses_cnn(self) -> None:
        self.assertEqual(
            [model_name_for_client(client_id, 4, "homogeneous") for client_id in range(4)],
            ["cnn"] * 4,
        )


if __name__ == "__main__":
    unittest.main()
