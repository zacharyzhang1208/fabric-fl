from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype_synthesis import synthesize_prototype_images, total_variation


class ToyFeatureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = images.flatten(start_dim=1)[:, :2] * self.scale
        return F.log_softmax(embeddings, dim=1), embeddings


class PrototypeSynthesisTests(unittest.TestCase):
    def test_total_variation_is_zero_for_constant_images(self) -> None:
        images = torch.ones(3, 1, 4, 4)
        self.assertEqual(total_variation(images).item(), 0.0)

    def test_synthesizes_only_rare_active_classes_without_updating_model(self) -> None:
        model = ToyFeatureModel()
        model.train()
        original_scale = model.scale.detach().clone()

        result = synthesize_prototype_images(
            model=model,
            global_prototypes=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            global_counts=torch.tensor([10.0, 10.0]),
            class_counts=[0, 20],
            input_shape=(1, 2, 2),
            normalization_mean=(0.0,),
            normalization_std=(1.0,),
            target_count=10,
            samples_per_class=3,
            steps=80,
            learning_rate=0.2,
            temperature=0.1,
            min_margin=0.1,
            tv_weight=0.0,
            seed=1234,
        )

        self.assertEqual(result.attempted, 3)
        self.assertEqual(result.accepted, 3)
        self.assertEqual(result.classes, [0])
        self.assertTrue(torch.equal(result.labels, torch.zeros(3, dtype=torch.long)))
        self.assertTrue(torch.equal(model.scale.detach(), original_scale))
        self.assertTrue(model.training)
        self.assertTrue(model.scale.requires_grad)

    def test_skips_rare_class_without_a_global_prototype(self) -> None:
        result = synthesize_prototype_images(
            model=ToyFeatureModel(),
            global_prototypes=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            global_counts=torch.tensor([0.0, 10.0]),
            class_counts=[0, 20],
            input_shape=(1, 2, 2),
            normalization_mean=(0.0,),
            normalization_std=(1.0,),
            target_count=10,
            samples_per_class=3,
            steps=1,
            learning_rate=0.1,
            temperature=0.1,
            min_margin=0.1,
            tv_weight=0.0,
            seed=1234,
        )

        self.assertEqual(result.attempted, 0)
        self.assertEqual(result.accepted, 0)
        self.assertEqual(result.images.shape, (0, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
