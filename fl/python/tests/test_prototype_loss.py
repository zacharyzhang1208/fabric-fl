from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fl_client import cosine_similarity_logits, prototype_classification_loss


class PrototypeClassificationLossTests(unittest.TestCase):
    def test_prefers_the_correct_most_similar_prototype(self) -> None:
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        counts = torch.tensor([1.0, 1.0])
        labels = torch.tensor([0])

        correct_nearest = prototype_classification_loss(
            embeddings=torch.tensor([[0.9, 0.1]]),
            labels=labels,
            global_prototypes=prototypes,
            global_counts=counts,
            allowed_classes={0, 1},
            temperature=0.1,
        )
        wrong_nearest = prototype_classification_loss(
            embeddings=torch.tensor([[0.1, 0.9]]),
            labels=labels,
            global_prototypes=prototypes,
            global_counts=counts,
            allowed_classes={0, 1},
            temperature=0.1,
        )

        self.assertLess(correct_nearest.item(), wrong_nearest.item())

    def test_excludes_unavailable_and_disallowed_prototypes(self) -> None:
        embeddings = torch.tensor([[1.0, 0.0]], requires_grad=True)
        loss = prototype_classification_loss(
            embeddings=embeddings,
            labels=torch.tensor([0]),
            global_prototypes=torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
            ),
            global_counts=torch.tensor([1.0, 0.0, 1.0]),
            allowed_classes={0, 1},
            temperature=0.5,
        )

        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_cosine_logits_are_invariant_to_vector_scale(self) -> None:
        embeddings = torch.tensor([[1.0, 1.0]])
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

        original = cosine_similarity_logits(embeddings, prototypes, temperature=0.1)
        scaled = cosine_similarity_logits(
            embeddings * 7.0,
            prototypes * torch.tensor([[3.0], [5.0]]),
            temperature=0.1,
        )

        self.assertTrue(torch.allclose(original, scaled))


if __name__ == "__main__":
    unittest.main()
