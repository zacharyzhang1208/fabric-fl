from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.prototype_head import run_prototype_head
from fl_client import FederatedClient


class PrototypeHeadTests(unittest.TestCase):
    def test_local_class_restriction_and_prototype_inference(self) -> None:
        client = self._client(0, "cnn", samples=2)
        client.model = FixedPredictionModel()
        loader = DataLoader(
            TensorDataset(torch.zeros(1, 1, 28, 28), torch.tensor([7])),
            batch_size=1,
        )
        global_prototypes = torch.zeros(10, 2)
        global_prototypes[2] = 1.0
        global_prototypes[3] = 2.0
        global_counts = torch.zeros(10)
        global_counts[[2, 3, 7]] = 1

        self.assertEqual(client.evaluate(loader), 0.0)
        self.assertEqual(client.evaluate(loader, allowed_classes={2, 3, 7}), 1.0)
        self.assertEqual(
            client.evaluate_with_prototypes(
                loader,
                global_prototypes,
                global_counts,
                allowed_classes={2, 3, 7},
            ),
            1.0,
        )

    def test_heterogeneous_clients_share_and_synchronize_classifier(self) -> None:
        clients = [
            self._client(client_id, model_name, samples=client_id + 2)
            for client_id, model_name in enumerate(("mlp", "cnn", "mini_resnet"))
        ]
        reference = clients[0].get_classifier_state()

        self.assertEqual(
            {name: tuple(tensor.shape) for name, tensor in reference.items()},
            {"weight": (10, 50), "bias": (10,)},
        )

        args = SimpleNamespace(
            rounds=1,
            local_epochs=1,
            proto_weight=0.5,
            model_config="heterogeneous",
            num_clients=3,
        )
        loaders = {"local": [client.prototype_loader for client in clients]}
        with contextlib.redirect_stdout(io.StringIO()):
            communication = run_prototype_head(
                args=args,
                clients=clients,
                eval_loaders=loaders,
                evaluation_clients=[0, 1, 2],
                device=torch.device("cpu"),
                num_classes=10,
                client_label_sets=[
                    set(range(client_id + 2))
                    for client_id in range(3)
                ],
            )

        final_states = [client.get_classifier_state() for client in clients]
        for state in final_states[1:]:
            self.assertTrue(torch.equal(state["weight"], final_states[0]["weight"]))
            self.assertTrue(torch.equal(state["bias"], final_states[0]["bias"]))

        prototype_bytes = (10 * 50 + 10) * 4
        classifier_bytes = (10 * 50 + 10) * 4
        self.assertEqual(communication, 3 * (prototype_bytes + classifier_bytes))

    @staticmethod
    def _client(client_id: int, model_name: str, samples: int) -> FederatedClient:
        images = torch.randn(samples, 1, 28, 28)
        labels = torch.arange(samples) % 10
        loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
        return FederatedClient(
            client_id=client_id,
            train_loader=loader,
            prototype_loader=loader,
            device=torch.device("cpu"),
            lr=0.01,
            input_shape=(1, 28, 28),
            num_classes=10,
            dataset_name="mnist",
            optimizer_name="sgd",
            model_name=model_name,
        )


class FixedPredictionModel(nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(images.size(0), 10)
        logits[:, 7] = 2.0
        logits[:, 9] = 3.0
        embeddings = torch.zeros(images.size(0), 2)
        return F.log_softmax(logits, dim=1), embeddings


if __name__ == "__main__":
    unittest.main()
