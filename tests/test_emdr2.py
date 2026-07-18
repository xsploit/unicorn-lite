from __future__ import annotations

import unittest

import torch

from unicorn_lite.emdr2 import emdr2_retriever_loss, reader_document


class EMDR2Tests(unittest.TestCase):
    def test_retriever_objective_stops_reader_gradient(self) -> None:
        scores = torch.tensor([[0.2, 0.8]], requires_grad=True)
        likelihood = torch.tensor([[-4.0, -1.0]], requires_grad=True)
        loss = emdr2_retriever_loss(scores, likelihood)
        loss.backward()
        self.assertIsNotNone(scores.grad)
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)
        self.assertIsNone(likelihood.grad)

    def test_reader_document_preserves_query_actor_and_memory(self) -> None:
        value = reader_document("where were we?", "Subsect", "training Neuro")
        self.assertIn("question: where were we?", value)
        self.assertIn("context from Subsect: training Neuro", value)


if __name__ == "__main__":
    unittest.main()
