from __future__ import annotations

import unittest

import numpy as np
import torch

from unicorn_lite.memory_reranker import (
    AnswerAwareMemoryReranker,
    reranker_from_checkpoint,
)


class MemoryRerankerTests(unittest.TestCase):
    def test_selector_returns_normalized_weights_and_ranking(self) -> None:
        torch.manual_seed(2)
        model = AnswerAwareMemoryReranker(embedding_dim=8, hidden_dim=6)
        query = torch.randn(2, 8)
        candidates = torch.randn(2, 4, 8)
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        output = model(query, candidates, mask)
        self.assertEqual(tuple(output.scores.shape), (2, 4))
        self.assertTrue(torch.allclose(output.weights.sum(dim=1), torch.ones(2)))
        self.assertEqual(float(output.weights[0, 2].detach()), 0.0)

        order, scores = model.rank(
            query[0].detach().numpy(), candidates[0].detach().numpy()
        )
        self.assertEqual(set(order.tolist()), {0, 1, 2, 3})
        self.assertEqual(scores.shape, (4,))

    def test_reply_supervision_reaches_selector(self) -> None:
        torch.manual_seed(4)
        model = AnswerAwareMemoryReranker(embedding_dim=8, hidden_dim=6)
        query = torch.randn(3, 8)
        candidates = torch.randn(3, 4, 8)
        replies = torch.randn(3, 8)
        loss, _ = model.training_loss(query, candidates, replies)
        loss.backward()
        gradient = model.selector[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.0)

    def test_checkpoint_roundtrip(self) -> None:
        model = AnswerAwareMemoryReranker(embedding_dim=8, hidden_dim=6)
        restored = reranker_from_checkpoint(model.checkpoint(accepted=False))
        self.assertEqual(restored.embedding_dim, 8)
        self.assertFalse(restored.training)


if __name__ == "__main__":
    unittest.main()
