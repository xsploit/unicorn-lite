from __future__ import annotations

import unittest

import torch

from unicorn_lite.response_gate import (
    PonderResponseGate,
    ResponseGate,
    response_gate_from_checkpoint,
)


class PonderGateTests(unittest.TestCase):
    def test_halt_weights_form_probability_distribution(self) -> None:
        probabilities = torch.tensor([0.2, 0.4, 0.7, 0.1])
        weights = PonderResponseGate._halt_weights(probabilities)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue(bool(torch.all(weights >= 0)))

    def test_adaptive_gate_respects_learned_halt(self) -> None:
        model = PonderResponseGate(embedding_dim=8, hidden_dim=4, feature_dim=3)
        model.halt_head.weight.data.zero_()
        model.halt_head.bias.data.fill_(10.0)
        output = model.ponder(
            torch.zeros(8),
            torch.zeros(3),
            torch.zeros(4),
            adaptive=True,
        )
        self.assertEqual(output.steps, 1)
        self.assertEqual(len(output.probability_path), 1)

    def test_forced_depth_uses_full_recursive_budget(self) -> None:
        model = PonderResponseGate(
            embedding_dim=8, hidden_dim=4, feature_dim=3, max_steps=5
        )
        output = model.ponder(
            torch.zeros(8),
            torch.zeros(3),
            torch.zeros(4),
            adaptive=False,
        )
        self.assertEqual(output.steps, 5)
        self.assertAlmostEqual(float(output.halt_weights.sum()), 1.0, places=6)

    def test_checkpoint_factory_preserves_legacy_and_ponder_models(self) -> None:
        legacy = ResponseGate(8, 4, 3)
        loaded_legacy = response_gate_from_checkpoint(
            {
                "model": legacy.state_dict(),
                "embedding_dim": 8,
                "hidden_dim": 4,
                "feature_dim": 3,
            }
        )
        self.assertIs(type(loaded_legacy), ResponseGate)

        ponder = PonderResponseGate(8, 4, 3, max_steps=6, min_steps=2)
        loaded_ponder = response_gate_from_checkpoint(
            {
                "architecture": "ponder_v1",
                "model": ponder.state_dict(),
                "embedding_dim": 8,
                "hidden_dim": 4,
                "feature_dim": 3,
                "ponder": {"max_steps": 6, "min_steps": 2, "halt_threshold": 0.6},
            }
        )
        self.assertIsInstance(loaded_ponder, PonderResponseGate)
        self.assertEqual(loaded_ponder.max_steps, 6)
        self.assertEqual(loaded_ponder.min_steps, 2)


if __name__ == "__main__":
    unittest.main()
