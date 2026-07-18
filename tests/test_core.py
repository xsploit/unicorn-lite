from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from unicorn_lite.core import NeuralCore
from unicorn_lite.encoder import HashEncoder
from unicorn_lite.store import MemoryStore


class CoreTests(unittest.TestCase):
    def test_fast_memory_and_state_change(self) -> None:
        encoder = HashEncoder()
        core = NeuralCore(device="cpu")
        before_hidden, before_fast, before_count = core.export_state()
        output = core.process(encoder.encode("a persistent experience"))
        after_hidden, after_fast, after_count = core.export_state()
        self.assertEqual(before_count, 0)
        self.assertEqual(after_count, 1)
        self.assertFalse(np.allclose(before_hidden, after_hidden))
        self.assertFalse(np.allclose(before_fast, after_fast))
        self.assertGreater(output.memory_norm, 0.0)

    def test_state_import_roundtrip(self) -> None:
        encoder = HashEncoder()
        first = NeuralCore(device="cpu")
        first.process(encoder.encode("remember this"))
        state = first.export_state()
        second = NeuralCore(device="cpu")
        second.import_state(*state)
        restored = second.export_state()
        np.testing.assert_allclose(restored[0], state[0])
        np.testing.assert_allclose(restored[1], state[1])
        self.assertEqual(restored[2], state[2])

    def test_step_exposes_current_and_next_predictions(self) -> None:
        encoder = HashEncoder()
        core = NeuralCore(device="cpu")
        item = torch.from_numpy(encoder.encode("current experience"))
        hidden = torch.zeros(core.latent_dim)
        fast = torch.zeros(core.latent_dim, core.latent_dim)
        result = core.model.step(item, hidden, fast)
        self.assertEqual(len(result), 7)
        current_prediction = result[1]
        next_prediction = result[2]
        self.assertFalse(torch.allclose(current_prediction, next_prediction))

    def test_state_is_scoped_to_controller_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            hidden = np.zeros(96, dtype=np.float32)
            fast = np.zeros((96, 96), dtype=np.float32)
            store.save_core_state(hidden, fast, 3, "event", model_id="model-a")
            self.assertIsNotNone(store.load_core_state("model-a"))
            self.assertIsNone(store.load_core_state("model-b"))
            store.close()


if __name__ == "__main__":
    unittest.main()
