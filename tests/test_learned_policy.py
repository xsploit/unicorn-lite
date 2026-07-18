from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from unicorn_lite.core import CoreOutput
from unicorn_lite.encoder import HashEncoder
from unicorn_lite.learned_policy import LearnedResponsePolicy
from unicorn_lite.models import Experience
from unicorn_lite.response_gate import ResponseGate
from unicorn_lite.store import MemoryStore


class LearnedPolicyTests(unittest.TestCase):
    def _checkpoint(self, path: Path) -> None:
        model = ResponseGate(384, 64)
        for parameter in model.parameters():
            parameter.data.zero_()
        torch.save(
            {
                "model": model.state_dict(),
                "embedding_dim": 384,
                "hidden_dim": 64,
                "threshold": 0.6,
                "acceptance_passed": True,
                "policy_mode": "direct_obligation_plus_learned_initiative",
            },
            path,
        )

    def test_direct_message_does_not_bypass_learned_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "gate.pt"
            self._checkpoint(checkpoint)
            store = MemoryStore(root / "memory.db")
            encoder = HashEncoder()
            core = CoreOutput(0.5, np.zeros(3), np.array([0, 0, 1, 0]), 0, 0)
            policy = LearnedResponsePolicy(checkpoint, encoder, store)

            direct = policy.decide(
                Experience("hello", channel="c", metadata={"mentioned": True}),
                core,
                [],
            )
            quiet = policy.decide(
                Experience("background", channel="c", metadata={}), core, []
            )
            self.assertEqual(direct.action, "OBSERVE")
            self.assertEqual(quiet.action, "OBSERVE")

            reloaded = LearnedResponsePolicy(checkpoint, encoder, store)
            reloaded.decide(
                Experience("more background", channel="c", metadata={}), core, []
            )
            self.assertEqual(reloaded.messages_seen["c"], 3)
            store.close()

    def test_rejected_checkpoint_cannot_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "gate.pt"
            model = ResponseGate(384, 64)
            torch.save(
                {
                    "model": model.state_dict(),
                    "embedding_dim": 384,
                    "hidden_dim": 64,
                    "threshold": 0.5,
                    "acceptance_passed": False,
                },
                checkpoint,
            )
            store = MemoryStore(root / "memory.db")
            with self.assertRaisesRegex(RuntimeError, "acceptance"):
                LearnedResponsePolicy(checkpoint, HashEncoder(), store)
            store.close()

    def test_channel_threshold_override_changes_learned_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "gate.pt"
            self._checkpoint(checkpoint)
            store = MemoryStore(root / "memory.db")
            core = CoreOutput(0.5, np.zeros(3), np.array([0, 0, 1, 0]), 0, 0)
            policy = LearnedResponsePolicy(
                checkpoint,
                HashEncoder(),
                store,
                threshold_overrides={"general": 0.4},
            )
            decision = policy.decide(
                Experience("hello", channel="general", metadata={}), core, []
            )
            self.assertEqual(policy.threshold_for("general"), 0.4)
            self.assertEqual(policy.threshold_for("other"), 0.6)
            self.assertTrue(decision.action.startswith("REPLY"))
            store.close()


if __name__ == "__main__":
    unittest.main()
