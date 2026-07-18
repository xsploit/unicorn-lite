from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from unicorn_lite.encoder import HashEncoder
from unicorn_lite.training import train_replay


class TrainingTests(unittest.TestCase):
    def test_replay_training_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = root / "replay.jsonl"
            checkpoint = root / "controller.pt"
            rows = [
                {"content": "background chatter", "action": "OBSERVE"},
                {"content": "can you help me", "action": "REPLY_FLASH"},
                {"content": "analyze the architecture", "action": "REPLY_PRO"},
            ]
            replay.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            report = train_replay(
                replay, HashEncoder(), checkpoint, device="cpu", epochs=1
            )
            self.assertTrue(checkpoint.exists())
            self.assertEqual(report["records"], 3)
            self.assertGreaterEqual(float(report["loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
