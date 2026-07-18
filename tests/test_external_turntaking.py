from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from unicorn_lite.external_turntaking import convert_ishiki, convert_when2speak


class ExternalTurnTakingTests(unittest.TestCase):
    def test_ishiki_conversion_preserves_implicit_speech(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "output.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "decision_point_id": "d1",
                        "sequence_id": "s1",
                        "target_speaker": "D",
                        "context_turns": [{"speaker": "A", "text": "hello"}],
                        "current_turn": {"speaker": "C", "text": "thoughts?"},
                        "target_is_addressed": False,
                        "decision": "SPEAK",
                        "category": "SPEAK_implicit",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = convert_ishiki([source], output)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["reply_examples"], 1)
            self.assertEqual(row["action"], "REPLY_FLASH")
            self.assertFalse(row["direct"])

    def test_when2speak_conversion_maps_silent_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "output.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Speaker_0: hello"},
                            {"role": "assistant", "content": ">"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = convert_when2speak(source, output)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["observe_examples"], 1)
            self.assertEqual(row["action"], "OBSERVE")


if __name__ == "__main__":
    unittest.main()
