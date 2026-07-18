from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_refractory_pairs import build_pairs


class RefractoryPairTests(unittest.TestCase):
    def test_repeat_negatives_are_balanced_by_followup_positives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "policy.jsonl"
            output = Path(directory) / "pairs.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "occurred_at": "2026-01-01T00:00:00+00:00",
                        "content": "What would you remember about this conversation?",
                        "action": "REPLY_FLASH",
                        "direct": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = build_pairs(
                source, output, "2026-01-15T00:00:00+00:00", pair_count=1
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["training_rows"], 4)
            self.assertEqual(
                [row["action"] for row in rows].count("REPLY_FLASH"), 3
            )
            repeat = next(
                row
                for row in rows
                if row["label_source"] == "training_only_refractory_repeat"
            )
            followup = next(
                row
                for row in rows
                if row["label_source"] == "training_only_refractory_followup"
            )
            self.assertTrue(repeat["recent_answered_near_duplicate"])
            self.assertFalse(followup["recent_answered_near_duplicate"])


if __name__ == "__main__":
    unittest.main()
