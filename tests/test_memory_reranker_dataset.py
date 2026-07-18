from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_memory_reranker_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_memory_reranker_dataset", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MemoryRerankerDatasetTests(unittest.TestCase):
    def test_groups_are_strictly_causal(self) -> None:
        rows = []
        for index in range(6):
            rows.append(
                {
                    "message_id": str(index),
                    "occurred_at": f"2026-01-01T00:00:0{index}+00:00",
                    "channel_id": "channel",
                    "author_id": "human",
                    "content": f"prior {index}",
                }
            )
        rows.append(
            {
                "message_id": "query",
                "occurred_at": "2026-01-01T00:00:06+00:00",
                "channel_id": "channel",
                "author_id": "human",
                "content": "current question",
            }
        )
        rows.append(
            {
                "message_id": "reply",
                "occurred_at": "2026-01-01T00:00:07+00:00",
                "channel_id": "channel",
                "author_id": "agent",
                "content": "current answer",
                "reply_to_message_id": "query",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "history.jsonl"
            destination = Path(directory) / "groups.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = MODULE.build_groups(
                [source], "agent", destination, minimum_candidates=4
            )
            self.assertEqual(report["training_groups"], 1)
            group = json.loads(destination.read_text(encoding="utf-8"))
            ids = {candidate["event_id"] for candidate in group["candidates"]}
            self.assertNotIn("query", ids)
            self.assertNotIn("reply", ids)
            self.assertEqual(len(ids), 6)


if __name__ == "__main__":
    unittest.main()
