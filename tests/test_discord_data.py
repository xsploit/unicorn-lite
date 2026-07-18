from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from unicorn_lite.discord_data import (
    _read_existing,
    _write_records,
    build_discord_labels,
)


class DiscordDataTests(unittest.TestCase):
    def test_records_are_deduplicated_and_written_chronologically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            records = {
                "2": {"message_id": "2", "occurred_at": "2026-01-02T00:00:00Z"},
                "1": {"message_id": "1", "occurred_at": "2026-01-01T00:00:00Z"},
            }
            _write_records(path, records)
            loaded = _read_existing(path)
            self.assertEqual(set(loaded), {"1", "2"})
            self.assertTrue(path.read_text(encoding="utf-8").startswith('{"message_id": "1"'))

    def test_explicit_reply_becomes_positive_policy_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.jsonl"
            output = Path(directory) / "labels.jsonl"
            records = {
                "1": {
                    "message_id": "1",
                    "occurred_at": "2026-01-01T00:00:00+00:00",
                    "author_id": "human",
                    "author_is_bot": False,
                    "content": "hello",
                    "channel_id": "channel",
                    "mentioned_user_ids": ["agent"],
                    "reply_to_message_id": None,
                },
                "2": {
                    "message_id": "2",
                    "occurred_at": "2026-01-01T00:00:05+00:00",
                    "author_id": "agent",
                    "author_is_bot": True,
                    "content": "hi",
                    "channel_id": "channel",
                    "mentioned_user_ids": [],
                    "reply_to_message_id": "1",
                },
            }
            _write_records(history, records)
            report = build_discord_labels(history, "agent", output)
            self.assertEqual(report["explicit_response_examples"], 1)
            label = __import__("json").loads(output.read_text(encoding="utf-8"))
            self.assertEqual(label["action"], "REPLY_FLASH")
            self.assertEqual(label["expected_reply"], "hi")


if __name__ == "__main__":
    unittest.main()
