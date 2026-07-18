from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from unicorn_lite.attribution import build_contextual_policy_dataset
from unicorn_lite.encoder import HashEncoder


def _row(
    message_id: str,
    occurred_at: str,
    author_id: str,
    content: str,
    *,
    bot: bool = False,
    mentions: list[str] | None = None,
) -> dict[str, object]:
    return {
        "message_id": message_id,
        "occurred_at": occurred_at,
        "author_id": author_id,
        "author_is_bot": bot,
        "content": content,
        "channel_id": "channel",
        "mentioned_user_ids": mentions or [],
        "reply_to_message_id": None,
    }


class AttributionTests(unittest.TestCase):
    def _build(self, records: list[dict[str, object]], lookback_hours: float = 24.0):
        root = Path(self.directory.name)
        history = root / "history.jsonl"
        output = root / "policy.jsonl"
        history.write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
        report = build_contextual_policy_dataset(
            [history], "agent", HashEncoder(), output, lookback_hours=lookback_hours
        )
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        return report, rows

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_response_history_is_not_visible_before_response_occurs(self) -> None:
        _, rows = self._build(
            [
                _row("1", "2026-01-01T00:00:00+00:00", "alice", "agent hello", mentions=["agent"]),
                _row("2", "2026-01-01T00:00:05+00:00", "alice", "still waiting"),
                _row("3", "2026-01-01T00:00:10+00:00", "agent", "hello alice", bot=True, mentions=["alice"]),
                _row("4", "2026-01-01T00:00:20+00:00", "alice", "one more thing"),
            ]
        )
        by_id = {row["discord_message_id"]: row for row in rows}
        self.assertEqual(by_id["2"]["author_replies_seen"], 0)
        self.assertEqual(by_id["4"]["author_replies_seen"], 1)

    def test_named_response_never_falls_back_to_different_adjacent_user(self) -> None:
        report, rows = self._build(
            [
                _row("1", "2026-01-01T00:00:00+00:00", "alice", "old question"),
                _row("2", "2026-01-01T00:09:55+00:00", "bob", "recent unrelated question?"),
                _row("3", "2026-01-01T00:10:00+00:00", "agent", "alice response", bot=True, mentions=["alice"]),
            ],
            lookback_hours=0.001,
        )
        by_id = {row["discord_message_id"]: row for row in rows}
        self.assertEqual(by_id["2"]["action"], "OBSERVE")
        self.assertEqual(report["reply_examples"], 0)


if __name__ == "__main__":
    unittest.main()
