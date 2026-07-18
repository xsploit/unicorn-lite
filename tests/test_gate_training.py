from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from unicorn_lite.gate_training import (
    _expand_feature_checkpoint,
    _load_hard_negative_rows,
    _load_positive_anchor_rows,
    _load_refractory_rows,
)
from unicorn_lite.response_gate import PonderResponseGate
import torch


class HardNegativeSelectionTests(unittest.TestCase):
    def test_feature_expansion_preserves_old_columns_and_zeros_new_ones(self) -> None:
        old = PonderResponseGate(embedding_dim=8, hidden_dim=4, feature_dim=3)
        new = PonderResponseGate(embedding_dim=8, hidden_dim=4, feature_dim=5)
        expanded = _expand_feature_checkpoint(new, old.state_dict(), 3)
        self.assertTrue(
            torch.equal(expanded["feature_project.0.weight"][:, :3], old.state_dict()["feature_project.0.weight"])
        )
        self.assertEqual(
            int(torch.count_nonzero(expanded["feature_project.0.weight"][:, 3:])), 0
        )
        self.assertTrue(
            torch.equal(expanded["halt_head.weight"][:, :4], old.state_dict()["halt_head.weight"][:, :4])
        )
        self.assertEqual(
            int(torch.count_nonzero(expanded["halt_head.weight"][:, 7:])), 0
        )

    def test_only_pre_cutoff_direct_silence_is_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl"
            rows = [
                {
                    "discord_message_id": "valid",
                    "occurred_at": "2026-01-01T00:00:00+00:00",
                    "content": "@agent sup",
                    "direct": True,
                    "channel": "general",
                    "action": "OBSERVE",
                },
                {
                    "occurred_at": "2026-01-01T00:00:01+00:00",
                    "content": "background",
                    "direct": False,
                    "action": "OBSERVE",
                },
                {
                    "occurred_at": "2026-01-01T00:00:02+00:00",
                    "content": "answered",
                    "direct": True,
                    "action": "REPLY_FLASH",
                },
                {
                    "occurred_at": "2026-01-01T00:00:03+00:00",
                    "content": "what is your purpose?",
                    "direct": True,
                    "action": "OBSERVE",
                },
                {
                    "occurred_at": "2026-01-01T00:00:04+00:00",
                    "content": "this direct message has too many words",
                    "direct": True,
                    "action": "OBSERVE",
                },
                {
                    "occurred_at": "2026-01-01T00:00:05+00:00",
                    "content": "explain recursion",
                    "direct": True,
                    "action": "OBSERVE",
                },
                {
                    "occurred_at": "2026-02-01T00:00:00+00:00",
                    "content": "future silence",
                    "direct": True,
                    "action": "OBSERVE",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            selected, report = _load_hard_negative_rows(
                [path],
                "2026-01-15T00:00:00+00:00",
                max_words=4,
                exclude_questions=True,
                exclude_requests=True,
            )

            self.assertEqual([row["content"] for row in selected], ["@agent sup"])
            self.assertEqual(report["selected"], 1)
            self.assertEqual(report["after_cutoff"], 1)
            self.assertEqual(report["not_direct"], 2)
            self.assertEqual(report["question"], 1)
            self.assertEqual(report["too_long"], 1)
            self.assertEqual(report["request"], 1)
            self.assertTrue(str(selected[0]["channel"]).startswith("hard-negative:"))

    def test_positive_anchors_are_concise_direct_requests_before_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl"
            rows = [
                {
                    "discord_message_id": "question",
                    "occurred_at": "2026-01-01T00:00:00+00:00",
                    "content": "what do you think?",
                    "direct": True,
                    "channel": "general",
                    "action": "REPLY_FLASH",
                },
                {
                    "occurred_at": "2026-01-01T00:00:01+00:00",
                    "content": "hello",
                    "direct": True,
                    "action": "REPLY_FLASH",
                },
                {
                    "occurred_at": "2026-01-01T00:00:02+00:00",
                    "content": "what do you think?",
                    "direct": True,
                    "action": "OBSERVE",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            selected, report = _load_positive_anchor_rows(
                [path], "2026-01-15T00:00:00+00:00", limit=1
            )

            self.assertEqual([row["content"] for row in selected], ["what do you think?"])
            self.assertEqual(report["candidates"], 1)
            self.assertEqual(report["selected"], 1)
            self.assertTrue(str(selected[0]["channel"]).startswith("positive-anchor:"))

    def test_refractory_loader_requires_training_only_pair_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "refractory.jsonl"
            rows = [
                {
                    "occurred_at": "2026-01-01T00:00:00+00:00",
                    "content": "What do you think?",
                    "action": "REPLY_FLASH",
                    "refractory_pair_id": "pair-1",
                    "label_source": "training_only_refractory_seed",
                },
                {
                    "occurred_at": "2026-01-01T00:01:00+00:00",
                    "content": "Can you answer that again?",
                    "action": "OBSERVE",
                    "refractory_pair_id": "pair-1",
                    "label_source": "training_only_refractory_repeat",
                },
                {
                    "occurred_at": "2026-01-01T00:02:00+00:00",
                    "content": "unmarked row",
                    "action": "OBSERVE",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            selected, report = _load_refractory_rows(
                [path], "2026-01-15T00:00:00+00:00"
            )
            self.assertEqual(len(selected), 2)
            self.assertEqual(report["pairs"], 1)
            self.assertEqual(report["invalid"], 1)
            self.assertTrue(str(selected[0]["channel"]).startswith("refractory:"))


if __name__ == "__main__":
    unittest.main()
