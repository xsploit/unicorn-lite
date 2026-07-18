from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from unicorn_lite.discord_runtime import decision_audit_payload, is_lexically_addressed
from unicorn_lite.models import Decision, Experience, MemoryHit


class DiscordRuntimeTests(unittest.TestCase):
    def test_opening_vocative_is_detected(self) -> None:
        aliases = {"Neuro-sama", "neuro"}
        self.assertTrue(is_lexically_addressed("hey neuro i retrained you", aliases))
        self.assertTrue(is_lexically_addressed("Neuro-sama, thoughts?", aliases))

    def test_incidental_name_is_not_an_address(self) -> None:
        aliases = {"neuro"}
        self.assertFalse(is_lexically_addressed("I was talking about neuro earlier", aliases))
        self.assertFalse(is_lexically_addressed("neurology is interesting", aliases))

    def test_audit_identifies_memory_mode_without_serializing_embeddings(self) -> None:
        policy = SimpleNamespace(threshold_for=lambda _: 0.28)
        agent = SimpleNamespace(
            policy=policy,
            writer=None,
            memory_reranker=object(),
            memory_candidate_limit=20,
        )
        event = Experience(
            "test",
            metadata={"memory_reranker": {"applied": True}},
        )
        memory = MemoryHit(
            event_id="memory",
            content="prior text",
            actor="user",
            similarity=0.7,
            occurred_at="2026-01-01T00:00:00+00:00",
            embedding=np.ones(384, dtype=np.float32),
        )
        result = SimpleNamespace(
            decision=Decision("REPLY_FLASH", 2, 0.7, "test"),
            surprise=0.2,
            reply="reply",
            writer_error=None,
            memories=[memory],
            facts_updated=[],
        )
        payload = decision_audit_payload(
            agent,
            event,
            result,
            writer_allowed=True,
            discord_reply_sent=True,
            discord_send_error=None,
            jump_url="https://example.test",
        )
        self.assertEqual(payload["runtime"]["audit_schema_version"], 2)
        self.assertEqual(payload["runtime"]["memory_selector"], "answer-aware-v1")
        self.assertNotIn("embedding", payload["retrieved_memories"][0])


if __name__ == "__main__":
    unittest.main()
