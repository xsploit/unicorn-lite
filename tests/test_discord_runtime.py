from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from unicorn_lite.discord_runtime import (
    audit_delivery_mode,
    decision_audit_payload,
    is_alias_role_mentioned,
    is_lexically_addressed,
    is_speech_eligible,
    is_trusted_bot,
    is_trusted_webhook,
)
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

    def test_same_named_discord_role_is_an_address_signal(self) -> None:
        aliases = {"Neuro-sama", "neuro"}
        self.assertTrue(is_alias_role_mentioned(["Neuro-sama"], aliases))
        self.assertTrue(is_alias_role_mentioned(["neuro sama"], aliases))
        self.assertFalse(is_alias_role_mentioned(["AI bots"], aliases))

    def test_only_explicitly_allowlisted_webhook_is_trusted(self) -> None:
        trusted = {"webhook-1"}
        self.assertTrue(is_trusted_webhook("webhook-1", trusted))
        self.assertFalse(is_trusted_webhook("webhook-2", trusted))
        self.assertFalse(is_trusted_webhook(None, trusted))

    def test_only_explicitly_allowlisted_bot_is_trusted(self) -> None:
        trusted = {"bot-1"}
        self.assertTrue(is_trusted_bot("bot-1", trusted))
        self.assertFalse(is_trusted_bot("bot-2", trusted))

    def test_directed_messages_are_eligible_in_any_channel(self) -> None:
        common = {
            "mode": "active",
            "is_bot": False,
            "direct": False,
            "allow_unsolicited": False,
            "channel": "outside-allowlist",
            "unsolicited_channel_ids": {"general"},
        }
        for signal in ("mentioned", "lexically_addressed", "replied_to_target"):
            values = {
                "mentioned": False,
                "lexically_addressed": False,
                "replied_to_target": False,
                **common,
            }
            values[signal] = True
            self.assertTrue(is_speech_eligible(**values), signal)

    def test_ordinary_messages_require_unsolicited_allowlist(self) -> None:
        common = {
            "mode": "active",
            "is_bot": False,
            "direct": False,
            "mentioned": False,
            "lexically_addressed": False,
            "replied_to_target": False,
            "allow_unsolicited": True,
            "unsolicited_channel_ids": {"general"},
        }
        self.assertTrue(is_speech_eligible(channel="general", **common))
        self.assertFalse(is_speech_eligible(channel="random", **common))

    def test_audit_keeps_receipt_when_attachments_are_forbidden(self) -> None:
        self.assertEqual(
            audit_delivery_mode(can_embed=True, can_attach=True), "embed_file"
        )
        self.assertEqual(
            audit_delivery_mode(can_embed=True, can_attach=False), "embed"
        )
        self.assertEqual(
            audit_delivery_mode(can_embed=False, can_attach=True), "text"
        )
        self.assertEqual(
            audit_delivery_mode(
                can_send=False, can_embed=True, can_attach=True
            ),
            "none",
        )

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
