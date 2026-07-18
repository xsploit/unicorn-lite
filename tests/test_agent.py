from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from unicorn_lite.agent import UnicornAgent
from unicorn_lite.encoder import HashEncoder, encoder_identity
from unicorn_lite.memory_reranker import AnswerAwareMemoryReranker
from unicorn_lite.models import Experience, MemoryHit
from unicorn_lite.writer import WriterUnavailable


class BrokenWriter:
    async def write(self, **_: object) -> None:
        raise WriterUnavailable("test writer offline")


class CountingWriter:
    def __init__(self) -> None:
        self.calls = 0
        self.memories = []

    async def write(self, **kwargs: object) -> str:
        self.calls += 1
        self.memories = list(kwargs.get("memories", []))
        return "called"


class FakeEMDR2Retriever:
    def __init__(self, *_: object, **__: object) -> None:
        self.metadata = {"architecture": "discord_emdr2_v1"}
        self.records: list[dict[str, object]] = []

    def refresh(self, records: list[dict[str, object]]) -> None:
        self.records = list(records)

    def add(self, record: dict[str, object]) -> None:
        self.records.append(record)

    def retrieve(self, _: str, limit: int = 5) -> list[MemoryHit]:
        return [
            MemoryHit(
                event_id=str(record["event_id"]),
                content=str(record["content"]),
                actor=str(record["actor"]),
                similarity=0.9 - index * 0.01,
                occurred_at=str(record["occurred_at"]),
                decision_action=str(record.get("decision_action") or "OBSERVE"),
            )
            for index, record in enumerate(reversed(self.records[-limit:]))
        ]


class AgentTests(unittest.TestCase):
    def test_ingest_and_restart_preserve_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "agent.db"
            first = UnicornAgent(db, HashEncoder(), device="cpu")
            result = asyncio.run(
                first.ingest(
                    Experience("i prefer local models", actor="subsect"),
                    allow_writer=False,
                )
            )
            saved = first.core.export_state()
            first.close()

            second = UnicornAgent(db, HashEncoder(), device="cpu")
            restored = second.core.export_state()
            self.assertEqual(result.decision.action, "OBSERVE")
            self.assertEqual(second.store.event_count(), 1)
            self.assertEqual(second.store.current_facts("subsect")[0]["value"], "local models")
            np.testing.assert_allclose(saved[0], restored[0])
            np.testing.assert_allclose(saved[1], restored[1])
            self.assertEqual(saved[2], restored[2])
            second.close()

    def test_writer_failure_does_not_lose_experience(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = UnicornAgent(
                Path(directory) / "agent.db",
                HashEncoder(),
                device="cpu",
                writer=BrokenWriter(),  # type: ignore[arg-type]
            )
            result = asyncio.run(
                agent.ingest(
                    Experience("answer me", metadata={"direct": True}),
                    allow_writer=True,
                )
            )
            self.assertEqual(agent.store.event_count(), 1)
            self.assertEqual(result.writer_error, "test writer offline")
            self.assertIsNone(result.reply)
            agent.close()

    def test_direct_message_does_not_call_writer_without_explicit_permission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = CountingWriter()
            agent = UnicornAgent(
                Path(directory) / "agent.db",
                HashEncoder(),
                device="cpu",
                writer=writer,  # type: ignore[arg-type]
            )
            result = asyncio.run(
                agent.ingest(Experience("answer me", metadata={"direct": True}))
            )
            self.assertEqual(result.decision.action, "REPLY_FLASH")
            self.assertEqual(writer.calls, 0)
            self.assertIsNone(result.reply)
            agent.close()

    def test_accepted_reranker_runs_only_on_reply_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoder = HashEncoder(dimension=16)
            checkpoint = Path(directory) / "reranker.pt"
            model = AnswerAwareMemoryReranker(embedding_dim=16, hidden_dim=8)
            torch.save(
                model.checkpoint(
                    encoder=encoder_identity(encoder),
                    candidate_limit=20,
                    accepted_for_runtime=True,
                ),
                checkpoint,
            )
            writer = CountingWriter()
            agent = UnicornAgent(
                Path(directory) / "agent.db",
                encoder,
                device="cpu",
                writer=writer,  # type: ignore[arg-type]
                memory_reranker_checkpoint=str(checkpoint),
            )
            for index in range(6):
                background = Experience(f"background memory {index}")
                asyncio.run(agent.ingest(background))
                self.assertNotIn("memory_reranker", background.metadata)
            event = Experience("answer this", metadata={"direct": True})
            result = asyncio.run(agent.ingest(event, allow_writer=True))
            self.assertEqual(result.decision.action, "REPLY_FLASH")
            self.assertTrue(event.metadata["memory_reranker"]["applied"])
            self.assertIn("cosine_event_ids", event.metadata["memory_reranker"])
            self.assertIn("changed_top5", event.metadata["memory_reranker"])
            self.assertIn("selected_probabilities", event.metadata["memory_reranker"])
            probabilities = event.metadata["memory_reranker"][
                "selected_probabilities"
            ]
            self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities))
            self.assertEqual(probabilities, sorted(probabilities, reverse=True))
            self.assertEqual(writer.calls, 1)
            self.assertEqual(len(writer.memories), 5)
            agent.close()

    def test_emdr2_index_runs_only_after_reply_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "unicorn_lite.agent.EMDR2RuntimeRetriever", FakeEMDR2Retriever
        ):
            writer = CountingWriter()
            agent = UnicornAgent(
                Path(directory) / "agent.db",
                HashEncoder(),
                device="cpu",
                writer=writer,  # type: ignore[arg-type]
                emdr2_checkpoint="fake-directory",
            )
            for index in range(6):
                event = Experience(f"background {index}")
                asyncio.run(agent.ingest(event))
                self.assertNotIn("emdr2", event.metadata)
            direct = Experience("answer this", metadata={"direct": True})
            result = asyncio.run(agent.ingest(direct, allow_writer=True))
            self.assertEqual(result.decision.action, "REPLY_FLASH")
            self.assertTrue(direct.metadata["emdr2"]["applied"])
            self.assertEqual(direct.metadata["emdr2"]["indexed_memories"], 6)
            self.assertEqual(writer.calls, 1)
            self.assertEqual(len(writer.memories), 5)
            self.assertEqual(len(agent.emdr2_retriever.records), 7)
            agent.close()


if __name__ == "__main__":
    unittest.main()
