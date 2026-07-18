from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np

from unicorn_lite.agent import UnicornAgent
from unicorn_lite.encoder import HashEncoder
from unicorn_lite.models import Experience
from unicorn_lite.writer import WriterUnavailable


class BrokenWriter:
    async def write(self, **_: object) -> None:
        raise WriterUnavailable("test writer offline")


class CountingWriter:
    def __init__(self) -> None:
        self.calls = 0

    async def write(self, **_: object) -> str:
        self.calls += 1
        return "called"


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


if __name__ == "__main__":
    unittest.main()
