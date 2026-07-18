from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from unicorn_lite.discord_context import DiscordContextTracker
from unicorn_lite.store import MemoryStore


class DiscordContextTrackerTests(unittest.TestCase):
    def test_mention_is_exposed_as_effective_direct_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.db")
            tracker = DiscordContextTracker(store)
            metadata = tracker.metadata(
                channel="general",
                actor="user",
                content="@Neuro thoughts?",
                occurred_at="2026-01-01T00:00:00+00:00",
                direct=False,
                mentioned=True,
                replied_to_target=False,
            )
            self.assertTrue(metadata["direct"])
            self.assertIn("[CURRENT direct=1]", str(metadata["policy_input"]))
            store.close()


if __name__ == "__main__":
    unittest.main()
