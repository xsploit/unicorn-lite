from __future__ import annotations

import unittest

from unicorn_lite.models import Decision, Experience
from unicorn_lite.writer import _writer_messages


class WriterPacketTests(unittest.TestCase):
    def test_persona_and_event_are_kept_separate_from_memory_evidence(self) -> None:
        messages = _writer_messages(
            "persona prompt",
            Experience("hello", actor="person"),
            Decision("REPLY_FLASH", 2, 0.9, "direct"),
            [],
            [],
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("persona prompt", messages[0]["content"])
        self.assertIn("hello", messages[1]["content"])
        self.assertIn("Relevant memories", messages[1]["content"])

    def test_recent_speaker_labelled_conversation_reaches_writer(self) -> None:
        event = Experience(
            "what do you mean?",
            actor="person",
            metadata={
                "display_name": "Subsect",
                "conversation_context": (
                    "[OTHER_USER: Karah] the controller seems selective\n"
                    "[CURRENT_USER: Subsect] what do you mean?"
                ),
            },
        )
        messages = _writer_messages(
            "persona prompt",
            event,
            Decision("REPLY_FLASH", 2, 0.7, "selected"),
            [],
            [],
        )
        self.assertIn("Recent channel conversation", messages[1]["content"])
        self.assertIn("[OTHER_USER: Karah]", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
