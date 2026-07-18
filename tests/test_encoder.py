from __future__ import annotations

import unittest

from unicorn_lite.encoder import (
    HashEncoder,
    _conversation_lines,
    encoder_identity,
    prepare_encoder_text,
)


class ConversationEncoderHelpersTests(unittest.TestCase):
    def test_structured_conversation_lines_preserve_roles(self) -> None:
        self.assertEqual(
            _conversation_lines("[AGENT] hello\n[CURRENT_USER] answer >>>"),
            [("AGENT", "hello"), ("CURRENT_USER", "answer >>>")],
        )

    def test_unstructured_text_becomes_current_utterance(self) -> None:
        self.assertEqual(
            _conversation_lines("plain message"),
            [("CURRENT", "plain message")],
        )

    def test_non_conversation_encoder_receives_normalized_text(self) -> None:
        self.assertEqual(prepare_encoder_text(HashEncoder(), "Ans >>>"), "Ans")

    def test_structure_aware_encoder_keeps_speaker_lines(self) -> None:
        class StructureAware:
            preserve_structure = True

        text = "[OTHER_USER] hello\n[CURRENT_USER] answer >>>"
        self.assertEqual(prepare_encoder_text(StructureAware(), text), text)

    def test_encoder_identity_distinguishes_equal_width_models(self) -> None:
        class NamedEncoder:
            dimension = 768

            def __init__(self, name: str) -> None:
                self.model_name = name

        self.assertNotEqual(
            encoder_identity(NamedEncoder("bert-base")),
            encoder_identity(NamedEncoder("mpc-bert")),
        )


if __name__ == "__main__":
    unittest.main()
