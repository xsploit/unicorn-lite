from __future__ import annotations

import unittest

from unicorn_lite.discord_runtime import is_lexically_addressed


class DiscordRuntimeTests(unittest.TestCase):
    def test_opening_vocative_is_detected(self) -> None:
        aliases = {"Neuro-sama", "neuro"}
        self.assertTrue(is_lexically_addressed("hey neuro i retrained you", aliases))
        self.assertTrue(is_lexically_addressed("Neuro-sama, thoughts?", aliases))

    def test_incidental_name_is_not_an_address(self) -> None:
        aliases = {"neuro"}
        self.assertFalse(is_lexically_addressed("I was talking about neuro earlier", aliases))
        self.assertFalse(is_lexically_addressed("neurology is interesting", aliases))


if __name__ == "__main__":
    unittest.main()
