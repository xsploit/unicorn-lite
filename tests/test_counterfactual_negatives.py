from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_counterfactual_emoji_negatives.py"
SPEC = importlib.util.spec_from_file_location("counterfactual_negatives", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CounterfactualNegativeTests(unittest.TestCase):
    def test_only_emoji_or_custom_emoji_is_selected(self) -> None:
        self.assertTrue(MODULE.emoji_or_symbol_only("🥒"))
        self.assertTrue(MODULE.emoji_or_symbol_only("<:troll:907846765519196251>"))
        self.assertFalse(MODULE.emoji_or_symbol_only("hello 🥒"))
        self.assertFalse(MODULE.emoji_or_symbol_only("https://tenor.com/example"))
        self.assertFalse(MODULE.emoji_or_symbol_only("<@123> 🥒"))


if __name__ == "__main__":
    unittest.main()
