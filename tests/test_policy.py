from __future__ import annotations

import unittest

import numpy as np

from unicorn_lite.core import CoreOutput
from unicorn_lite.models import Experience
from unicorn_lite.policy import CostAwarePolicy


def core_output(surprise: float = 0.5) -> CoreOutput:
    return CoreOutput(surprise, np.zeros(3), np.zeros(4), 1.0, 1.0)


class PolicyTests(unittest.TestCase):
    def test_background_chatter_is_silent(self) -> None:
        decision = CostAwarePolicy().decide(
            Experience("lol okay", metadata={}), core_output(), []
        )
        self.assertEqual(decision.action, "OBSERVE")
        self.assertEqual(decision.compute_tier, 0)

    def test_direct_request_uses_flash(self) -> None:
        decision = CostAwarePolicy().decide(
            Experience("can you remember this?", metadata={"direct": True}),
            core_output(),
            [],
        )
        self.assertEqual(decision.action, "REPLY_FLASH")

    def test_complex_direct_request_uses_pro(self) -> None:
        text = "research and analyze this architecture " + "carefully " * 100
        decision = CostAwarePolicy().decide(
            Experience(text, metadata={"direct": True}), core_output(), []
        )
        self.assertEqual(decision.action, "REPLY_PRO")


if __name__ == "__main__":
    unittest.main()
