from __future__ import annotations

import re

import numpy as np

from .core import CoreOutput
from .models import Decision, Experience, MemoryHit


FLASH_MODEL = "deepseek/deepseek-v4-flash"
PRO_MODEL = "deepseek/deepseek-v4-pro"


class CostAwarePolicy:
    """Cold-start policy: most events update memory without invoking an LLM."""

    def __init__(
        self,
        flash_model: str = FLASH_MODEL,
        pro_model: str = PRO_MODEL,
    ) -> None:
        self.flash_model = flash_model
        self.pro_model = pro_model

    def decide(
        self,
        event: Experience,
        core: CoreOutput,
        memories: list[MemoryHit],
    ) -> Decision:
        text = event.content.strip()
        words = re.findall(r"\S+", text)
        direct = bool(event.metadata.get("direct") or event.metadata.get("mentioned"))
        question = "?" in text
        command = bool(event.metadata.get("command"))
        complex_request = (
            len(words) >= 90
            or text.count("\n") >= 8
            or "```" in text
            or sum(token in text.lower() for token in ("research", "debug", "architecture", "analyze")) >= 2
        )
        highest_similarity = memories[0].similarity if memories else 0.0

        if direct or command:
            if complex_request:
                return Decision(
                    action="REPLY_PRO",
                    compute_tier=3,
                    confidence=0.88,
                    reason="direct complex request; deeper writer is worth its cost",
                    model=self.pro_model,
                )
            return Decision(
                action="REPLY_FLASH",
                compute_tier=2,
                confidence=0.9,
                reason="direct interaction; cheap writer should be sufficient",
                model=self.flash_model,
            )

        if question and core.surprise > 0.72 and len(words) >= 8:
            return Decision(
                action="REPLY_FLASH",
                compute_tier=2,
                confidence=0.62,
                reason="novel open question with likely conversational value",
                model=self.flash_model,
            )

        if highest_similarity > 0.82:
            return Decision(
                action="OBSERVE",
                compute_tier=0,
                confidence=0.91,
                reason="repetitive event; memory update only",
            )

        return Decision(
            action="OBSERVE",
            compute_tier=0,
            confidence=0.82,
            reason="no response has enough expected value to justify generation",
        )
