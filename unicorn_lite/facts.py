from __future__ import annotations

import re

from .models import Experience
from .store import MemoryStore


PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("name", re.compile(r"\b(?:my name is|call me)\s+([^.!?,]{1,60})", re.I)),
    ("location", re.compile(r"\b(?:i live in|i'm from|i am from)\s+([^.!?]{1,100})", re.I)),
    ("preference", re.compile(r"\b(?:i like|i prefer)\s+([^.!?]{1,160})", re.I)),
    ("project", re.compile(r"\b(?:i'm|i am) working on\s+([^.!?]{1,180})", re.I)),
)


def extract_and_store(event: Experience, store: MemoryStore) -> list[str]:
    updated: list[str] = []
    for predicate, pattern in PATTERNS:
        match = pattern.search(event.content)
        if match is None:
            continue
        value = match.group(1).strip()
        if store.upsert_fact(
            subject=event.actor,
            predicate=predicate,
            value=value,
            source_event_id=event.event_id,
        ):
            updated.append(f"{event.actor}.{predicate}={value}")
    return updated
