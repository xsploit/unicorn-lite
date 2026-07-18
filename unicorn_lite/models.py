from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class Experience:
    content: str
    actor: str = "user"
    source: str = "cli"
    channel: str = "default"
    occurred_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryHit:
    event_id: str
    content: str
    actor: str
    similarity: float
    occurred_at: str
    decision_action: str | None = None
    embedding: Any = field(default=None, repr=False)


@dataclass(slots=True)
class Decision:
    action: str
    compute_tier: int
    confidence: float
    reason: str
    model: str | None = None


@dataclass(slots=True)
class AgentResult:
    event_id: str
    decision: Decision
    surprise: float
    memories: list[MemoryHit]
    facts_updated: list[str]
    reply: str | None = None
    writer_error: str | None = None
