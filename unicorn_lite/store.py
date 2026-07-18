from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .models import Decision, Experience, MemoryHit


def _array_blob(value: np.ndarray) -> bytes:
    return np.asarray(value, dtype=np.float32).tobytes()


def _blob_array(value: bytes, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(value, dtype=np.float32).reshape(shape).copy()


class MemoryStore:
    """SQLite is canonical: raw events are immutable and beliefs are versioned."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                occurred_at TEXT NOT NULL,
                source TEXT NOT NULL,
                actor TEXT NOT NULL,
                channel TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL REFERENCES events(event_id),
                action TEXT NOT NULL,
                compute_tier INTEGER NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                model TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS core_state (
                name TEXT PRIMARY KEY,
                model_id TEXT NOT NULL DEFAULT 'legacy',
                hidden BLOB NOT NULL,
                hidden_dim INTEGER NOT NULL,
                fast BLOB NOT NULL,
                latent_dim INTEGER NOT NULL,
                events_seen INTEGER NOT NULL,
                last_event_id TEXT
            );

            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_event_id TEXT NOT NULL REFERENCES events(event_id),
                valid_from TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                valid_to TEXT
            );

            CREATE INDEX IF NOT EXISTS facts_current
                ON facts(subject, predicate, valid_to);

            CREATE TABLE IF NOT EXISTS policy_state (
                channel TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                hidden BLOB NOT NULL,
                hidden_dim INTEGER NOT NULL,
                messages_seen INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS runtime_state (
                name TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        core_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(core_state)").fetchall()
        }
        if "model_id" not in core_columns:
            self.connection.execute(
                "ALTER TABLE core_state ADD COLUMN model_id TEXT NOT NULL DEFAULT 'legacy'"
            )
        self.connection.commit()

    def append_event(self, event: Experience, embedding: np.ndarray) -> None:
        self.connection.execute(
            """
            INSERT INTO events
            (event_id, occurred_at, source, actor, channel, content,
             metadata_json, embedding, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.occurred_at,
                event.source,
                event.actor,
                event.channel,
                event.content,
                json.dumps(event.metadata, ensure_ascii=False),
                _array_blob(embedding),
                int(embedding.size),
            ),
        )
        self.connection.commit()

    def update_event_metadata(self, event_id: str, metadata: dict[str, object]) -> None:
        """Persist policy diagnostics added after the immutable event payload."""
        self.connection.execute(
            "UPDATE events SET metadata_json=? WHERE event_id=?",
            (json.dumps(metadata, ensure_ascii=False), event_id),
        )
        self.connection.commit()

    def retrieve(
        self,
        query: np.ndarray,
        limit: int = 5,
        exclude_event_id: str | None = None,
    ) -> list[MemoryHit]:
        rows = self.connection.execute(
            """
            SELECT e.event_id, e.content, e.actor, e.occurred_at,
                   e.embedding, e.embedding_dim,
                   (SELECT d.action FROM decisions d
                    WHERE d.event_id=e.event_id
                    ORDER BY d.id DESC LIMIT 1) AS decision_action
            FROM events e
            """
        ).fetchall()
        query_norm = float(np.linalg.norm(query)) or 1.0
        hits: list[MemoryHit] = []
        for row in rows:
            if row["event_id"] == exclude_event_id:
                continue
            # Encoder challengers can coexist in one event ledger. Retrieval is
            # only meaningful inside a shared representation dimension.
            if int(row["embedding_dim"]) != int(query.size):
                continue
            vector = _blob_array(row["embedding"], (row["embedding_dim"],))
            denominator = query_norm * (float(np.linalg.norm(vector)) or 1.0)
            similarity = float(np.dot(query, vector) / denominator)
            hits.append(
                MemoryHit(
                    event_id=row["event_id"],
                    content=row["content"],
                    actor=row["actor"],
                    similarity=similarity,
                    occurred_at=row["occurred_at"],
                    decision_action=row["decision_action"],
                )
            )
        hits.sort(key=lambda item: item.similarity, reverse=True)
        return hits[:limit]

    def save_core_state(
        self,
        hidden: np.ndarray,
        fast: np.ndarray,
        events_seen: int,
        last_event_id: str,
        model_id: str = "legacy",
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO core_state
                (name, model_id, hidden, hidden_dim, fast, latent_dim, events_seen, last_event_id)
            VALUES ('primary', ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                model_id=excluded.model_id,
                hidden=excluded.hidden,
                hidden_dim=excluded.hidden_dim,
                fast=excluded.fast,
                latent_dim=excluded.latent_dim,
                events_seen=excluded.events_seen,
                last_event_id=excluded.last_event_id
            """,
            (
                model_id,
                _array_blob(hidden),
                int(hidden.shape[0]),
                _array_blob(fast),
                int(fast.shape[0]),
                events_seen,
                last_event_id,
            ),
        )
        self.connection.commit()

    def load_core_state(
        self, model_id: str = "legacy"
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        row = self.connection.execute(
            """
            SELECT hidden, hidden_dim, fast, latent_dim, events_seen
            FROM core_state WHERE name='primary' AND model_id=?
            """,
            (model_id,),
        ).fetchone()
        if row is None:
            return None
        latent_dim = int(row["latent_dim"])
        return (
            _blob_array(row["hidden"], (int(row["hidden_dim"]),)),
            _blob_array(row["fast"], (latent_dim, latent_dim)),
            int(row["events_seen"]),
        )

    def record_decision(self, event_id: str, decision: Decision) -> None:
        self.connection.execute(
            """
            INSERT INTO decisions
                (event_id, action, compute_tier, confidence, reason, model)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                decision.action,
                decision.compute_tier,
                decision.confidence,
                decision.reason,
                decision.model,
            ),
        )
        self.connection.commit()

    def save_policy_state(
        self,
        channel: str,
        model_id: str,
        hidden: np.ndarray,
        messages_seen: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO policy_state
                (channel, model_id, hidden, hidden_dim, messages_seen)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(channel) DO UPDATE SET
                model_id=excluded.model_id,
                hidden=excluded.hidden,
                hidden_dim=excluded.hidden_dim,
                messages_seen=excluded.messages_seen,
                updated_at=CURRENT_TIMESTAMP
            """,
            (channel, model_id, _array_blob(hidden), int(hidden.size), messages_seen),
        )
        self.connection.commit()

    def load_policy_state(
        self, channel: str, model_id: str
    ) -> tuple[np.ndarray, int] | None:
        row = self.connection.execute(
            """
            SELECT hidden, hidden_dim, messages_seen
            FROM policy_state WHERE channel=? AND model_id=?
            """,
            (channel, model_id),
        ).fetchone()
        if row is None:
            return None
        return (
            _blob_array(row["hidden"], (int(row["hidden_dim"]),)),
            int(row["messages_seen"]),
        )

    def save_runtime_state(self, name: str, value: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO runtime_state(name, value_json) VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET
                value_json=excluded.value_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (name, json.dumps(value, ensure_ascii=False)),
        )
        self.connection.commit()

    def load_runtime_state(self, name: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT value_json FROM runtime_state WHERE name=?", (name,)
        ).fetchone()
        return json.loads(row["value_json"]) if row is not None else None

    def upsert_fact(
        self,
        subject: str,
        predicate: str,
        value: str,
        source_event_id: str,
        confidence: float = 0.9,
    ) -> bool:
        existing = self.connection.execute(
            """
            SELECT id, value FROM facts
            WHERE subject=? AND predicate=? AND valid_to IS NULL
            ORDER BY id DESC LIMIT 1
            """,
            (subject, predicate),
        ).fetchone()
        if existing is not None and existing["value"].casefold() == value.casefold():
            return False
        if existing is not None:
            self.connection.execute(
                "UPDATE facts SET valid_to=CURRENT_TIMESTAMP WHERE id=?",
                (existing["id"],),
            )
        self.connection.execute(
            """
            INSERT INTO facts(subject, predicate, value, confidence, source_event_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (subject, predicate, value, confidence, source_event_id),
        )
        self.connection.commit()
        return True

    def current_facts(self, subject: str | None = None) -> list[dict[str, Any]]:
        if subject is None:
            rows = self.connection.execute(
                "SELECT subject, predicate, value, confidence, source_event_id FROM facts WHERE valid_to IS NULL"
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT subject, predicate, value, confidence, source_event_id
                FROM facts WHERE valid_to IS NULL AND subject=?
                """,
                (subject,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fact_history(self, subject: str, predicate: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT value, valid_from, valid_to, source_event_id
            FROM facts WHERE subject=? AND predicate=? ORDER BY id
            """,
            (subject, predicate),
        ).fetchall()
        return [dict(row) for row in rows]

    def event_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def close(self) -> None:
        self.connection.close()
