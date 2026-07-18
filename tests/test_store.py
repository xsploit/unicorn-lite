from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from unicorn_lite.encoder import HashEncoder
from unicorn_lite.models import Decision, Experience
from unicorn_lite.store import MemoryStore


class StoreTests(unittest.TestCase):
    def test_policy_diagnostics_can_be_persisted_after_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.db")
            encoder = HashEncoder()
            event = Experience("hello", metadata={"direct": True})
            store.append_event(event, encoder.encode(event.content))
            event.metadata["gate_memory_signals"] = {
                "recent_answered_similarity": 0.95
            }
            store.update_event_metadata(event.event_id, event.metadata)
            row = store.connection.execute(
                "SELECT metadata_json FROM events WHERE event_id=?", (event.event_id,)
            ).fetchone()
            self.assertEqual(
                json.loads(row["metadata_json"])["gate_memory_signals"][
                    "recent_answered_similarity"
                ],
                0.95,
            )
            store.close()

    def test_event_retrieval_and_fact_supersession(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.db")
            encoder = HashEncoder()
            old = Experience("my name is Alpha", actor="u")
            new = Experience("call me Beta", actor="u")
            store.append_event(old, encoder.encode(old.content))
            store.append_event(new, encoder.encode(new.content))
            self.assertTrue(store.upsert_fact("u", "name", "Alpha", old.event_id))
            self.assertTrue(store.upsert_fact("u", "name", "Beta", new.event_id))
            current = store.current_facts("u")
            history = store.fact_history("u", "name")
            self.assertEqual(current[0]["value"], "Beta")
            self.assertEqual([item["value"] for item in history], ["Alpha", "Beta"])
            self.assertIsNotNone(history[0]["valid_to"])
            hits = store.retrieve(encoder.encode("Alpha name"), limit=2)
            self.assertEqual(len(hits), 2)
            store.close()

    def test_retrieval_ignores_embeddings_from_another_encoder_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.db")
            event_384 = Experience("MiniLM memory", actor="u")
            event_768 = Experience("BERT memory", actor="u")
            store.append_event(event_384, np.ones(384, dtype=np.float32))
            store.append_event(event_768, np.ones(768, dtype=np.float32))

            hits_384 = store.retrieve(np.ones(384, dtype=np.float32))
            hits_768 = store.retrieve(np.ones(768, dtype=np.float32))

            self.assertEqual([hit.event_id for hit in hits_384], [event_384.event_id])
            self.assertEqual([hit.event_id for hit in hits_768], [event_768.event_id])
            store.close()

    def test_memory_records_preserve_provenance_and_latest_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.db")
            event = Experience("remember this", actor="subsect")
            store.append_event(event, np.ones(4, dtype=np.float32))
            store.record_decision(event.event_id, Decision("OBSERVE", 0, 0.8, "test"))
            rows = store.memory_records()
            self.assertEqual(rows[0]["event_id"], event.event_id)
            self.assertEqual(rows[0]["actor"], "subsect")
            self.assertEqual(rows[0]["decision_action"], "OBSERVE")
            store.close()


if __name__ == "__main__":
    unittest.main()
