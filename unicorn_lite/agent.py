from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .core import NeuralCore
from .emdr2 import EMDR2RuntimeRetriever
from .encoder import Encoder, encoder_identity
from .facts import extract_and_store
from .memory_reranker import reranker_from_checkpoint
from .models import AgentResult, Experience
from .policy import CostAwarePolicy
from .store import MemoryStore
from .writer import Writer, WriterUnavailable


class UnicornAgent:
    def __init__(
        self,
        db_path: str | Path,
        encoder: Encoder,
        device: str = "cpu",
        latent_dim: int = 96,
        writer: Writer | None = None,
        policy: CostAwarePolicy | None = None,
        checkpoint: str | None = None,
        memory_reranker_checkpoint: str | None = None,
        emdr2_checkpoint: str | None = None,
    ) -> None:
        if memory_reranker_checkpoint and emdr2_checkpoint:
            raise ValueError("select either the proxy reranker or EMDR2, not both")
        self.store = MemoryStore(db_path)
        self.encoder = encoder
        self.core = NeuralCore(
            embedding_dim=encoder.dimension,
            latent_dim=latent_dim,
            device=device,
        )
        if checkpoint:
            self.core.load_checkpoint(checkpoint)
        stored = self.store.load_core_state(self.core.checkpoint_id)
        if stored is not None:
            self.core.import_state(*stored)
        self.policy = policy or CostAwarePolicy()
        self.writer = writer
        self.memory_reranker = None
        self.emdr2_retriever = None
        self.memory_candidate_limit = 5
        if memory_reranker_checkpoint:
            reranker_checkpoint = torch.load(
                memory_reranker_checkpoint, map_location=device, weights_only=True
            )
            if not bool(reranker_checkpoint.get("accepted_for_runtime")):
                raise ValueError("memory reranker did not pass offline acceptance")
            if str(reranker_checkpoint.get("encoder")) != encoder_identity(encoder):
                raise ValueError("memory reranker encoder does not match agent encoder")
            self.memory_reranker = reranker_from_checkpoint(
                reranker_checkpoint, device=device
            )
            self.memory_candidate_limit = int(
                reranker_checkpoint.get("candidate_limit", 20)
            )
        if emdr2_checkpoint:
            self.emdr2_retriever = EMDR2RuntimeRetriever(
                emdr2_checkpoint, device=device
            )
            self.emdr2_retriever.refresh(self.store.memory_records())

    async def ingest(self, event: Experience, allow_writer: bool = False) -> AgentResult:
        embedding = self.encoder.encode(event.content)
        candidates = self.store.retrieve(embedding, limit=self.memory_candidate_limit)
        policy_memories = candidates[:5]
        core_output = self.core.process(embedding)
        self.store.append_event(event, embedding)
        facts_updated = extract_and_store(event, self.store)
        decision = self.policy.decide(event, core_output, policy_memories)
        memories = policy_memories
        if (
            decision.action.startswith("REPLY")
            and self.memory_reranker is not None
            and candidates
        ):
            candidate_vectors = np.stack([hit.embedding for hit in candidates])
            order, scores = self.memory_reranker.rank(embedding, candidate_vectors)
            normalized = np.exp(scores - float(np.max(scores)))
            probabilities = normalized / float(np.sum(normalized))
            memories = [candidates[index] for index in order[:5]]
            event.metadata["memory_reranker"] = {
                "applied": True,
                "candidates": len(candidates),
                "cosine_event_ids": [hit.event_id for hit in candidates[:5]],
                "selected_event_ids": [hit.event_id for hit in memories],
                "selected_scores": [float(scores[index]) for index in order[:5]],
                "selected_probabilities": [
                    float(probabilities[index]) for index in order[:5]
                ],
                "selector_margin": (
                    float(scores[order[0]] - scores[order[1]])
                    if len(order) > 1
                    else 0.0
                ),
                "changed_top5": [hit.event_id for hit in memories]
                != [hit.event_id for hit in candidates[:5]],
            }
        elif decision.action.startswith("REPLY") and self.emdr2_retriever is not None:
            memories = self.emdr2_retriever.retrieve(event.content, limit=5)
            event.metadata["emdr2"] = {
                "applied": True,
                "architecture": self.emdr2_retriever.metadata["architecture"],
                "indexed_memories": len(self.emdr2_retriever.records),
                "cosine_event_ids": [hit.event_id for hit in policy_memories],
                "selected_event_ids": [hit.event_id for hit in memories],
                "changed_top5": [hit.event_id for hit in memories]
                != [hit.event_id for hit in policy_memories],
            }
        if self.emdr2_retriever is not None:
            self.emdr2_retriever.add(
                {
                    "event_id": event.event_id,
                    "content": event.content,
                    "actor": event.actor,
                    "occurred_at": event.occurred_at,
                    "decision_action": decision.action,
                }
            )
        # The learned policy adds memory and ponder diagnostics to metadata.
        # Store them after the decision so live failures remain trainable from
        # the canonical event ledger instead of only from Discord audit files.
        self.store.update_event_metadata(event.event_id, event.metadata)
        self.store.record_decision(event.event_id, decision)
        hidden, fast, count = self.core.export_state()
        self.store.save_core_state(
            hidden,
            fast,
            count,
            event.event_id,
            model_id=self.core.checkpoint_id,
        )

        reply = None
        writer_error = None
        if (
            allow_writer
            and decision.action.startswith("REPLY")
            and self.writer is not None
        ):
            try:
                reply = await self.writer.write(
                    event=event,
                    decision=decision,
                    memories=memories,
                    facts=self.store.current_facts(),
                )
            except WriterUnavailable as exc:
                writer_error = str(exc)
        return AgentResult(
            event_id=event.event_id,
            decision=decision,
            surprise=core_output.surprise,
            memories=memories,
            facts_updated=facts_updated,
            reply=reply,
            writer_error=writer_error,
        )

    def close(self) -> None:
        self.store.close()

    async def warm_writer(self) -> bool:
        warmup = getattr(self.writer, "warmup", None)
        if warmup is None:
            return False
        await warmup()
        return True
