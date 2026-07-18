from __future__ import annotations

from pathlib import Path

from .core import NeuralCore
from .encoder import Encoder
from .facts import extract_and_store
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
    ) -> None:
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

    async def ingest(self, event: Experience, allow_writer: bool = False) -> AgentResult:
        embedding = self.encoder.encode(event.content)
        memories = self.store.retrieve(embedding, limit=5)
        core_output = self.core.process(embedding)
        self.store.append_event(event, embedding)
        facts_updated = extract_and_store(event, self.store)
        decision = self.policy.decide(event, core_output, memories)
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
