from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .core import CoreOutput
from .encoder import Encoder, encoder_identity, prepare_encoder_text
from .models import Decision, Experience, MemoryHit
from .policy import FLASH_MODEL, PRO_MODEL
from .response_gate import (
    LEGACY_FEATURE_DIM,
    PonderResponseGate,
    ResponseGate,
    answered_refractory_features,
    calibrated_reply_probability,
    normalize_policy_text,
    policy_features,
    response_gate_from_checkpoint,
)
from .store import MemoryStore


class LearnedResponsePolicy:
    """Accepted local reply/compute policy; the writer is only a selected action."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        encoder: Encoder,
        store: MemoryStore,
        device: str = "cpu",
        threshold_overrides: dict[str, float] | None = None,
    ) -> None:
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if not bool(checkpoint.get("acceptance_passed")):
            raise RuntimeError(
                "learned response checkpoint failed offline acceptance; active use refused"
            )
        checkpoint_encoder_id = checkpoint.get("encoder_id")
        if checkpoint_encoder_id and str(checkpoint_encoder_id) != encoder_identity(encoder):
            raise RuntimeError(
                "learned response checkpoint was trained with a different encoder"
            )
        self.device = torch.device(device)
        self.encoder = encoder
        self.store = store
        feature_dim = int(
            checkpoint.get(
                "feature_dim",
                checkpoint["model"]["structural_action_head.weight"].shape[1]
                if "structural_action_head.weight" in checkpoint["model"]
                else LEGACY_FEATURE_DIM,
            )
        )
        self.model = response_gate_from_checkpoint(checkpoint).to(self.device)
        self.model.eval()
        self.threshold = float(checkpoint["threshold"])
        self.threshold_overrides = dict(threshold_overrides or {})
        self.mode = str(checkpoint.get("policy_mode", "learned"))
        self.refractory_calibrator = checkpoint.get("refractory_calibrator")
        self.model_id = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        self.states: dict[str, torch.Tensor] = {}
        self.messages_seen: dict[str, int] = {}

    def threshold_for(self, channel: str) -> float:
        return float(self.threshold_overrides.get(channel, self.threshold))

    def _state(self, channel: str) -> torch.Tensor:
        if channel not in self.states:
            stored = self.store.load_policy_state(channel, self.model_id)
            if stored is None:
                self.states[channel] = torch.zeros(
                    self.model.hidden_dim, device=self.device
                )
                self.messages_seen[channel] = 0
            else:
                hidden, seen = stored
                self.states[channel] = torch.as_tensor(
                    hidden, dtype=torch.float32, device=self.device
                )
                self.messages_seen[channel] = seen
        return self.states[channel]

    def decide(
        self,
        event: Experience,
        core: CoreOutput,
        memories: list[MemoryHit],
    ) -> Decision:
        if bool(event.metadata.get("author_is_bot")):
            return Decision(
                action="OBSERVE",
                compute_tier=0,
                confidence=1.0,
                reason="bot-authored event updates memory but cannot trigger speech",
            )

        row = dict(event.metadata)
        row["content"] = event.content
        similarities = sorted(
            (float(hit.similarity) for hit in memories), reverse=True
        )
        normalized_content = normalize_policy_text(event.content).casefold()
        exact_hits = [
            hit
            for hit in memories
            if normalize_policy_text(hit.content).casefold() == normalized_content
        ]
        answered_hits = [
            hit for hit in memories if str(hit.decision_action or "").startswith("REPLY")
        ]
        now = datetime.fromisoformat(event.occurred_at.replace("Z", "+00:00"))
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        refractory = answered_refractory_features(
            now,
            (
                (
                    float(hit.similarity),
                    datetime.fromisoformat(hit.occurred_at.replace("Z", "+00:00")),
                )
                for hit in answered_hits
            ),
        )
        row.update(
            {
                "max_memory_similarity": similarities[0] if similarities else 0.0,
                "mean_top3_memory_similarity": (
                    sum(similarities[:3]) / len(similarities[:3])
                    if similarities
                    else 0.0
                ),
                "near_duplicate_count": sum(value >= 0.90 for value in similarities),
                "exact_duplicate": bool(exact_hits),
                "max_answered_similarity": max(
                    (float(hit.similarity) for hit in answered_hits), default=0.0
                ),
                "recent_answered_exact_duplicate": any(
                    str(hit.decision_action or "").startswith("REPLY")
                    and 0.0
                    <= (now - datetime.fromisoformat(hit.occurred_at.replace("Z", "+00:00"))).total_seconds()
                    <= 86400.0
                    for hit in exact_hits
                ),
                **refractory,
            }
        )
        event.metadata["gate_memory_signals"] = {
            key: row[key]
            for key in (
                "max_memory_similarity",
                "mean_top3_memory_similarity",
                "near_duplicate_count",
                "exact_duplicate",
                "max_answered_similarity",
                "recent_answered_exact_duplicate",
                "recent_answered_similarity",
                "recent_answered_near_duplicate",
                "seconds_since_answered_match",
            )
        }
        model_input = prepare_encoder_text(
            self.encoder, str(row.get("policy_input") or event.content)
        )
        embedding = torch.as_tensor(
            self.encoder.encode(model_input), dtype=torch.float32, device=self.device
        )
        features = torch.as_tensor(
            policy_features(row, self.model.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        channel = event.channel
        with torch.no_grad():
            if isinstance(self.model, PonderResponseGate):
                ponder = self.model.ponder(
                    embedding,
                    features,
                    self._state(channel),
                    adaptive=True,
                )
                hidden, logits = ponder.hidden, ponder.logits
                event.metadata["ponder"] = {
                    "steps": ponder.steps,
                    "max_steps": self.model.max_steps,
                    "halt_probability": float(ponder.halt_probabilities[-1]),
                    "halt_threshold": self.model.halt_threshold,
                    "probability_path": ponder.probability_path,
                }
            else:
                hidden, logits = self.model.step(
                    embedding, features, self._state(channel)
                )
            base_probability = float(F.softmax(logits, dim=-1)[1])
            probability = calibrated_reply_probability(
                base_probability, row, self.refractory_calibrator
            )
            event.metadata["refractory_calibration"] = {
                "base_probability": base_probability,
                "calibrated_probability": probability,
                "applied": bool(self.refractory_calibrator),
            }
        self.states[channel] = hidden
        self.messages_seen[channel] = self.messages_seen.get(channel, 0) + 1
        self.store.save_policy_state(
            channel,
            self.model_id,
            hidden.detach().cpu().numpy().astype(np.float32),
            self.messages_seen[channel],
        )

        threshold = self.threshold_for(channel)
        should_reply = probability >= threshold
        if not should_reply:
            return Decision(
                action="OBSERVE",
                compute_tier=0,
                confidence=1.0 - probability,
                reason=(
                    f"learned reply probability {probability:.3f} below "
                    f"{threshold:.3f}; no writer call"
                ),
            )

        learned_tier = int(np.argmax(core.compute_logits))
        compute_tier = 3 if learned_tier == 3 else 2
        model = PRO_MODEL if compute_tier == 3 else FLASH_MODEL
        return Decision(
            action="REPLY_PRO" if compute_tier == 3 else "REPLY_FLASH",
            compute_tier=compute_tier,
            confidence=probability,
            reason=f"learned gate selected speech at probability {probability:.3f}",
            model=model,
        )
