from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LEGACY_FEATURE_DIM = 16
FEATURE_DIM = 22
REFRACTORY_FEATURE_DIM = 4
POLICY_TEXT_NORMALIZATION = "semantic-alphanumeric-v2"

QUESTION_OPENERS = {
    "are", "can", "could", "did", "do", "does", "has", "have", "how",
    "is", "should", "was", "were", "what", "when", "where", "which",
    "who", "why", "will", "would",
}


def normalize_policy_text(text: str) -> str:
    """Remove decorative punctuation leverage without changing stored/user text."""
    text = re.sub(r"<@!?\d+>", " MENTION ", text)
    output: list[str] = []
    for index, character in enumerate(text):
        if character.isalnum() or character.isspace():
            output.append(character)
            continue
        if (
            character in {"'", "-"}
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isalnum()
            and text[index + 1].isalnum()
        ):
            output.append(character)
        else:
            output.append(" ")
    normalized = re.sub(r"\s+", " ", "".join(output)).strip()
    return normalized if re.search(r"[A-Za-z0-9]", normalized) else "PUNCTUATION_ONLY"


def answered_refractory_features(
    now: datetime,
    matches: Iterable[tuple[float, datetime]],
    *,
    history_window_seconds: float = 86400.0,
    near_duplicate_seconds: float = 600.0,
) -> dict[str, float | bool]:
    """Summarize causally available similarity to previously answered turns."""
    recent: list[tuple[float, float]] = []
    for similarity, occurred_at in matches:
        age = (now - occurred_at).total_seconds()
        if 0.0 <= age <= history_window_seconds:
            recent.append((float(similarity), float(age)))
    if not recent:
        return {
            "recent_answered_similarity": 0.0,
            "recent_answered_near_duplicate": False,
            "seconds_since_answered_match": history_window_seconds,
        }
    similarity, age = max(recent, key=lambda item: item[0])
    return {
        "recent_answered_similarity": similarity,
        "recent_answered_near_duplicate": (
            similarity >= 0.90 and age <= near_duplicate_seconds
        ),
        "seconds_since_answered_match": age,
    }


def looks_like_question(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    if len(words) < 2:
        return False
    return "?" in text or words[0] in QUESTION_OPENERS


def refractory_features(row: Mapping[str, object]) -> np.ndarray:
    """Features for the frozen-gate additive cooldown calibrator."""
    similarity = min(
        1.0, max(0.0, float(row.get("recent_answered_similarity", 0.0)))
    )
    age = min(
        1.0,
        math.log1p(float(row.get("seconds_since_answered_match", 86400.0)))
        / math.log(86401.0),
    )
    freshness = 1.0 - age
    return np.asarray(
        [
            similarity,
            float(bool(row.get("recent_answered_near_duplicate"))),
            freshness,
            similarity * freshness,
        ],
        dtype=np.float32,
    )


def calibrated_reply_probability(
    probability: float,
    row: Mapping[str, object],
    calibrator: Mapping[str, object] | None,
) -> float:
    if not calibrator:
        return float(probability)
    weights = np.asarray(calibrator.get("weights", []), dtype=np.float32)
    features = refractory_features(row)
    if weights.shape != features.shape:
        raise ValueError("refractory calibrator has incompatible weights")
    clipped = min(1.0 - 1e-6, max(1e-6, float(probability)))
    logit = math.log(clipped / (1.0 - clipped))
    return float(1.0 / (1.0 + math.exp(-(logit + float(np.dot(weights, features))))))


def policy_features(
    row: Mapping[str, object], feature_dim: int = FEATURE_DIM
) -> np.ndarray:
    raw_text = str(row.get("content", ""))
    text = normalize_policy_text(raw_text)
    length = len(text)
    values = np.asarray(
        [
            float(bool(row.get("direct"))),
            float(looks_like_question(raw_text)),
            min(1.0, math.log1p(length) / math.log(2001.0)),
            min(1.0, text.count("\n") / 8.0),
            float(bool(re.search(r"https?://", text))),
            float("```" in text),
            min(
                1.0,
                math.log1p(float(row.get("seconds_since_agent", 86400.0)))
                / math.log(86401.0),
            ),
            min(
                1.0,
                math.log1p(float(row.get("messages_since_agent", 0)))
                / math.log(101.0),
            ),
            min(1.0, max(0.0, float(row.get("author_reply_rate_past", 0.0)))),
            min(
                1.0,
                math.log1p(float(row.get("author_messages_seen", 0)))
                / math.log(101.0),
            ),
            float(bool(row.get("mentioned_target") or row.get("mentioned"))),
            float(bool(row.get("replied_to_target"))),
            min(1.0, max(0.0, float(row.get("recent_agent_activity", 0.0)))),
            min(1.0, max(0.0, float(row.get("channel_reply_rate_past", 0.0)))),
            min(1.0, max(0.0, float(row.get("global_reply_rate_past", 0.0)))),
            min(
                1.0,
                math.log1p(float(row.get("channel_human_messages_seen", 0)))
                / math.log(2001.0),
            ),
            min(1.0, max(-1.0, float(row.get("max_memory_similarity", 0.0)))),
            min(1.0, max(-1.0, float(row.get("mean_top3_memory_similarity", 0.0)))),
            min(1.0, max(0.0, float(row.get("near_duplicate_count", 0.0))) / 5.0),
            float(bool(row.get("exact_duplicate"))),
            min(1.0, max(-1.0, float(row.get("max_answered_similarity", 0.0)))),
            float(bool(row.get("recent_answered_exact_duplicate"))),
        ],
        dtype=np.float32,
    )
    if feature_dim < 1 or feature_dim > len(values):
        raise ValueError(f"unsupported policy feature dimension: {feature_dim}")
    return values[:feature_dim]


class ResponseGate(nn.Module):
    """Cheap recurrent reply/silence policy over local semantic representations."""

    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 64,
        feature_dim: int = FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.embedding_project = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.feature_project = nn.Sequential(
            nn.Linear(feature_dim, 24), nn.SiLU(), nn.LayerNorm(24)
        )
        self.recurrent = nn.GRUCell(hidden_dim + 24, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.action_head = nn.Linear(hidden_dim, 2)
        # Preserve immediate conversation-graph evidence (mention, reply edge,
        # recent agent activity) instead of forcing it through recurrent memory.
        # The contribution remains learned jointly with the semantic state.
        self.structural_action_head = nn.Linear(feature_dim, 2)

    def step(
        self,
        embedding: torch.Tensor,
        features: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic = self.embedding_project(embedding)
        context = self.feature_project(features)
        hidden = self.recurrent(torch.cat((semantic, context), dim=-1), hidden)
        logits = self.action_head(self.dropout(hidden))
        logits = logits + self.structural_action_head(features)
        return hidden, logits


@dataclass
class PonderOutput:
    """One adaptive-depth policy decision and its auditable compute trace."""

    hidden: torch.Tensor
    logits: torch.Tensor
    logits_by_step: list[torch.Tensor]
    halt_probabilities: torch.Tensor
    halt_weights: torch.Tensor
    steps: int
    probability_path: list[float]


class PonderResponseGate(ResponseGate):
    """Response gate that reuses its recurrent core until it learns to halt.

    This is deliberately a tiny local controller, not recursive LLM prompting.
    During training every depth receives supervision and the expected depth is
    penalized. At inference, the learned halt head chooses 1..max_steps updates.
    """

    architecture = "ponder_v1"

    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 64,
        feature_dim: int = FEATURE_DIM,
        max_steps: int = 8,
        min_steps: int = 1,
        halt_threshold: float = 0.5,
    ) -> None:
        super().__init__(embedding_dim, hidden_dim, feature_dim)
        if not 1 <= min_steps <= max_steps:
            raise ValueError("ponder steps must satisfy 1 <= min_steps <= max_steps")
        self.max_steps = int(max_steps)
        self.min_steps = int(min_steps)
        self.halt_threshold = float(halt_threshold)
        self.halt_head = nn.Linear(hidden_dim + feature_dim, 1)
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, -1.0)

    @staticmethod
    def _halt_weights(halt_probabilities: torch.Tensor) -> torch.Tensor:
        """Turn conditional halt probabilities into a distribution over depth."""
        remaining = torch.ones((), device=halt_probabilities.device)
        weights: list[torch.Tensor] = []
        for index, probability in enumerate(halt_probabilities):
            if index == len(halt_probabilities) - 1:
                weight = remaining
            else:
                weight = remaining * probability
                remaining = remaining * (1.0 - probability)
            weights.append(weight)
        return torch.stack(weights)

    def ponder(
        self,
        embedding: torch.Tensor,
        features: torch.Tensor,
        hidden: torch.Tensor,
        *,
        adaptive: bool,
        forced_steps: int | None = None,
    ) -> PonderOutput:
        limit = self.max_steps if forced_steps is None else int(forced_steps)
        if not 1 <= limit <= self.max_steps:
            raise ValueError("forced ponder depth is outside the configured range")
        logits_by_step: list[torch.Tensor] = []
        hidden_by_step: list[torch.Tensor] = []
        halt_probabilities: list[torch.Tensor] = []
        probability_path: list[float] = []
        for step_index in range(limit):
            hidden, logits = self.step(embedding, features, hidden)
            halt = torch.sigmoid(
                self.halt_head(torch.cat((hidden, features), dim=-1))
            ).squeeze(-1)
            logits_by_step.append(logits)
            hidden_by_step.append(hidden)
            halt_probabilities.append(halt)
            probability_path.append(float(F.softmax(logits.detach(), dim=-1)[1]))
            if (
                adaptive
                and step_index + 1 >= self.min_steps
                and float(halt.detach()) >= self.halt_threshold
            ):
                break
        halt_tensor = torch.stack(halt_probabilities)
        halt_weights = self._halt_weights(halt_tensor)
        if adaptive or forced_steps is not None:
            selected_hidden = hidden_by_step[-1]
            selected_logits = logits_by_step[-1]
        else:
            selected_hidden = torch.stack(hidden_by_step).mul(
                halt_weights.unsqueeze(-1)
            ).sum(dim=0)
            selected_logits = torch.stack(logits_by_step).mul(
                halt_weights.unsqueeze(-1)
            ).sum(dim=0)
        return PonderOutput(
            hidden=selected_hidden,
            logits=selected_logits,
            logits_by_step=logits_by_step,
            halt_probabilities=halt_tensor,
            halt_weights=halt_weights,
            steps=len(logits_by_step),
            probability_path=probability_path,
        )


def response_gate_from_checkpoint(checkpoint: Mapping[str, Any]) -> ResponseGate:
    """Construct old and adaptive gates without breaking deployed checkpoints."""
    state = checkpoint["model"]
    feature_dim = int(
        checkpoint.get(
            "feature_dim",
            state["structural_action_head.weight"].shape[1]
            if "structural_action_head.weight" in state
            else LEGACY_FEATURE_DIM,
        )
    )
    common = {
        "embedding_dim": int(checkpoint["embedding_dim"]),
        "hidden_dim": int(checkpoint["hidden_dim"]),
        "feature_dim": feature_dim,
    }
    if checkpoint.get("architecture") == PonderResponseGate.architecture:
        ponder = dict(checkpoint.get("ponder", {}))
        model: ResponseGate = PonderResponseGate(
            **common,
            max_steps=int(ponder.get("max_steps", 8)),
            min_steps=int(ponder.get("min_steps", 1)),
            halt_threshold=float(ponder.get("halt_threshold", 0.5)),
        )
    else:
        model = ResponseGate(**common)
    model.load_state_dict(state)
    return model
