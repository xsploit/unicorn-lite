from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class RerankerOutput:
    scores: torch.Tensor
    weights: torch.Tensor
    predicted_reply: torch.Tensor


class AnswerAwareMemoryReranker(nn.Module):
    """Learn memory selection from downstream reply representations.

    The reply reader exists only to provide a differentiable training signal.
    Runtime ranking uses the selector scores and does not generate text.
    """

    architecture = "answer_aware_memory_reranker_v1"

    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.query_projection = nn.Linear(embedding_dim, hidden_dim)
        self.memory_projection = nn.Linear(embedding_dim, hidden_dim)
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.reader = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> RerankerOutput:
        if query.ndim != 2 or candidates.ndim != 3:
            raise ValueError("expected query [B,D] and candidates [B,K,D]")
        if query.shape[0] != candidates.shape[0]:
            raise ValueError("query and candidate batches differ")
        if query.shape[-1] != self.embedding_dim or candidates.shape[-1] != self.embedding_dim:
            raise ValueError("reranker embedding dimension mismatch")
        if mask is None:
            mask = torch.ones(candidates.shape[:2], dtype=torch.bool, device=query.device)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every group needs at least one candidate")

        query_normalized = F.normalize(query, dim=-1)
        candidate_normalized = F.normalize(candidates, dim=-1)
        query_hidden = torch.tanh(self.query_projection(query_normalized))
        memory_hidden = torch.tanh(self.memory_projection(candidate_normalized))
        expanded_query = query_hidden.unsqueeze(1).expand_as(memory_hidden)
        cosine = torch.sum(
            query_normalized.unsqueeze(1) * candidate_normalized, dim=-1, keepdim=True
        )
        pair = torch.cat(
            (
                expanded_query,
                memory_hidden,
                expanded_query * memory_hidden,
                torch.abs(expanded_query - memory_hidden),
                cosine,
            ),
            dim=-1,
        )
        scores = self.selector(pair).squeeze(-1).masked_fill(~mask, -1e4)
        weights = F.softmax(scores, dim=-1)
        selected_memory = torch.sum(weights.unsqueeze(-1) * memory_hidden, dim=1)
        predicted_reply = F.normalize(
            self.reader(torch.cat((query_hidden, selected_memory), dim=-1)), dim=-1
        )
        return RerankerOutput(scores=scores, weights=weights, predicted_reply=predicted_reply)

    def training_loss(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        target_reply: torch.Tensor,
        mask: torch.Tensor | None = None,
        temperature: float = 0.08,
        selector_targets: torch.Tensor | None = None,
        selector_loss_weight: float = 0.75,
    ) -> tuple[torch.Tensor, RerankerOutput]:
        output = self(query, candidates, mask)
        target = F.normalize(target_reply, dim=-1)
        cosine_loss = (1.0 - torch.sum(output.predicted_reply * target, dim=-1)).mean()
        contrastive = torch.zeros((), device=query.device)
        if query.shape[0] > 1:
            logits = output.predicted_reply @ target.T / temperature
            labels = torch.arange(query.shape[0], device=query.device)
            contrastive = F.cross_entropy(logits, labels)
        selector_loss = torch.zeros((), device=query.device)
        if selector_targets is not None:
            if selector_targets.shape != output.scores.shape:
                raise ValueError("selector targets do not match candidate scores")
            selector_loss = -(
                selector_targets * F.log_softmax(output.scores, dim=-1)
            ).sum(dim=-1).mean()
        return (
            cosine_loss + 0.25 * contrastive + selector_loss_weight * selector_loss,
            output,
        )

    @torch.no_grad()
    def rank(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        device = next(self.parameters()).device
        query_tensor = torch.as_tensor(query, dtype=torch.float32, device=device).unsqueeze(0)
        candidate_tensor = torch.as_tensor(
            candidates, dtype=torch.float32, device=device
        ).unsqueeze(0)
        output = self(query_tensor, candidate_tensor)
        scores = output.scores[0].cpu().numpy()
        order = np.argsort(-scores)
        return order, scores

    def checkpoint(self, **metadata: Any) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "model_state": self.state_dict(),
            **metadata,
        }


def reranker_from_checkpoint(
    checkpoint: Mapping[str, Any], device: str = "cpu"
) -> AnswerAwareMemoryReranker:
    if checkpoint.get("architecture") != AnswerAwareMemoryReranker.architecture:
        raise ValueError("unsupported memory-reranker checkpoint")
    model = AnswerAwareMemoryReranker(
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()
