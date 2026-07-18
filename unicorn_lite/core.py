from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class CoreOutput:
    surprise: float
    response_logits: np.ndarray
    compute_logits: np.ndarray
    state_norm: float
    memory_norm: float


class LatentModel(nn.Module):
    """Compact recurrent controller with an external fast-weight memory."""

    def __init__(self, embedding_dim: int = 384, latent_dim: int = 96) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim), nn.SiLU(), nn.LayerNorm(latent_dim)
        )
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.recurrent = nn.GRUCell(latent_dim * 2, latent_dim)
        self.predict_next = nn.Linear(latent_dim, embedding_dim)
        self.response_head = nn.Linear(latent_dim, 3)
        self.compute_head = nn.Linear(latent_dim, 4)

    def step(
        self, embedding: torch.Tensor, hidden: torch.Tensor, fast: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        x = self.project(embedding)
        key = F.normalize(self.key(x), dim=-1)
        value = torch.tanh(self.value(x))
        recalled = torch.mv(fast, key)
        new_hidden = self.recurrent(torch.cat((x, recalled), dim=-1), hidden)
        current_prediction = F.normalize(self.predict_next(hidden), dim=-1)
        next_prediction = F.normalize(self.predict_next(new_hidden), dim=-1)
        return (
            new_hidden,
            current_prediction,
            next_prediction,
            key,
            value,
            self.response_head(new_hidden),
            self.compute_head(new_hidden),
        )


class NeuralCore:
    """Owns persistent activations and a delta-rule associative memory."""

    def __init__(
        self,
        embedding_dim: int = 384,
        latent_dim: int = 96,
        device: str = "cpu",
        learning_rate: float = 0.08,
        decay: float = 0.0005,
        seed: int = 7,
    ) -> None:
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.model = LatentModel(embedding_dim, latent_dim).to(self.device)
        self.model.eval()
        self.hidden = torch.zeros(latent_dim, device=self.device)
        self.fast = torch.zeros(latent_dim, latent_dim, device=self.device)
        self.events_seen = 0
        self.learning_rate = learning_rate
        self.decay = decay
        self.checkpoint_id = (
            f"latent-v1:{embedding_dim}:{latent_dim}:seed:{seed}"
        )

    @property
    def embedding_dim(self) -> int:
        return self.model.embedding_dim

    @property
    def latent_dim(self) -> int:
        return self.model.latent_dim

    def process(self, embedding: np.ndarray) -> CoreOutput:
        item = torch.as_tensor(embedding, dtype=torch.float32, device=self.device)
        if item.numel() != self.embedding_dim:
            raise ValueError(
                f"encoder produced {item.numel()} values, expected {self.embedding_dim}"
            )
        with torch.no_grad():
            (
                hidden,
                predicted,
                _,
                key,
                value,
                response,
                compute,
            ) = self.model.step(item, self.hidden, self.fast)
            target = F.normalize(item, dim=-1)
            surprise = float((1.0 - F.cosine_similarity(predicted, target, dim=0)) / 2.0)
            recalled = torch.mv(self.fast, key)
            error = value - recalled
            adaptive_rate = self.learning_rate * (0.25 + 0.75 * surprise)
            self.fast.mul_(1.0 - self.decay).add_(
                torch.outer(error, key), alpha=adaptive_rate
            )
            self.hidden = hidden
            self.events_seen += 1
            return CoreOutput(
                surprise=max(0.0, min(1.0, surprise)),
                response_logits=response.cpu().numpy(),
                compute_logits=compute.cpu().numpy(),
                state_norm=float(torch.linalg.vector_norm(self.hidden)),
                memory_norm=float(torch.linalg.vector_norm(self.fast)),
            )

    def export_state(self) -> tuple[np.ndarray, np.ndarray, int]:
        return (
            self.hidden.detach().cpu().numpy().astype(np.float32),
            self.fast.detach().cpu().numpy().astype(np.float32),
            self.events_seen,
        )

    def import_state(
        self, hidden: np.ndarray, fast: np.ndarray, events_seen: int
    ) -> None:
        if hidden.shape != (self.latent_dim,):
            raise ValueError("stored hidden state has the wrong shape")
        if fast.shape != (self.latent_dim, self.latent_dim):
            raise ValueError("stored fast memory has the wrong shape")
        self.hidden = torch.as_tensor(hidden, device=self.device).clone()
        self.fast = torch.as_tensor(fast, device=self.device).clone()
        self.events_seen = events_seen

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.checkpoint_id = f"sha256:{digest}"
