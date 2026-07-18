from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from unicorn_lite.encoder import encoder_identity, make_encoder
from unicorn_lite.memory_reranker import AnswerAwareMemoryReranker


def _rows(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _embed_texts(rows: list[dict[str, Any]], encoder: Any) -> dict[str, np.ndarray]:
    texts: dict[str, None] = {}
    for row in rows:
        texts[str(row["query"])] = None
        texts[str(row["reply"])] = None
        for candidate in row["candidates"]:
            texts[str(candidate["content"])] = None
    ordered = list(texts)
    matrix = encoder.encode_many(ordered, batch_size=128)
    return {text: matrix[index] for index, text in enumerate(ordered)}


def _prepare(
    rows: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
    candidate_limit: int,
    minimum_memory_lift: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for row in rows:
        query = vectors[str(row["query"])]
        reply = vectors[str(row["reply"])]
        candidates = np.stack(
            [vectors[str(candidate["content"])] for candidate in row["candidates"]]
        )
        cosine = candidates @ query
        order = np.argsort(-cosine)[:candidate_limit]
        selected = candidates[order]
        memory_lift = float(np.max(selected @ reply) - np.dot(query, reply))
        if memory_lift >= minimum_memory_lift:
            prepared.append((query, selected, reply))
    return prepared


def _batch(
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    indices: list[int],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(groups[index][1].shape[0] for index in indices)
    dimension = groups[indices[0]][0].shape[0]
    queries = np.stack([groups[index][0] for index in indices])
    replies = np.stack([groups[index][2] for index in indices])
    candidates = np.zeros((len(indices), width, dimension), dtype=np.float32)
    mask = np.zeros((len(indices), width), dtype=bool)
    for batch_index, group_index in enumerate(indices):
        values = groups[group_index][1]
        candidates[batch_index, : len(values)] = values
        mask[batch_index, : len(values)] = True
    return (
        torch.as_tensor(queries, dtype=torch.float32, device=device),
        torch.as_tensor(candidates, dtype=torch.float32, device=device),
        torch.as_tensor(replies, dtype=torch.float32, device=device),
        torch.as_tensor(mask, dtype=torch.bool, device=device),
    )


def _answer_aware_targets(
    query: torch.Tensor,
    candidates: torch.Tensor,
    reply: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Approximate the latent useful-memory posterior using the known reply."""
    query_similarity = torch.sum(
        F.normalize(candidates, dim=-1) * F.normalize(query, dim=-1).unsqueeze(1),
        dim=-1,
    )
    answer_similarity = torch.sum(
        F.normalize(candidates, dim=-1) * F.normalize(reply, dim=-1).unsqueeze(1),
        dim=-1,
    )
    logits = (0.25 * query_similarity + 0.75 * answer_similarity) / temperature
    return F.softmax(logits.masked_fill(~mask, -1e4), dim=-1)


@torch.no_grad()
def _evaluate(
    model: AnswerAwareMemoryReranker,
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    indices: list[int],
    device: str,
) -> dict[str, float]:
    reply_cosines: list[float] = []
    query_cosines: list[float] = []
    learned_target: list[float] = []
    cosine_target: list[float] = []
    learned_rr: list[float] = []
    cosine_rr: list[float] = []
    for index in indices:
        query, candidates, reply = groups[index]
        q = torch.as_tensor(query, dtype=torch.float32, device=device).unsqueeze(0)
        c = torch.as_tensor(candidates, dtype=torch.float32, device=device).unsqueeze(0)
        output = model(q, c)
        scores = output.scores[0].cpu().numpy()
        target_similarity = candidates @ reply
        query_similarity = candidates @ query
        oracle = int(np.argmax(target_similarity))
        learned_order = np.argsort(-scores)
        cosine_order = np.argsort(-query_similarity)
        learned_rank = int(np.flatnonzero(learned_order == oracle)[0]) + 1
        cosine_rank = int(np.flatnonzero(cosine_order == oracle)[0]) + 1
        learned_target.append(float(target_similarity[learned_order[0]]))
        cosine_target.append(float(target_similarity[cosine_order[0]]))
        learned_rr.append(1.0 / learned_rank)
        cosine_rr.append(1.0 / cosine_rank)
        reply_cosines.append(
            float(F.cosine_similarity(output.predicted_reply[0], torch.as_tensor(reply, device=device), dim=0))
        )
        query_cosines.append(float(np.dot(query, reply)))
    return {
        "examples": float(len(indices)),
        "predicted_reply_cosine": float(np.mean(reply_cosines)),
        "query_only_reply_cosine": float(np.mean(query_cosines)),
        "learned_top1_target_similarity": float(np.mean(learned_target)),
        "cosine_top1_target_similarity": float(np.mean(cosine_target)),
        "learned_oracle_mrr": float(np.mean(learned_rr)),
        "cosine_oracle_mrr": float(np.mean(cosine_rr)),
    }


def train(
    groups_path: str | Path,
    output_path: str | Path,
    *,
    encoder_name: str = "local",
    device: str = "cuda",
    candidate_limit: int = 20,
    hidden_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 24,
    learning_rate: float = 2e-3,
    minimum_memory_lift: float = 0.05,
    seed: int = 23,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows = _rows(groups_path)
    if len(rows) < 20:
        raise ValueError("at least 20 chronological reply groups are required")
    encoder = make_encoder(encoder_name, device=device)
    vectors = _embed_texts(rows, encoder)
    groups = _prepare(rows, vectors, candidate_limit, minimum_memory_lift)
    if len(groups) < 20:
        raise ValueError("fewer than 20 memory-dependent groups survived filtering")
    split = max(1, int(len(groups) * 0.8))
    train_indices = list(range(split))
    validation_indices = list(range(split, len(groups)))
    model = AnswerAwareMemoryReranker(encoder.dimension, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    for _ in range(epochs):
        random.shuffle(train_indices)
        model.train()
        epoch_losses: list[float] = []
        for offset in range(0, len(train_indices), batch_size):
            selected = train_indices[offset : offset + batch_size]
            query, candidates, reply, mask = _batch(groups, selected, device)
            targets = _answer_aware_targets(query, candidates, reply, mask)
            loss, _ = model.training_loss(
                query, candidates, reply, mask, selector_targets=targets
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))

    model.eval()
    metrics = _evaluate(model, groups, validation_indices, device)
    accepted = bool(
        metrics["learned_oracle_mrr"] > metrics["cosine_oracle_mrr"]
        and metrics["learned_top1_target_similarity"]
        > metrics["cosine_top1_target_similarity"]
    )
    checkpoint = model.checkpoint(
        encoder=encoder_identity(encoder),
        candidate_limit=candidate_limit,
        minimum_memory_lift=minimum_memory_lift,
        training_groups=len(train_indices),
        validation_groups=len(validation_indices),
        accepted_for_runtime=accepted,
        metrics=metrics,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    report: dict[str, Any] = {
        "groups": len(groups),
        "source_groups": len(rows),
        "training_groups": len(train_indices),
        "validation_groups": len(validation_indices),
        "encoder": encoder_identity(encoder),
        "candidate_limit": candidate_limit,
        "minimum_memory_lift": minimum_memory_lift,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "accepted_for_runtime": accepted,
        "metrics": metrics,
        "output": str(destination.resolve()),
    }
    destination.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an answer-supervised local memory reranker"
    )
    parser.add_argument("groups_jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--minimum-memory-lift", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    train(
        args.groups_jsonl,
        args.output,
        encoder_name=args.encoder,
        device=args.device,
        candidate_limit=args.candidate_limit,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        minimum_memory_lift=args.minimum_memory_lift,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
