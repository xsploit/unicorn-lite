from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .core import LatentModel
from .encoder import Encoder


ACTION_LABELS = {"OBSERVE": 0, "REPLY_FLASH": 1, "REPLY_PRO": 2}


@dataclass(slots=True)
class NeuroPair:
    user: str
    assistant: str
    action: int
    compute: int


def _complex_request(text: str) -> bool:
    lowered = text.lower()
    words = text.split()
    return (
        len(words) >= 90
        or text.count("\n") >= 8
        or "```" in text
        or sum(
            token in lowered
            for token in ("research", "debug", "architecture", "analyze")
        )
        >= 2
    )


def load_neuro_pairs(path: str | Path) -> list[NeuroPair]:
    pairs: list[NeuroPair] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages", [])
        user = next(
            (str(item.get("content", "")) for item in messages if item.get("role") == "user"),
            "",
        )
        assistant = next(
            (
                str(item.get("content", ""))
                for item in messages
                if item.get("role") == "assistant"
            ),
            "",
        )
        if not user or not assistant:
            raise ValueError(f"row {line_number} lacks a user/assistant pair")
        pro = _complex_request(user)
        pairs.append(
            NeuroPair(
                user=user,
                assistant=assistant,
                action=ACTION_LABELS["REPLY_PRO" if pro else "REPLY_FLASH"],
                compute=3 if pro else 2,
            )
        )
    if not pairs:
        raise ValueError("no user/assistant pairs found")
    return pairs


def _encode_pairs(
    pairs: list[NeuroPair], encoder: Encoder, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    user = encoder.encode_many([pair.user for pair in pairs], batch_size=batch_size)
    assistant = encoder.encode_many(
        [pair.assistant for pair in pairs], batch_size=batch_size
    )
    return torch.from_numpy(user), torch.from_numpy(assistant)


def _evaluate_pairs(
    model: LatentModel,
    users: torch.Tensor,
    assistants: torch.Tensor,
    pairs: list[NeuroPair],
    device: str,
) -> dict[str, float]:
    model.eval()
    cosines: list[float] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    action_correct = 0
    compute_correct = 0
    with torch.no_grad():
        for index, pair in enumerate(pairs):
            hidden = torch.zeros(model.latent_dim, device=device)
            fast = torch.zeros(model.latent_dim, model.latent_dim, device=device)
            hidden, _, prediction, _, _, response, compute = model.step(
                users[index].to(device), hidden, fast
            )
            del hidden
            target = F.normalize(assistants[index].to(device), dim=-1)
            cosines.append(float(F.cosine_similarity(prediction, target, dim=0)))
            predictions.append(prediction.detach().cpu())
            targets.append(target.detach().cpu())
            action_correct += int(response.argmax().item() == pair.action)
            compute_correct += int(compute.argmax().item() == pair.compute)
    count = max(1, len(pairs))
    prediction_matrix = F.normalize(torch.stack(predictions), dim=-1)
    target_matrix = F.normalize(torch.stack(targets), dim=-1)
    similarities = prediction_matrix @ target_matrix.T
    sorted_targets = similarities.argsort(dim=1, descending=True)
    expected = torch.arange(len(pairs)).unsqueeze(1)
    ranks = (sorted_targets == expected).nonzero()[:, 1] + 1
    shifted = torch.roll(torch.arange(len(pairs)), shifts=1)
    return {
        "next_embedding_cosine": float(np.mean(cosines)),
        "shuffled_reply_cosine": float(
            similarities[torch.arange(len(pairs)), shifted].mean()
        ),
        "reply_retrieval_top1": float((ranks <= 1).float().mean()),
        "reply_retrieval_top5": float((ranks <= 5).float().mean()),
        "reply_retrieval_mrr": float((1.0 / ranks.float()).mean()),
        "response_accuracy": action_correct / count,
        "compute_accuracy": compute_correct / count,
    }


def train_neuro_pairs(
    train_path: str | Path,
    validation_path: str | Path,
    encoder: Encoder,
    output_path: str | Path,
    device: str = "cpu",
    latent_dim: int = 96,
    epochs: int = 12,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    seed: int = 7,
) -> dict[str, object]:
    """Train the controller on independent user -> assistant transitions."""
    random.seed(seed)
    torch.manual_seed(seed)
    train_pairs = load_neuro_pairs(train_path)
    validation_pairs = load_neuro_pairs(validation_path)
    train_users, train_assistants = _encode_pairs(train_pairs, encoder, batch_size)
    val_users, val_assistants = _encode_pairs(validation_pairs, encoder, batch_size)

    model = LatentModel(encoder.dimension, latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    baseline = _evaluate_pairs(
        model, val_users, val_assistants, validation_pairs, device
    )
    final_loss = 0.0
    indices = list(range(len(train_pairs)))
    for _ in range(epochs):
        random.shuffle(indices)
        model.train()
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset : offset + batch_size]
            losses: list[torch.Tensor] = []
            for index in selected:
                pair = train_pairs[index]
                hidden = torch.zeros(latent_dim, device=device)
                fast = torch.zeros(latent_dim, latent_dim, device=device)
                hidden, _, prediction, _, _, response, compute = model.step(
                    train_users[index].to(device), hidden, fast
                )
                del hidden
                target = F.normalize(train_assistants[index].to(device), dim=-1)
                prediction_loss = 1.0 - F.cosine_similarity(
                    prediction, target, dim=0
                )
                action = torch.tensor([pair.action], device=device)
                compute_label = torch.tensor([pair.compute], device=device)
                losses.extend(
                    (
                        prediction_loss,
                        F.cross_entropy(response.unsqueeze(0), action) * 0.25,
                        F.cross_entropy(compute.unsqueeze(0), compute_label) * 0.1,
                    )
                )
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach())

    validation = _evaluate_pairs(
        model, val_users, val_assistants, validation_pairs, device
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    report: dict[str, object] = {
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "epochs": epochs,
        "batch_size": batch_size,
        "final_batch_loss": final_loss,
        "baseline_validation": baseline,
        "trained_validation": validation,
        "checkpoint": str(output.resolve()),
        "scope": (
            "learns user-to-assistant latent transitions and direct-response cost tiers; "
            "does not learn long-term chronology or safe silence from independent Q/A pairs"
        ),
    }
    output.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def train_replay(
    jsonl_path: str | Path,
    encoder: Encoder,
    output_path: str | Path,
    device: str = "cpu",
    latent_dim: int = 96,
    epochs: int = 3,
    learning_rate: float = 2e-3,
) -> dict[str, float | int]:
    """Train next-event prediction plus optional response labels on chronology."""
    records = [
        json.loads(line)
        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) < 2:
        raise ValueError("replay training needs at least two JSONL records")
    encoded = encoder.encode_many([str(row["content"]) for row in records])
    vectors = torch.as_tensor(encoded, device=device)
    model = LatentModel(encoder.dimension, latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    final_loss = 0.0
    for _ in range(epochs):
        hidden = torch.zeros(latent_dim, device=device)
        fast = torch.zeros(latent_dim, latent_dim, device=device)
        losses: list[torch.Tensor] = []
        for index in range(len(vectors) - 1):
            hidden, _, prediction, key, value, response, _ = model.step(
                vectors[index], hidden, fast
            )
            target = F.normalize(vectors[index + 1], dim=-1)
            losses.append(1.0 - F.cosine_similarity(prediction, target, dim=0))
            label_name = records[index].get("action")
            if label_name in ACTION_LABELS:
                label = torch.tensor([ACTION_LABELS[label_name]], device=device)
                losses.append(F.cross_entropy(response.unsqueeze(0), label) * 0.35)
            recalled = torch.mv(fast, key)
            fast = (fast * 0.999 + 0.08 * torch.outer(value - recalled, key)).detach()
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return {"records": len(records), "epochs": epochs, "loss": final_loss}
