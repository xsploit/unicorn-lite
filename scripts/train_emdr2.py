from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unicorn_lite.emdr2 import DiscordEMDR2


def _rows(path: str | Path, max_groups: int = 0) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_groups > 0:
        values = values[:max_groups]
    return values


@torch.no_grad()
def _retrieval_order(
    model: DiscordEMDR2, row: dict[str, Any], chunk_size: int = 64
) -> list[int]:
    candidates = list(row["candidates"])
    query = model.encode_queries([str(row["query"])], 192)
    scores: list[torch.Tensor] = []
    for offset in range(0, len(candidates), chunk_size):
        block = candidates[offset : offset + chunk_size]
        memory = model.encode_memories(
            [str(candidate["content"]) for candidate in block], 192
        )
        scores.append((query @ memory.T)[0].float().cpu())
    values = torch.cat(scores)
    return torch.argsort(values, descending=True).tolist()


@torch.no_grad()
def _refresh_pools(
    model: DiscordEMDR2,
    rows: list[dict[str, Any]],
    candidate_pool: int,
) -> list[dict[str, Any]]:
    model.eval()
    query_texts = list(dict.fromkeys(str(row["query"]) for row in rows))
    memory_texts = list(
        dict.fromkeys(
            str(candidate["content"])
            for row in rows
            for candidate in row["candidates"]
        )
    )

    def encode_unique(texts: list[str], *, query: bool) -> dict[str, np.ndarray]:
        vectors: dict[str, np.ndarray] = {}
        for offset in range(0, len(texts), 128):
            block = texts[offset : offset + 128]
            encoded = (
                model.encode_queries(block, 192)
                if query
                else model.encode_memories(block, 192)
            ).float().cpu().numpy()
            vectors.update(zip(block, encoded))
        return vectors

    query_vectors = encode_unique(query_texts, query=True)
    memory_vectors = encode_unique(memory_texts, query=False)
    prepared: list[dict[str, Any]] = []
    for row in rows:
        query = query_vectors[str(row["query"])]
        candidates = np.stack(
            [memory_vectors[str(candidate["content"])] for candidate in row["candidates"]]
        )
        order = np.argsort(-(candidates @ query)).tolist()
        selected = order[: min(candidate_pool, len(order))]
        prepared.append(
            {
                **row,
                "candidates": [row["candidates"][index] for index in selected],
            }
        )
    return prepared


@torch.no_grad()
def _baseline_ids(
    model: DiscordEMDR2, rows: list[dict[str, Any]]
) -> dict[str, str]:
    model.eval()
    return {
        str(row["query_id"]): str(
            row["candidates"][_retrieval_order(model, row)[0]]["event_id"]
        )
        for row in rows
    }


@torch.no_grad()
def _evaluate(
    model: DiscordEMDR2,
    rows: list[dict[str, Any]],
    baseline_ids: dict[str, str],
    *,
    top_k: int,
    maximum_examples: int,
) -> dict[str, float]:
    model.eval()
    reader_losses: list[float] = []
    learned_likelihoods: list[float] = []
    baseline_likelihoods: list[float] = []
    wins: list[float] = []
    for row in rows[:maximum_examples]:
        candidates = list(row["candidates"])
        order = _retrieval_order(model, row)
        learned_index = order[0]
        baseline_id = baseline_ids[str(row["query_id"])]
        baseline_index = next(
            index
            for index, candidate in enumerate(candidates)
            if str(candidate["event_id"]) == baseline_id
        )
        likelihood = model.answer_log_likelihood(
            query=str(row["query"]),
            reply=str(row["reply"]),
            candidates=candidates,
            indices=[learned_index, baseline_index],
        ).float().cpu()
        learned = float(likelihood[0])
        baseline = float(likelihood[1])
        learned_likelihoods.append(learned)
        baseline_likelihoods.append(baseline)
        wins.append(float(learned > baseline) + 0.5 * float(learned == baseline))

        pool_indices = order[: min(24, len(order))]
        pool = [{**candidates[index]} for index in pool_indices]
        step = model.training_step(
            query=str(row["query"]),
            reply=str(row["reply"]),
            candidates=pool,
            top_k=top_k,
        )
        reader_losses.append(float(step.reader_loss.float().cpu()))
    learned_mean = float(np.mean(learned_likelihoods))
    baseline_mean = float(np.mean(baseline_likelihoods))
    return {
        "examples": float(len(reader_losses)),
        "reader_nll": float(np.mean(reader_losses)),
        "reader_perplexity": float(math.exp(min(20.0, np.mean(reader_losses)))),
        "learned_top1_answer_log_likelihood": learned_mean,
        "initial_top1_answer_log_likelihood": baseline_mean,
        "answer_log_likelihood_gain": learned_mean - baseline_mean,
        "selection_win_rate": float(np.mean(wins)),
    }


def train(
    groups_path: str | Path,
    output_path: str | Path,
    *,
    retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    reader_model: str = "google/flan-t5-small",
    device: str = "cuda",
    epochs: int = 2,
    candidate_pool: int = 24,
    top_k: int = 4,
    gradient_accumulation: int = 8,
    retriever_learning_rate: float = 2e-5,
    reader_learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    refresh_every_epochs: int = 1,
    validation_examples: int = 48,
    max_groups: int = 0,
    precision: str = "bf16",
    seed: int = 53,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    rows = _rows(groups_path, max_groups)
    if len(rows) < 40:
        raise ValueError("at least 40 chronological EMDR2 groups are required")
    split = max(1, int(len(rows) * 0.8))
    train_rows = rows[:split]
    validation_rows = rows[split:]
    model = DiscordEMDR2(
        retriever_model,
        reader_model,
        device=device,
        gradient_checkpointing=True,
    )
    baseline = _baseline_ids(model, validation_rows)
    prepared_train = _refresh_pools(model, train_rows, candidate_pool)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.query_encoder.parameters())
                + list(model.memory_encoder.parameters()),
                "lr": retriever_learning_rate,
            },
            {"params": model.reader.parameters(), "lr": reader_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    use_cuda_amp = device.startswith("cuda") and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_cuda_amp and precision == "fp16"
    )
    history: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        if epoch and refresh_every_epochs > 0 and epoch % refresh_every_epochs == 0:
            prepared_train = _refresh_pools(model, train_rows, candidate_pool)
        order = list(range(len(prepared_train)))
        random.shuffle(order)
        model.train()
        total_loss = 0.0
        total_reader = 0.0
        total_retriever = 0.0
        for position, row_index in enumerate(order, 1):
            row = prepared_train[row_index]
            context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_cuda_amp
                else nullcontext()
            )
            with context:
                step = model.training_step(
                    query=str(row["query"]),
                    reply=str(row["reply"]),
                    candidates=list(row["candidates"]),
                    top_k=top_k,
                )
                scaled_loss = step.loss / gradient_accumulation
            scaler.scale(scaled_loss).backward()
            if position % gradient_accumulation == 0 or position == len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(step.loss.detach().float().cpu())
            total_reader += float(step.reader_loss.detach().float().cpu())
            total_retriever += float(step.retriever_loss.detach().float().cpu())
            if position % 50 == 0 or position == len(order):
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": position,
                            "steps": len(order),
                            "loss": total_loss / position,
                            "reader_loss": total_reader / position,
                            "retriever_loss": total_retriever / position,
                        }
                    ),
                    flush=True,
                )
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": total_loss / len(order),
                "reader_loss": total_reader / len(order),
                "retriever_loss": total_retriever / len(order),
            }
        )

    metrics = _evaluate(
        model,
        validation_rows,
        baseline,
        top_k=top_k,
        maximum_examples=min(validation_examples, len(validation_rows)),
    )
    accepted = bool(
        metrics["examples"] >= 32
        and math.isfinite(metrics["reader_nll"])
        and metrics["answer_log_likelihood_gain"] > 0.0
        and metrics["selection_win_rate"] > 0.5
    )
    report: dict[str, Any] = {
        "source_groups": len(rows),
        "training_groups": len(train_rows),
        "validation_groups": len(validation_rows),
        "epochs": epochs,
        "candidate_pool": candidate_pool,
        "top_k": top_k,
        "gradient_accumulation": gradient_accumulation,
        "precision": precision,
        "seed": seed,
        "history": history,
        "metrics": metrics,
        "accepted_retriever_for_runtime": accepted,
        "accepted_reader_for_runtime": False,
    }
    destination = model.save_checkpoint(output_path, report)
    (destination / "training-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({**report, "output": str(destination.resolve())}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a reduced shared-reader EMDR2 model on causal Discord memory"
    )
    parser.add_argument("groups_jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--retriever-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--reader-model", default="google/flan-t5-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--candidate-pool", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--retriever-learning-rate", type=float, default=2e-5)
    parser.add_argument("--reader-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--refresh-every-epochs", type=int, default=1)
    parser.add_argument("--validation-examples", type=int, default=48)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    train(
        args.groups_jsonl,
        args.output,
        retriever_model=args.retriever_model,
        reader_model=args.reader_model,
        device=args.device,
        epochs=args.epochs,
        candidate_pool=args.candidate_pool,
        top_k=args.top_k,
        gradient_accumulation=args.gradient_accumulation,
        retriever_learning_rate=args.retriever_learning_rate,
        reader_learning_rate=args.reader_learning_rate,
        weight_decay=args.weight_decay,
        refresh_every_epochs=args.refresh_every_epochs,
        validation_examples=args.validation_examples,
        max_groups=args.max_groups,
        precision=args.precision,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
