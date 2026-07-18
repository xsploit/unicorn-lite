from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from unicorn_lite.agent import UnicornAgent
from unicorn_lite.encoder import make_encoder
from unicorn_lite.models import Experience


def _backup(source: Path, destination: Path) -> None:
    with sqlite3.connect(source) as original, sqlite3.connect(destination) as copy:
        original.backup(copy)


def preflight(
    database: str | Path,
    reranker_checkpoint: str | Path,
    *,
    core_checkpoint: str | None = None,
    encoder_name: str = "local",
    device: str = "cuda",
) -> dict[str, object]:
    source = Path(database)
    with tempfile.TemporaryDirectory(prefix="unicorn-reranker-preflight-") as directory:
        baseline_db = Path(directory) / "baseline.db"
        challenger_db = Path(directory) / "challenger.db"
        _backup(source, baseline_db)
        _backup(source, challenger_db)
        encoder = make_encoder(encoder_name, device=device)
        baseline = UnicornAgent(
            baseline_db,
            encoder,
            device=device,
            checkpoint=core_checkpoint,
        )
        challenger = UnicornAgent(
            challenger_db,
            encoder,
            device=device,
            checkpoint=core_checkpoint,
            memory_reranker_checkpoint=str(reranker_checkpoint),
        )
        prompt = "What past interaction would help you understand how I test your memory?"
        embedding = encoder.encode(prompt)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        baseline_event = Experience(prompt, metadata={"direct": True})
        challenger_event = Experience(prompt, metadata={"direct": True})
        started = time.perf_counter()
        baseline_result = asyncio.run(baseline.ingest(baseline_event))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        baseline_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        challenger_result = asyncio.run(challenger.ingest(challenger_event))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        challenger_ms = (time.perf_counter() - started) * 1000.0
        reranker = challenger_event.metadata.get("memory_reranker", {})
        candidate_hits = challenger.store.retrieve(embedding, limit=20)
        candidate_vectors = np.stack([hit.embedding for hit in candidate_hits])
        challenger.memory_reranker.rank(embedding, candidate_vectors)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        selector_started = time.perf_counter()
        for _ in range(100):
            challenger.memory_reranker.rank(embedding, candidate_vectors)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        selector_ms = (time.perf_counter() - selector_started) * 10.0
        report: dict[str, object] = {
            "passed": bool(
                baseline_result.decision.action == challenger_result.decision.action
                and abs(
                    baseline_result.decision.confidence
                    - challenger_result.decision.confidence
                )
                < 1e-7
                and abs(baseline_result.surprise - challenger_result.surprise) < 1e-7
                and baseline_result.reply is None
                and challenger_result.reply is None
                and isinstance(reranker, dict)
                and bool(reranker.get("applied"))
                and selector_ms < 10.0
            ),
            "decision": challenger_result.decision.action,
            "decision_confidence_delta": abs(
                baseline_result.decision.confidence
                - challenger_result.decision.confidence
            ),
            "surprise_delta": abs(
                baseline_result.surprise - challenger_result.surprise
            ),
            "baseline_latency_ms": baseline_ms,
            "challenger_latency_ms": challenger_ms,
            "additional_latency_ms": challenger_ms - baseline_ms,
            "selector_latency_ms": selector_ms,
            "candidate_count": (
                int(reranker.get("candidates", 0)) if isinstance(reranker, dict) else 0
            ),
            "changed_cosine_top5": (
                bool(reranker.get("changed_top5"))
                if isinstance(reranker, dict)
                else False
            ),
            "writer_called": False,
        }
        baseline.close()
        challenger.close()
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight an accepted reranker against a backup of a live database"
    )
    parser.add_argument("database")
    parser.add_argument("reranker_checkpoint")
    parser.add_argument("--core-checkpoint")
    parser.add_argument("--encoder", default="local")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = preflight(
        args.database,
        args.reranker_checkpoint,
        core_checkpoint=args.core_checkpoint,
        encoder_name=args.encoder,
        device=args.device,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
