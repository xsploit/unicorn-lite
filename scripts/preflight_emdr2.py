from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import torch

from unicorn_lite.agent import UnicornAgent
from unicorn_lite.encoder import make_encoder
from unicorn_lite.models import Experience


def _backup(source: Path, destination: Path) -> None:
    with sqlite3.connect(source) as original, sqlite3.connect(destination) as copy:
        original.backup(copy)


def preflight(
    database: str | Path,
    emdr2_checkpoint: str | Path,
    *,
    core_checkpoint: str | None = None,
    encoder_name: str = "local",
    device: str = "cuda",
) -> dict[str, object]:
    source = Path(database)
    with tempfile.TemporaryDirectory(prefix="unicorn-emdr2-preflight-") as directory:
        baseline_db = Path(directory) / "baseline.db"
        challenger_db = Path(directory) / "challenger.db"
        _backup(source, baseline_db)
        _backup(source, challenger_db)
        encoder = make_encoder(encoder_name, device=device)
        baseline = UnicornAgent(
            baseline_db, encoder, device=device, checkpoint=core_checkpoint
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        startup = time.perf_counter()
        challenger = UnicornAgent(
            challenger_db,
            encoder,
            device=device,
            checkpoint=core_checkpoint,
            emdr2_checkpoint=str(emdr2_checkpoint),
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        startup_ms = (time.perf_counter() - startup) * 1000.0

        prompt = "What past interaction would help you understand how I test your memory?"
        baseline_event = Experience(prompt, metadata={"direct": True})
        challenger_event = Experience(prompt, metadata={"direct": True})
        baseline_result = asyncio.run(baseline.ingest(baseline_event))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        challenger_result = asyncio.run(challenger.ingest(challenger_event))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ingest_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        for _ in range(20):
            challenger.emdr2_retriever.retrieve(prompt, limit=5)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        retrieval_ms = (time.perf_counter() - started) * 50.0
        metadata = challenger_event.metadata.get("emdr2", {})
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
                and isinstance(metadata, dict)
                and bool(metadata.get("applied"))
                and retrieval_ms < 100.0
            ),
            "decision": challenger_result.decision.action,
            "decision_confidence_delta": abs(
                baseline_result.decision.confidence
                - challenger_result.decision.confidence
            ),
            "surprise_delta": abs(
                baseline_result.surprise - challenger_result.surprise
            ),
            "startup_index_ms": startup_ms,
            "challenger_ingest_ms": ingest_ms,
            "retrieval_latency_ms": retrieval_ms,
            "indexed_memories": (
                int(metadata.get("indexed_memories", 0))
                if isinstance(metadata, dict)
                else 0
            ),
            "changed_cosine_top5": (
                bool(metadata.get("changed_top5"))
                if isinstance(metadata, dict)
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
        description="Preflight a Discord-scale EMDR2 retriever on database backups"
    )
    parser.add_argument("database")
    parser.add_argument("emdr2_checkpoint")
    parser.add_argument("--core-checkpoint")
    parser.add_argument("--encoder", default="local")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = preflight(
        args.database,
        args.emdr2_checkpoint,
        core_checkpoint=args.core_checkpoint,
        encoder_name=args.encoder,
        device=args.device,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
