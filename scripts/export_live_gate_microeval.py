from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from unicorn_lite.response_gate import (
    answered_refractory_features,
    normalize_policy_text,
)


def _vector(blob: bytes, dimension: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(dimension)


def export_cases(
    database: str | Path,
    cases: list[tuple[str, str]],
    output: str | Path,
) -> dict[str, object]:
    connection = sqlite3.connect(Path(database))
    connection.row_factory = sqlite3.Row
    exported: list[dict[str, object]] = []
    for discord_message_id, action in cases:
        event = connection.execute(
            """
            SELECT * FROM events
            WHERE json_extract(metadata_json, '$.discord_message_id')=?
            ORDER BY occurred_at DESC LIMIT 1
            """,
            (discord_message_id,),
        ).fetchone()
        if event is None:
            raise ValueError(f"Discord message was not found: {discord_message_id}")
        metadata = json.loads(event["metadata_json"])
        query = _vector(event["embedding"], int(event["embedding_dim"]))
        query_norm = float(np.linalg.norm(query)) or 1.0
        prior = connection.execute(
            """
            SELECT e.content, e.occurred_at, e.embedding, e.embedding_dim,
                   (SELECT d.action FROM decisions d WHERE d.event_id=e.event_id
                    ORDER BY d.id DESC LIMIT 1) AS decision_action
            FROM events e
            WHERE e.occurred_at < ? AND e.embedding_dim=?
            """,
            (event["occurred_at"], event["embedding_dim"]),
        ).fetchall()
        matches: list[tuple[float, sqlite3.Row]] = []
        for row in prior:
            vector = _vector(row["embedding"], int(row["embedding_dim"]))
            similarity = float(
                np.dot(query, vector)
                / (query_norm * (float(np.linalg.norm(vector)) or 1.0))
            )
            matches.append((similarity, row))
        matches.sort(key=lambda item: item[0], reverse=True)
        matches = matches[:5]
        normalized = normalize_policy_text(str(event["content"])).casefold()
        exact = [
            row
            for _, row in matches
            if normalize_policy_text(str(row["content"])).casefold() == normalized
        ]
        answered = [
            (similarity, row)
            for similarity, row in matches
            if str(row["decision_action"] or "").startswith("REPLY")
        ]
        now = datetime.fromisoformat(str(event["occurred_at"]).replace("Z", "+00:00"))
        refractory = answered_refractory_features(
            now,
            (
                (
                    similarity,
                    datetime.fromisoformat(
                        str(row["occurred_at"]).replace("Z", "+00:00")
                    ),
                )
                for similarity, row in answered
            ),
        )
        metadata.update(
            {
                "content": event["content"],
                "model_input": metadata.get("policy_input") or event["content"],
                "occurred_at": event["occurred_at"],
                "channel": "live-repeat-heldout",
                "discord_message_id": discord_message_id,
                "action": action,
                "max_memory_similarity": matches[0][0] if matches else 0.0,
                "mean_top3_memory_similarity": (
                    sum(value for value, _ in matches[:3]) / len(matches[:3])
                    if matches
                    else 0.0
                ),
                "near_duplicate_count": sum(value >= 0.90 for value, _ in matches),
                "exact_duplicate": bool(exact),
                "max_answered_similarity": max(
                    (value for value, _ in answered), default=0.0
                ),
                "recent_answered_exact_duplicate": any(
                    0.0
                    <= (
                        now
                        - datetime.fromisoformat(
                            str(row["occurred_at"]).replace("Z", "+00:00")
                        )
                    ).total_seconds()
                    <= 86400.0
                    and str(row["decision_action"] or "").startswith("REPLY")
                    for row in exact
                ),
                **refractory,
            }
        )
        exported.append(metadata)
    connection.close()
    destination = Path(output)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exported),
        encoding="utf-8",
    )
    report = {
        "database": str(Path(database).resolve()),
        "output": str(destination.resolve()),
        "cases": len(exported),
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export held-out live Discord decisions as causal gate JSONL"
    )
    parser.add_argument("database")
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases: list[tuple[str, str]] = []
    for value in args.case:
        message_id, separator, action = value.partition("=")
        if not separator or action not in {"OBSERVE", "REPLY_FLASH"}:
            raise ValueError("cases must use DISCORD_MESSAGE_ID=OBSERVE|REPLY_FLASH")
        cases.append((message_id, action))
    export_cases(args.database, cases, args.output)


if __name__ == "__main__":
    main()
