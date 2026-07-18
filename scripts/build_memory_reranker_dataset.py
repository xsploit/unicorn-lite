from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_histories(paths: list[str | Path]) -> list[dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                by_id[str(row["message_id"])] = row
    return sorted(
        by_id.values(), key=lambda row: (str(row["occurred_at"]), str(row["message_id"]))
    )


def build_groups(
    history_paths: list[str | Path],
    target_author_id: str,
    output_path: str | Path,
    *,
    candidate_pool: int = 128,
    minimum_candidates: int = 4,
) -> dict[str, object]:
    history = _read_histories(history_paths)
    by_id = {str(row["message_id"]): index for index, row in enumerate(history)}
    channel_history: dict[str, list[int]] = {}
    groups: list[dict[str, object]] = []
    for index, row in enumerate(history):
        channel = str(row["channel_id"])
        prior = channel_history.setdefault(channel, [])
        if str(row["author_id"]) == target_author_id and row.get("reply_to_message_id"):
            query_index = by_id.get(str(row["reply_to_message_id"]))
            if query_index is not None and query_index < index:
                query = history[query_index]
                if str(query["author_id"]) != target_author_id:
                    candidates = [
                        history[candidate_index]
                        for candidate_index in prior
                        if candidate_index < query_index
                        and str(history[candidate_index].get("content") or "").strip()
                    ][-candidate_pool:]
                    if len(candidates) >= minimum_candidates:
                        groups.append(
                            {
                                "occurred_at": query["occurred_at"],
                                "query_id": query["message_id"],
                                "query": query["content"],
                                "reply_id": row["message_id"],
                                "reply": row["content"],
                                "channel": channel,
                                "candidates": [
                                    {
                                        "event_id": item["message_id"],
                                        "actor": item["author_id"],
                                        "content": item["content"],
                                        "occurred_at": item["occurred_at"],
                                    }
                                    for item in candidates
                                ],
                            }
                        )
        prior.append(index)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(group, ensure_ascii=False) + "\n" for group in groups),
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "history_messages": len(history),
        "training_groups": len(groups),
        "candidate_pool": candidate_pool,
        "minimum_candidates": minimum_candidates,
        "output": str(destination.resolve()),
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build causal query-memory-reply groups for answer-aware reranking"
    )
    parser.add_argument("history_jsonl", nargs="+")
    parser.add_argument("--target-author-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-pool", type=int, default=128)
    parser.add_argument("--minimum-candidates", type=int, default=4)
    args = parser.parse_args()
    build_groups(
        args.history_jsonl,
        args.target_author_id,
        args.output,
        candidate_pool=args.candidate_pool,
        minimum_candidates=args.minimum_candidates,
    )


if __name__ == "__main__":
    main()
