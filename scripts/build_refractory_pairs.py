from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


FOLLOW_UPS = (
    "What makes you say that?",
    "Can you explain why?",
    "How did you reach that conclusion?",
    "What would change your answer?",
)


def _time(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _without_mentions(content: str) -> str:
    return re.sub(r"<@!?\d+>", " ", content).strip()


def _direct(row: dict[str, Any]) -> bool:
    return bool(
        row.get("direct")
        or row.get("mentioned_target")
        or row.get("mentioned")
        or row.get("replied_to_target")
    )


def _base_row(source: dict[str, Any], pair_id: str, occurred_at: datetime) -> dict[str, Any]:
    content = str(source.get("content") or "")
    row = copy.deepcopy(source)
    row.update(
        {
            "occurred_at": occurred_at.isoformat(),
            "content": content,
            "model_input": f"[CURRENT direct=1] {content}",
            "action": "REPLY_FLASH",
            "direct": True,
            "refractory_pair_id": pair_id,
            "label_source": "training_only_refractory_seed",
            "recent_answered_similarity": 0.0,
            "recent_answered_near_duplicate": False,
            "seconds_since_answered_match": 86400.0,
            "recent_answered_exact_duplicate": False,
        }
    )
    return row


def _second_row(
    source: dict[str, Any],
    pair_id: str,
    occurred_at: datetime,
    content: str,
    action: str,
    similarity: float,
    near_duplicate: bool,
    exact_duplicate: bool,
    label_source: str,
) -> dict[str, Any]:
    seed = str(source.get("content") or "")
    row = copy.deepcopy(source)
    row.update(
        {
            "occurred_at": occurred_at.isoformat(),
            "content": content,
            "model_input": (
                f"[CURRENT_USER] {seed}\n"
                "[AGENT] <previous answer>\n"
                f"[CURRENT direct=1] {content}"
            ),
            "action": action,
            "direct": True,
            "question": True,
            "seconds_since_agent": 60.0,
            "messages_since_agent": 0,
            "recent_agent_activity": max(
                0.20, float(source.get("recent_agent_activity", 0.0))
            ),
            "max_memory_similarity": similarity,
            "mean_top3_memory_similarity": similarity * 0.70,
            "near_duplicate_count": 1 if near_duplicate else 0,
            "exact_duplicate": exact_duplicate,
            "max_answered_similarity": similarity,
            "recent_answered_exact_duplicate": exact_duplicate,
            "recent_answered_similarity": similarity,
            "recent_answered_near_duplicate": near_duplicate,
            "seconds_since_answered_match": 60.0,
            "refractory_pair_id": pair_id,
            "label_source": label_source,
        }
    )
    return row


def build_pairs(
    policy_path: str | Path,
    output_path: str | Path,
    cutoff: str,
    pair_count: int = 32,
) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in Path(policy_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        content = str(row.get("content") or "")
        cleaned = _without_mentions(content)
        words = re.findall(r"[A-Za-z0-9']+", cleaned)
        if (
            row.get("action") != "REPLY_FLASH"
            or not _direct(row)
            or str(row.get("occurred_at") or "") > cutoff
            or not 5 <= len(words) <= 30
            or "?" not in cleaned
        ):
            continue
        identity = cleaned.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(row)
    candidates.sort(key=lambda row: str(row.get("occurred_at") or ""))
    if pair_count < 1:
        selected: list[dict[str, Any]] = []
    elif pair_count < len(candidates):
        indices = np.linspace(0, len(candidates) - 1, pair_count, dtype=np.int64)
        selected = [candidates[int(index)] for index in indices]
    else:
        selected = candidates

    output_rows: list[dict[str, Any]] = []
    for index, source in enumerate(selected):
        occurred = _time(source["occurred_at"])
        cleaned = _without_mentions(str(source.get("content") or ""))

        repeat_pair = f"repeat-{index:03d}"
        output_rows.append(_base_row(source, repeat_pair, occurred))
        repeated = (
            cleaned
            if index % 2 == 0
            else f"Can you answer this again: {cleaned}"
        )
        output_rows.append(
            _second_row(
                source,
                repeat_pair,
                occurred + timedelta(seconds=1),
                repeated,
                "OBSERVE",
                0.97 if index % 2 == 0 else 0.93,
                True,
                index % 2 == 0,
                "training_only_refractory_repeat",
            )
        )

        followup_pair = f"followup-{index:03d}"
        output_rows.append(_base_row(source, followup_pair, occurred + timedelta(seconds=2)))
        output_rows.append(
            _second_row(
                source,
                followup_pair,
                occurred + timedelta(seconds=3),
                FOLLOW_UPS[index % len(FOLLOW_UPS)],
                "REPLY_FLASH",
                0.62,
                False,
                False,
                "training_only_refractory_followup",
            )
        )

    output_rows.sort(key=lambda row: str(row.get("occurred_at") or ""))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    report = {
        "source": str(Path(policy_path).resolve()),
        "output": str(destination.resolve()),
        "cutoff": cutoff,
        "candidates": len(candidates),
        "selected_source_questions": len(selected),
        "training_rows": len(output_rows),
        "repeat_negative_pairs": len(selected),
        "followup_positive_pairs": len(selected),
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paired near-repeat negatives and legitimate follow-up controls"
    )
    parser.add_argument("policy_jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--pairs", type=int, default=32)
    args = parser.parse_args()
    build_pairs(args.policy_jsonl, args.output, args.cutoff, args.pairs)


if __name__ == "__main__":
    main()
