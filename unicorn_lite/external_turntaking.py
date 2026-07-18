from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _write(rows: Iterable[dict[str, Any]], output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"REPLY_FLASH": 0, "OBSERVE": 0}
    total = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[str(row["action"])] += 1
            total += 1
    return {
        "examples": total,
        "reply_examples": counts["REPLY_FLASH"],
        "observe_examples": counts["OBSERVE"],
        "reply_rate": counts["REPLY_FLASH"] / max(1, total),
        "output": str(output.resolve()),
    }


def _base_row(
    *, content: str, model_input: str, action: str, channel: str, direct: bool
) -> dict[str, Any]:
    return {
        "content": content,
        "model_input": model_input,
        "action": action,
        "channel": channel,
        "direct": direct,
        "mentioned_target": direct,
        "replied_to_target": False,
        "question": "?" in content,
        "seconds_since_agent": 86400.0,
        "messages_since_agent": 0,
        "author_messages_seen": 0,
        "author_reply_rate_past": 0.0,
        "channel_human_messages_seen": 0,
        "channel_reply_rate_past": 0.0,
        "global_reply_rate_past": 0.0,
        "recent_agent_activity": 0.0,
        "max_memory_similarity": 0.0,
        "mean_top3_memory_similarity": 0.0,
        "near_duplicate_count": 0,
        "exact_duplicate": False,
        "max_answered_similarity": 0.0,
        "recent_answered_exact_duplicate": False,
        "recent_answered_similarity": 0.0,
        "recent_answered_near_duplicate": False,
        "seconds_since_answered_match": 86400.0,
    }


def convert_ishiki(
    input_paths: list[str | Path], output_path: str | Path
) -> dict[str, Any]:
    """Convert Speak-or-Stay-Silent records to the local policy schema."""

    def rows() -> Iterable[dict[str, Any]]:
        for input_path in input_paths:
            source = Path(input_path)
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                target = str(item["target_speaker"])
                context = list(item.get("context_turns", []))
                current = dict(item["current_turn"])
                transcript = [
                    f"[SPEAKER {turn['speaker']}] {turn['text']}" for turn in context
                ]
                transcript.append(
                    f"[CURRENT SPEAKER {current['speaker']}] {current['text']}"
                )
                row = _base_row(
                    content=str(current["text"]),
                    model_input=(
                        f"[TARGET SPEAKER {target}]\n" + "\n".join(transcript)
                    ),
                    action=(
                        "REPLY_FLASH" if item["decision"] == "SPEAK" else "OBSERVE"
                    ),
                    channel=f"ishiki:{item['sequence_id']}:{target}",
                    direct=bool(item.get("target_is_addressed")),
                )
                row.update(
                    {
                        "external_source": "ishiki-labs/multi-party-dialogue",
                        "external_category": item.get("category"),
                        "external_id": item.get("decision_point_id"),
                    }
                )
                yield row

    return _write(rows(), output_path)


def convert_when2speak(
    input_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Convert When2Speak token records to the local policy schema."""

    def rows() -> Iterable[dict[str, Any]]:
        source = Path(input_path)
        for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            messages = list(item["messages"])
            if len(messages) < 2:
                continue
            label = str(messages[-1]["content"]).strip()
            turns = [str(message["content"]) for message in messages[:-1]]
            content = turns[-1]
            direct = "[agent]" in content.casefold()
            digest = hashlib.sha1(
                "\n".join(turns).encode("utf-8")
            ).hexdigest()[:16]
            row = _base_row(
                content=content,
                model_input="\n".join(turns),
                action="REPLY_FLASH" if label == "<" else "OBSERVE",
                channel=f"when2speak:{digest}",
                direct=direct,
            )
            row.update(
                {
                    "external_source": "duke-trust-lab/When2Speak",
                    "external_id": index,
                }
            )
            yield row

    return _write(rows(), output_path)
