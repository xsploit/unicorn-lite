"""Create training-only direct-mention variants of real silent emoji turns."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np


def is_direct(row: dict[str, object]) -> bool:
    return bool(
        row.get("direct")
        or row.get("mentioned_target")
        or row.get("mentioned")
        or row.get("replied_to_target")
    )


def emoji_or_symbol_only(text: str) -> bool:
    if not text.strip() or "http://" in text or "https://" in text:
        return False
    if re.search(r"<@!?\d+>", text):
        return False
    without_custom = re.sub(r"<a?:\w+:\d+>|:[^:\s]+:", " ", text)
    if re.search(r"[A-Za-z0-9]", without_custom):
        return False
    return bool(
        re.search(r"<a?:\w+:\d+>|:[^:\s]+:", text)
        or any(unicodedata.category(character) == "So" for character in text)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.source).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("action") in {"OBSERVE", "REPLY_FLASH"}]
    train_end = int(len(rows) * 0.70)
    train_rows = rows[:train_end]
    candidates = [
        row
        for row in train_rows
        if row.get("action") == "OBSERVE"
        and not is_direct(row)
        and emoji_or_symbol_only(str(row.get("content") or ""))
    ]
    if 0 < args.count < len(candidates):
        indices = np.linspace(0, len(candidates) - 1, args.count, dtype=np.int64)
        candidates = [candidates[int(index)] for index in indices]

    output_rows: list[dict[str, object]] = []
    for index, source in enumerate(candidates):
        content = str(source.get("content") or "")
        candidate = dict(source)
        candidate["event_id"] = f"counterfactual-emoji-{index}"
        candidate["discord_message_id"] = f"counterfactual-emoji-{index}"
        candidate["content"] = f"<@0> {content}"
        candidate["direct"] = True
        candidate["mentioned"] = True
        candidate["mentioned_target"] = True
        candidate["replied_to_target"] = False
        candidate["action"] = "OBSERVE"
        candidate["label_source"] = "counterfactual_direct_emoji_silence"
        model_input = str(source.get("model_input") or source.get("content") or "")
        lines = model_input.splitlines()
        if lines and lines[-1].startswith("[CURRENT"):
            lines[-1] = f"[CURRENT direct=1] <@0> {content}"
            candidate["model_input"] = "\n".join(lines)
        else:
            candidate["model_input"] = f"[CURRENT direct=1] <@0> {content}"
        output_rows.append(candidate)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_rows": len(rows),
                "training_rows": len(train_rows),
                "eligible_emoji_silences": len(
                    [
                        row
                        for row in train_rows
                        if row.get("action") == "OBSERVE"
                        and not is_direct(row)
                        and emoji_or_symbol_only(str(row.get("content") or ""))
                    ]
                ),
                "selected": len(output_rows),
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
