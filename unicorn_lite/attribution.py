from __future__ import annotations

import json
import math
import re
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .encoder import Encoder
from .response_gate import answered_refractory_features, normalize_policy_text


def _time(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _clean(text: object) -> str:
    return re.sub(r"<@!?\d+>", " ", str(text)).strip()


def _load_histories(paths: list[str | Path]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                records[str(row["message_id"])] = row
    return sorted(
        records.values(),
        key=lambda row: (str(row["occurred_at"]), str(row["message_id"])),
    )


def _candidate_score(
    response_vector: np.ndarray,
    candidate_vector: np.ndarray,
    candidate: dict[str, Any],
    target_author_id: str,
    delay_seconds: float,
) -> float:
    semantic = float(np.dot(response_vector, candidate_vector))
    mentions_target = target_author_id in {
        str(value) for value in candidate.get("mentioned_user_ids", [])
    }
    question = "?" in str(candidate.get("content", ""))
    recency = math.exp(-max(0.0, delay_seconds) / 1800.0)
    return semantic + 0.14 * float(mentions_target) + 0.04 * float(question) + 0.12 * recency


def _directed_at_target(
    row: dict[str, Any],
    target_author_id: str,
    by_id: dict[str, int],
    history: list[dict[str, Any]],
) -> bool:
    if target_author_id in {
        str(value) for value in row.get("mentioned_user_ids", [])
    }:
        return True
    reply_to = row.get("reply_to_message_id")
    if not reply_to or str(reply_to) not in by_id:
        return False
    parent = history[by_id[str(reply_to)]]
    return str(parent.get("author_id")) == target_author_id


def _context_text(
    channel_history: deque[dict[str, Any]],
    current: dict[str, Any],
    target_author_id: str,
) -> str:
    lines: list[str] = []
    actor = str(current["author_id"])
    for previous in channel_history:
        previous_actor = str(previous["author_id"])
        if previous_actor == target_author_id:
            role = "AGENT"
        elif previous_actor == actor:
            role = "CURRENT_USER"
        elif bool(previous.get("author_is_bot")):
            role = "OTHER_BOT"
        else:
            role = "OTHER_USER"
        content = str(previous.get("content", "")).replace("\n", " ")[:500]
        if content:
            lines.append(f"[{role}] {content}")
    direct = target_author_id in {
        str(value) for value in current.get("mentioned_user_ids", [])
    }
    current_text = str(current.get("content", "")).replace("\n", " ")
    lines.append(f"[CURRENT direct={int(direct)}] {current_text}")
    return "\n".join(lines)


def build_contextual_policy_dataset(
    history_paths: list[str | Path],
    target_author_id: str,
    encoder: Encoder,
    output_path: str | Path,
    lookback_hours: float = 24.0,
    context_messages: int = 10,
) -> dict[str, object]:
    """Attribute responses conservatively and emit contextual policy examples."""
    history = _load_histories(history_paths)
    if not history:
        raise ValueError("no Discord history records were found")
    texts = [_clean(row.get("content", "")) or "<empty>" for row in history]
    vectors = encoder.encode_many(texts, batch_size=64)
    by_id = {str(row["message_id"]): index for index, row in enumerate(history)}
    by_channel: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(history):
        by_channel[str(row["channel_id"])].append(index)

    matched: dict[int, dict[str, Any]] = {}
    ambiguous: set[int] = set()
    unmatched_responses = 0
    max_seconds = lookback_hours * 3600.0

    for channel_indices in by_channel.values():
        channel_position = {index: position for position, index in enumerate(channel_indices)}
        for response_index in channel_indices:
            response = history[response_index]
            if str(response["author_id"]) != target_author_id:
                continue
            response_time = _time(response["occurred_at"])
            reply_to = response.get("reply_to_message_id")
            if reply_to and str(reply_to) in by_id:
                candidate_index = by_id[str(reply_to)]
                candidate = history[candidate_index]
                if not bool(candidate.get("author_is_bot")):
                    matched.setdefault(
                        candidate_index,
                        {
                            "response_index": response_index,
                            "source": "discord_reply",
                            "score": 1.0,
                            "margin": 1.0,
                        },
                    )
                    continue

            mentioned_ids = {
                str(value) for value in response.get("mentioned_user_ids", [])
            }
            candidate_indices: list[int] = []
            position = channel_position[response_index]
            for candidate_index in reversed(channel_indices[:position]):
                candidate = history[candidate_index]
                delay = (response_time - _time(candidate["occurred_at"])).total_seconds()
                if delay > max_seconds:
                    break
                if bool(candidate.get("author_is_bot")):
                    continue
                if mentioned_ids and str(candidate["author_id"]) not in mentioned_ids:
                    continue
                candidate_indices.append(candidate_index)

            if mentioned_ids and candidate_indices:
                scored: list[tuple[float, int]] = []
                for candidate_index in candidate_indices:
                    candidate = history[candidate_index]
                    delay = (response_time - _time(candidate["occurred_at"])).total_seconds()
                    score = _candidate_score(
                        vectors[response_index],
                        vectors[candidate_index],
                        candidate,
                        target_author_id,
                        delay,
                    )
                    scored.append((score, candidate_index))
                scored.sort(reverse=True)
                best_score, best_index = scored[0]
                second_score = scored[1][0] if len(scored) > 1 else -1.0
                margin = best_score - second_score
                if best_score >= 0.22 and margin >= 0.025:
                    matched.setdefault(
                        best_index,
                        {
                            "response_index": response_index,
                            "source": "semantic_mention",
                            "score": best_score,
                            "margin": margin,
                        },
                    )
                    for score, candidate_index in scored[1:]:
                        if best_score - score <= 0.06:
                            ambiguous.add(candidate_index)
                    continue
                for score, candidate_index in scored:
                    if best_score - score <= 0.06:
                        ambiguous.add(candidate_index)

            # A response that explicitly names a user must not be attached to a
            # different user's adjacent message merely because it was recent.
            if mentioned_ids:
                unmatched_responses += 1
                continue

            previous_position = channel_position[response_index] - 1
            if previous_position >= 0:
                previous_index = channel_indices[previous_position]
                previous = history[previous_index]
                delay = (response_time - _time(previous["occurred_at"])).total_seconds()
                if not bool(previous.get("author_is_bot")) and delay <= 300:
                    semantic = float(np.dot(vectors[response_index], vectors[previous_index]))
                    directed = _directed_at_target(
                        previous, target_author_id, by_id, history
                    )
                    question = "?" in str(previous.get("content", ""))
                    if directed or question or semantic >= 0.12:
                        matched.setdefault(
                            previous_index,
                            {
                                "response_index": response_index,
                                "source": "adjacent_turn",
                                "score": semantic + 0.12 * math.exp(-delay / 300.0),
                                "margin": 1.0,
                            },
                        )
                        continue
            unmatched_responses += 1

    # Count the final one-to-one assignments, not intermediate candidates that
    # may have competed for the same antecedent.
    response_sources: dict[str, int] = defaultdict(int)
    response_to_actor: dict[int, str] = {}
    response_to_candidate: dict[int, int] = {}
    for candidate_index, pairing in matched.items():
        response_sources[str(pairing["source"])] += 1
        response_index = int(pairing["response_index"])
        response_to_actor[response_index] = str(
            history[candidate_index]["author_id"]
        )
        response_to_candidate[response_index] = candidate_index

    channel_context: dict[str, deque[dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=context_messages)
    )
    last_agent_time: dict[str, datetime] = {}
    messages_since_agent: dict[str, int] = defaultdict(int)
    author_seen: dict[str, int] = defaultdict(int)
    author_replied: dict[str, int] = defaultdict(int)
    channel_humans_seen: dict[str, int] = defaultdict(int)
    channel_agent_responses: dict[str, int] = defaultdict(int)
    global_humans_seen = 0
    global_agent_responses = 0
    recent_channel_activity: dict[str, deque[int]] = defaultdict(
        lambda: deque(maxlen=50)
    )
    prior_memory_indices: list[int] = []
    answered_indices: set[int] = set()
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(history):
        channel = str(row["channel_id"])
        actor = str(row["author_id"])
        occurred = _time(row["occurred_at"])
        if actor == target_author_id:
            replied_actor = response_to_actor.get(index)
            if replied_actor is not None:
                author_replied[replied_actor] += 1
                channel_agent_responses[channel] += 1
                global_agent_responses += 1
            last_agent_time[channel] = occurred
            messages_since_agent[channel] = 0
            recent_channel_activity[channel].append(1)
            channel_context[channel].append(row)
            answered_candidate = response_to_candidate.get(index)
            if answered_candidate is not None:
                answered_indices.add(answered_candidate)
            prior_memory_indices.append(index)
            continue
        if bool(row.get("author_is_bot")):
            recent_channel_activity[channel].append(0)
            channel_context[channel].append(row)
            prior_memory_indices.append(index)
            continue

        pairing = matched.get(index)
        is_ambiguous = index in ambiguous and pairing is None
        direct = target_author_id in {
            str(value) for value in row.get("mentioned_user_ids", [])
        }
        prior_seen = author_seen[actor]
        prior_replied = author_replied[actor]
        seconds_since_agent = (
            (occurred - last_agent_time[channel]).total_seconds()
            if channel in last_agent_time
            else 86400.0
        )
        memory_matches: list[tuple[float, int]] = sorted(
            (
                (float(np.dot(vectors[index], vectors[prior_index])), prior_index)
                for prior_index in prior_memory_indices
            ),
            reverse=True,
        )[:5]
        memory_similarities = [similarity for similarity, _ in memory_matches]
        normalized_content = normalize_policy_text(
            str(row.get("content", ""))
        ).casefold()
        exact_matches = [
            prior_index
            for _, prior_index in memory_matches
            if normalize_policy_text(
                str(history[prior_index].get("content", ""))
            ).casefold()
            == normalized_content
        ]
        answered_matches = [
            (similarity, prior_index)
            for similarity, prior_index in memory_matches
            if prior_index in answered_indices
        ]
        refractory = answered_refractory_features(
            occurred,
            (
                (similarity, _time(history[prior_index]["occurred_at"]))
                for similarity, prior_index in answered_matches
            ),
        )
        example: dict[str, Any] = {
            "content": row.get("content", ""),
            "model_input": _context_text(
                channel_context[channel], row, target_author_id
            ),
            "action": (
                "REPLY_FLASH" if pairing else "UNKNOWN" if is_ambiguous else "OBSERVE"
            ),
            "occurred_at": row["occurred_at"],
            "actor": actor,
            "channel": channel,
            "discord_message_id": str(row["message_id"]),
            "direct": direct,
            "question": "?" in str(row.get("content", "")),
            "seconds_since_agent": max(0.0, seconds_since_agent),
            "messages_since_agent": messages_since_agent[channel],
            "author_messages_seen": prior_seen,
            "author_replies_seen": prior_replied,
            "author_reply_rate_past": prior_replied / max(1, prior_seen),
            "channel_human_messages_seen": channel_humans_seen[channel],
            "channel_agent_responses_seen": channel_agent_responses[channel],
            "channel_reply_rate_past": channel_agent_responses[channel]
            / max(1, channel_humans_seen[channel]),
            "global_reply_rate_past": global_agent_responses
            / max(1, global_humans_seen),
            "recent_agent_activity": sum(recent_channel_activity[channel])
            / max(1, len(recent_channel_activity[channel])),
            "max_memory_similarity": (
                memory_similarities[0] if memory_similarities else 0.0
            ),
            "mean_top3_memory_similarity": (
                sum(memory_similarities[:3]) / len(memory_similarities[:3])
                if memory_similarities
                else 0.0
            ),
            "near_duplicate_count": sum(
                similarity >= 0.90 for similarity in memory_similarities
            ),
            "exact_duplicate": bool(exact_matches),
            "max_answered_similarity": max(
                (similarity for similarity, _ in answered_matches), default=0.0
            ),
            "recent_answered_exact_duplicate": any(
                prior_index in answered_indices
                and 0.0
                <= (occurred - _time(history[prior_index]["occurred_at"])).total_seconds()
                <= 86400.0
                for prior_index in exact_matches
            ),
            **refractory,
            "label_source": pairing["source"] if pairing else "ambiguous" if is_ambiguous else "no_response",
        }
        if pairing:
            response = history[pairing["response_index"]]
            example.update(
                {
                    "expected_reply": response.get("content", ""),
                    "response_message_id": str(response["message_id"]),
                    "response_delay_seconds": (
                        _time(response["occurred_at"]) - occurred
                    ).total_seconds(),
                    "attribution_score": pairing["score"],
                    "attribution_margin": pairing["margin"],
                }
            )
        author_seen[actor] += 1
        channel_humans_seen[channel] += 1
        global_humans_seen += 1
        examples.append(example)
        messages_since_agent[channel] += 1
        recent_channel_activity[channel].append(0)
        channel_context[channel].append(row)
        prior_memory_indices.append(index)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples),
        encoding="utf-8",
    )
    counts: dict[str, int] = defaultdict(int)
    for row in examples:
        counts[str(row["action"])] += 1
    report: dict[str, object] = {
        "history_messages": len(history),
        "policy_examples": len(examples),
        "reply_examples": counts["REPLY_FLASH"],
        "observe_examples": counts["OBSERVE"],
        "unknown_examples": counts["UNKNOWN"],
        "target_responses": sum(
            str(row["author_id"]) == target_author_id for row in history
        ),
        "unmatched_target_responses": unmatched_responses,
        "attribution_sources": dict(response_sources),
        "output": str(destination.resolve()),
    }
    destination.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
