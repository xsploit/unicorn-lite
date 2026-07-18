from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


API_ROOT = "https://discord.com/api/v10"
TEXT_CHANNEL_TYPES = {0, 5, 10, 11, 12}


@dataclass(slots=True)
class BackfillReport:
    bot_id: str
    guilds: int
    channels_scanned: int
    channels_skipped: int
    messages_fetched: int
    unique_messages: int
    output: str


def _token(token: str | None = None) -> str:
    value = token or os.getenv("DISCORD_BOT_TOKEN")
    if not value:
        raise RuntimeError("DISCORD_BOT_TOKEN is not set")
    return value


def _get(
    client: httpx.Client,
    path: str,
    params: dict[str, object] | None = None,
    allow_forbidden: bool = False,
) -> Any | None:
    for _ in range(6):
        response = client.get(path, params=params)
        if response.status_code == 429:
            retry_after = float(response.json().get("retry_after", 1.0))
            time.sleep(max(0.1, min(retry_after, 30.0)))
            continue
        if response.status_code >= 500:
            time.sleep(1.0)
            continue
        if allow_forbidden and response.status_code in {403, 404}:
            return None
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Discord rate limit did not clear for {path}")


def _message_record(
    message: dict[str, Any],
    bot_id: str,
    guild: dict[str, Any],
    channel: dict[str, Any],
) -> dict[str, object]:
    author = message.get("author", {})
    member = message.get("member") or {}
    reference = message.get("message_reference") or {}
    display_name = (
        member.get("nick")
        or author.get("global_name")
        or author.get("username")
        or "unknown"
    )
    author_id = str(author.get("id", ""))
    mentions = message.get("mentions") or []
    return {
        "message_id": str(message["id"]),
        "occurred_at": str(message["timestamp"]),
        "guild_id": str(guild["id"]),
        "guild_name": guild.get("name"),
        "channel_id": str(channel["id"]),
        "channel_name": channel.get("name"),
        "author_id": author_id,
        "author_name": author.get("username", "unknown"),
        "display_name": display_name,
        "author_is_bot": bool(author.get("bot", False)),
        "author_is_agent": author_id == bot_id,
        "content": str(message.get("content", "")),
        "reply_to_message_id": (
            str(reference["message_id"]) if reference.get("message_id") else None
        ),
        "mentioned_user_ids": [str(user.get("id")) for user in mentions],
        "mentions_agent": any(str(user.get("id")) == bot_id for user in mentions),
        "attachments": [
            {
                "id": str(item.get("id", "")),
                "filename": item.get("filename"),
                "content_type": item.get("content_type"),
                "size": item.get("size"),
                "url": item.get("url"),
            }
            for item in message.get("attachments", [])
        ],
        "embed_count": len(message.get("embeds", [])),
        "reaction_count": sum(
            int(reaction.get("count", 0)) for reaction in message.get("reactions", [])
        ),
    }


def _read_existing(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records[str(row["message_id"])] = row
    return records


def _write_records(path: Path, records: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        records.values(), key=lambda item: (str(item["occurred_at"]), str(item["message_id"]))
    )
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _client(token: str | None = None) -> httpx.Client:
    return httpx.Client(
        base_url=API_ROOT,
        headers={"Authorization": f"Bot {_token(token)}"},
        timeout=30,
    )


def discord_inventory(
    token: str | None = None, output_path: str | Path | None = None
) -> None:
    with _client(token) as client:
        bot = _get(client, "/users/@me")
        guilds = _get(client, "/users/@me/guilds")
        inventory: dict[str, object] = {
            "bot": bot.get("username"),
            "bot_id": str(bot["id"]),
            "guilds": [],
        }
        guild_rows: list[dict[str, object]] = []
        for guild in guilds:
            channels = _get(client, f"/guilds/{guild['id']}/channels")
            guild_rows.append(
                {
                    "id": str(guild["id"]),
                    "name": guild["name"],
                    "history_candidates": [
                        {
                            "id": str(channel["id"]),
                            "name": channel.get("name"),
                            "type": int(channel["type"]),
                            "parent_id": channel.get("parent_id"),
                        }
                        for channel in channels
                        if int(channel["type"]) in TEXT_CHANNEL_TYPES
                    ],
                }
            )
        inventory["guilds"] = guild_rows
        candidate_count = sum(
            len(row["history_candidates"]) for row in guild_rows
        )
        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        print(
            json.dumps(
                {
                    "bot": inventory["bot"],
                    "bot_id": inventory["bot_id"],
                    "guilds": len(guild_rows),
                    "history_candidates": candidate_count,
                    "output": str(Path(output_path).resolve()) if output_path else None,
                },
                indent=2,
            )
        )


def _channel_messages(
    client: httpx.Client,
    channel_id: str,
    limit: int | None,
) -> list[dict[str, Any]] | None:
    messages: list[dict[str, Any]] = []
    before: str | None = None
    while limit is None or len(messages) < limit:
        page_size = 100 if limit is None else min(100, limit - len(messages))
        params: dict[str, object] = {"limit": page_size}
        if before:
            params["before"] = before
        page = _get(
            client,
            f"/channels/{channel_id}/messages",
            params=params,
            allow_forbidden=True,
        )
        if page is None:
            return None
        if not page:
            break
        messages.extend(page)
        before = str(page[-1]["id"])
        if len(page) < page_size:
            break
    return messages


def backfill_discord(
    output_path: str | Path,
    token: str | None = None,
    guild_id: str | None = None,
    channel_ids: set[str] | None = None,
    limit_per_channel: int | None = 2000,
) -> None:
    output = Path(output_path)
    existing = _read_existing(output)
    scanned = 0
    skipped = 0
    fetched = 0
    with _client(token) as client:
        bot = _get(client, "/users/@me")
        bot_id = str(bot["id"])
        guilds = _get(client, "/users/@me/guilds")
        selected_guilds = [
            guild for guild in guilds if guild_id is None or str(guild["id"]) == guild_id
        ]
        for guild in selected_guilds:
            channels = _get(client, f"/guilds/{guild['id']}/channels")
            for channel in channels:
                channel_id = str(channel["id"])
                if int(channel["type"]) not in TEXT_CHANNEL_TYPES:
                    continue
                if channel_ids and channel_id not in channel_ids:
                    continue
                try:
                    page = _channel_messages(client, channel_id, limit_per_channel)
                except (httpx.HTTPError, RuntimeError) as exc:
                    skipped += 1
                    print(
                        f"skipped channel={channel_id} name={channel.get('name')}: "
                        f"{type(exc).__name__}"
                    )
                    _write_records(output, existing)
                    continue
                if page is None:
                    skipped += 1
                    _write_records(output, existing)
                    continue
                scanned += 1
                for message in page:
                    existing[str(message["id"])] = _message_record(
                        message, bot_id, guild, channel
                    )
                    fetched += 1
                _write_records(output, existing)
        _write_records(output, existing)
        report = BackfillReport(
            bot_id=bot_id,
            guilds=len(selected_guilds),
            channels_scanned=scanned,
            channels_skipped=skipped,
            messages_fetched=fetched,
            unique_messages=len(existing),
            output=str(output.resolve()),
        )
        print(json.dumps(asdict(report), indent=2))


def _timestamp(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def build_discord_labels(
    history_path: str | Path,
    target_author_id: str,
    output_path: str | Path,
    mention_lookback_hours: float = 24.0,
) -> dict[str, object]:
    """Create conservative policy labels from explicit target replies and mentions."""
    history = sorted(
        _read_existing(Path(history_path)).values(),
        key=lambda item: (str(item["occurred_at"]), str(item["message_id"])),
    )
    by_id = {str(row["message_id"]): row for row in history}
    paired: dict[str, tuple[dict[str, object], str]] = {}
    for index, row in enumerate(history):
        if str(row["author_id"]) != target_author_id:
            continue
        reply_to = row.get("reply_to_message_id")
        if reply_to and reply_to in by_id:
            parent = by_id[reply_to]
            if str(parent["author_id"]) != target_author_id:
                paired[str(reply_to)] = (row, "explicit_reply")
                continue
        mentioned_ids = {str(value) for value in row.get("mentioned_user_ids", [])}
        if not mentioned_ids:
            continue
        response_time = _timestamp(row["occurred_at"])
        for previous in reversed(history[:index]):
            delay_hours = (
                response_time - _timestamp(previous["occurred_at"])
            ).total_seconds() / 3600.0
            if delay_hours > mention_lookback_hours:
                break
            author_id = str(previous["author_id"])
            if author_id in mentioned_ids and author_id != target_author_id:
                paired.setdefault(
                    str(previous["message_id"]), (row, "explicit_author_mention")
                )
                mentioned_ids.remove(author_id)
                if not mentioned_ids:
                    break

    rows: list[dict[str, object]] = []
    positives = 0
    direct_unanswered = 0
    for row in history:
        if bool(row.get("author_is_bot")) or str(row["author_id"]) == target_author_id:
            continue
        message_id = str(row["message_id"])
        response_pair = paired.get(message_id)
        mentioned_target = target_author_id in {
            str(value) for value in row.get("mentioned_user_ids", [])
        }
        reply_parent = by_id.get(str(row.get("reply_to_message_id")))
        replied_to_target = bool(
            reply_parent and str(reply_parent["author_id"]) == target_author_id
        )
        direct = mentioned_target or replied_to_target
        output: dict[str, object] = {
            "content": row["content"],
            "action": "REPLY_FLASH" if response_pair else "OBSERVE",
            "occurred_at": row["occurred_at"],
            "actor": row["author_id"],
            "channel": row["channel_id"],
            "discord_message_id": message_id,
            "direct": direct,
            "mentioned_target": mentioned_target,
            "replied_to_target": replied_to_target,
            "label_source": response_pair[1] if response_pair else "no_explicit_response",
        }
        if response_pair:
            response, _ = response_pair
            output["expected_reply"] = response["content"]
            output["response_message_id"] = response["message_id"]
            output["response_delay_seconds"] = (
                _timestamp(response["occurred_at"]) - _timestamp(row["occurred_at"])
            ).total_seconds()
            positives += 1
        elif direct:
            direct_unanswered += 1
        rows.append(output)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "history_messages": len(history),
        "human_policy_examples": len(rows),
        "explicit_response_examples": positives,
        "observe_examples": len(rows) - positives,
        "direct_but_unanswered": direct_unanswered,
        "target_author_id": target_author_id,
        "output": str(destination.resolve()),
    }
    print(json.dumps(report, indent=2))
    return report
