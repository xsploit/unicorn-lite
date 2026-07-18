from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

from .store import MemoryStore


def _seconds_since(value: str | None, now: str) -> float:
    if not value:
        return 86400.0
    previous = datetime.fromisoformat(value.replace("Z", "+00:00"))
    current = datetime.fromisoformat(now.replace("Z", "+00:00"))
    return max(0.0, (current - previous).total_seconds())


class DiscordContextTracker:
    """Causal, restart-safe structural context for the learned Discord policy."""

    def __init__(self, store: MemoryStore, context_messages: int = 10) -> None:
        self.store = store
        self.context_messages = context_messages
        saved = store.load_runtime_state("discord_context") or {}
        self.global_humans_seen = int(saved.get("global_humans_seen", 0))
        self.global_agent_responses = int(saved.get("global_agent_responses", 0))
        self.authors: dict[str, dict[str, int]] = dict(saved.get("authors", {}))
        self.channels: dict[str, dict[str, Any]] = dict(saved.get("channels", {}))

    def _channel(self, channel: str) -> dict[str, Any]:
        return self.channels.setdefault(
            channel,
            {
                "last_agent_at": None,
                "messages_since_agent": 0,
                "human_seen": 0,
                "agent_responses": 0,
                "recent_activity": [],
                "recent_messages": [],
            },
        )

    def metadata(
        self,
        *,
        channel: str,
        actor: str,
        content: str,
        occurred_at: str,
        direct: bool,
        mentioned: bool,
        lexically_addressed: bool = False,
        replied_to_target: bool,
        display_name: str | None = None,
    ) -> dict[str, object]:
        state = self._channel(channel)
        author = self.authors.get(actor, {"seen": 0, "replied": 0})
        effective_direct = bool(
            direct or mentioned or lexically_addressed or replied_to_target
        )
        lines: list[str] = []
        writer_lines: list[str] = []
        for previous in state["recent_messages"][-self.context_messages :]:
            if previous["is_agent"]:
                role = "AGENT"
            elif previous["actor"] == actor:
                role = "CURRENT_USER"
            elif previous["is_bot"]:
                role = "OTHER_BOT"
            else:
                role = "OTHER_USER"
            text = str(previous["content"]).replace("\n", " ")[:500]
            if text:
                lines.append(f"[{role}] {text}")
                speaker = str(previous.get("display_name") or previous["actor"])
                writer_lines.append(f"[{role}: {speaker}] {text}")
        lines.append(
            f"[CURRENT direct={int(effective_direct)}] "
            f"{content.replace(chr(10), ' ')}"
        )
        writer_lines.append(
            f"[CURRENT_USER: {display_name or actor}] "
            f"{content.replace(chr(10), ' ')[:1000]}"
        )
        activity = state["recent_activity"][-50:]
        return {
            "policy_input": "\n".join(lines),
            "conversation_context": "\n".join(writer_lines),
            "direct": effective_direct,
            "mentioned": mentioned,
            "mentioned_target": mentioned or lexically_addressed,
            "lexically_addressed": lexically_addressed,
            "replied_to_target": replied_to_target,
            "question": "?" in content,
            "seconds_since_agent": _seconds_since(
                state.get("last_agent_at"), occurred_at
            ),
            "messages_since_agent": int(state["messages_since_agent"]),
            "author_messages_seen": int(author.get("seen", 0)),
            "author_replies_seen": int(author.get("replied", 0)),
            "author_reply_rate_past": int(author.get("replied", 0))
            / max(1, int(author.get("seen", 0))),
            "channel_human_messages_seen": int(state["human_seen"]),
            "channel_agent_responses_seen": int(state["agent_responses"]),
            "channel_reply_rate_past": int(state["agent_responses"])
            / max(1, int(state["human_seen"])),
            "global_reply_rate_past": self.global_agent_responses
            / max(1, self.global_humans_seen),
            "recent_agent_activity": sum(activity) / max(1, len(activity)),
        }

    def observe(
        self,
        *,
        channel: str,
        actor: str,
        content: str,
        occurred_at: str | None = None,
        is_bot: bool,
        is_agent: bool,
        message_id: str,
        reply_to_message_id: str | None = None,
        display_name: str | None = None,
    ) -> None:
        occurred_at = occurred_at or datetime.now(timezone.utc).isoformat()
        state = self._channel(channel)
        recent = state["recent_messages"]
        if is_agent:
            replied_actor = next(
                (
                    item["actor"]
                    for item in reversed(recent)
                    if reply_to_message_id and item["message_id"] == reply_to_message_id
                ),
                None,
            )
            if replied_actor is not None:
                author = self.authors.setdefault(
                    str(replied_actor), {"seen": 0, "replied": 0}
                )
                author["replied"] = int(author.get("replied", 0)) + 1
                state["agent_responses"] = int(state["agent_responses"]) + 1
                self.global_agent_responses += 1
            state["last_agent_at"] = occurred_at
            state["messages_since_agent"] = 0
            state["recent_activity"].append(1)
        else:
            state["recent_activity"].append(0)
            if not is_bot:
                author = self.authors.setdefault(actor, {"seen": 0, "replied": 0})
                author["seen"] = int(author.get("seen", 0)) + 1
                state["human_seen"] = int(state["human_seen"]) + 1
                self.global_humans_seen += 1
                state["messages_since_agent"] = int(state["messages_since_agent"]) + 1
        state["recent_activity"] = state["recent_activity"][-50:]
        recent.append(
            {
                "message_id": message_id,
                "actor": actor,
                "content": content,
                "display_name": display_name or actor,
                "is_bot": is_bot,
                "is_agent": is_agent,
            }
        )
        state["recent_messages"] = recent[-self.context_messages :]
        self.store.save_runtime_state(
            "discord_context",
            {
                "global_humans_seen": self.global_humans_seen,
                "global_agent_responses": self.global_agent_responses,
                "authors": self.authors,
                "channels": self.channels,
            },
        )
