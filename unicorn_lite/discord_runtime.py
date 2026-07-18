from __future__ import annotations

import asyncio
import io
import json
import os
import re
from dataclasses import asdict
from typing import Any

from .agent import UnicornAgent
from .discord_context import DiscordContextTracker
from .models import Experience


def is_lexically_addressed(text: str, aliases: set[str]) -> bool:
    """Detect a name used as an opening vocative without forcing a reply."""
    normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    if not normalized:
        return False
    words = normalized.split()
    prefixes = {"hey", "hi", "hello", "yo", "ok", "okay", "well"}
    start = 1 if words and words[0] in prefixes else 0
    for alias in aliases:
        normalized_alias = re.sub(r"[^a-z0-9]+", " ", alias.casefold()).strip()
        if not normalized_alias:
            continue
        alias_words = normalized_alias.split()
        if words[start : start + len(alias_words)] == alias_words:
            return True
        if words[start : start + 1] == [normalized_alias.replace(" ", "")]:
            return True
    return False


def decision_audit_payload(
    agent: UnicornAgent,
    event: Experience,
    result: Any,
    *,
    writer_allowed: bool,
    discord_reply_sent: bool,
    discord_send_error: str | None,
    jump_url: str,
) -> dict[str, Any]:
    replied = result.decision.action.startswith("REPLY")
    reply_probability = (
        result.decision.confidence if replied else 1.0 - result.decision.confidence
    )
    return {
        "event_id": event.event_id,
        "discord_message_id": event.metadata.get("discord_message_id"),
        "jump_url": jump_url,
        "occurred_at": event.occurred_at,
        "actor": event.actor,
        "channel": event.channel,
        "content": event.content,
        "decision": {
            **asdict(result.decision),
            "reply_probability": reply_probability,
            "threshold": (
                agent.policy.threshold_for(event.channel)
                if hasattr(agent.policy, "threshold_for")
                else getattr(agent.policy, "threshold", None)
            ),
            "passed_gate": replied,
        },
        "runtime": {
            "audit_schema_version": 2,
            "writer_provider": (
                type(agent.writer).__name__ if agent.writer is not None else None
            ),
            "writer_model": result.decision.model,
            "writer_allowed": writer_allowed,
            "writer_attempted": bool(
                writer_allowed and replied and agent.writer is not None
            ),
            "writer_produced_reply": result.reply is not None,
            "writer_error": result.writer_error,
            "discord_reply_sent": discord_reply_sent,
            "discord_send_error": discord_send_error,
            "memory_selector": (
                "answer-aware-v1"
                if agent.memory_reranker is not None
                else "cosine-top5"
            ),
            "memory_selector_applied": bool(
                event.metadata.get("memory_reranker", {}).get("applied")
            ),
            "memory_candidate_limit": agent.memory_candidate_limit,
        },
        "neural": {"surprise": result.surprise},
        "event_metadata": event.metadata,
        "retrieved_memories": [
            {
                "event_id": memory.event_id,
                "content": memory.content,
                "actor": memory.actor,
                "similarity": memory.similarity,
                "occurred_at": memory.occurred_at,
                "decision_action": memory.decision_action,
            }
            for memory in result.memories
        ],
        "facts_updated": result.facts_updated,
    }


def run_discord(
    agent: UnicornAgent,
    token: str | None = None,
    mode: str = "shadow",
    run_seconds: float | None = None,
    allow_unsolicited: bool = False,
    unsolicited_channel_ids: set[str] | None = None,
    decision_audit_channel_id: str | None = None,
    decision_audit_source_channel_ids: set[str] | None = None,
) -> None:
    import discord

    bot_token = token or os.getenv("DISCORD_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("DISCORD_BOT_TOKEN is not set")
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    tracker = DiscordContextTracker(agent.store)

    @client.event
    async def on_ready() -> None:
        print(f"connected as {client.user}; mode={mode}")
        if mode == "active":
            async def warm_writer() -> None:
                try:
                    if await agent.warm_writer():
                        print("local writer warmed and resident")
                except Exception as exc:
                    print(f"writer warmup failed: {exc}")

            asyncio.create_task(warm_writer())
        if run_seconds is not None:
            async def close_later() -> None:
                await asyncio.sleep(max(0.1, run_seconds))
                await client.close()

            asyncio.create_task(close_later())

    @client.event
    async def on_message(message: discord.Message) -> None:
        if client.user is None:
            return
        if (
            decision_audit_channel_id
            and str(message.channel.id) == decision_audit_channel_id
            and message.author.id == client.user.id
            and any(
                attachment.filename.startswith("decision-")
                for attachment in message.attachments
            )
        ):
            return
        is_agent = message.author.id == client.user.id
        is_bot = bool(message.author.bot)
        direct = isinstance(message.channel, discord.DMChannel)
        mentioned = client.user in message.mentions
        aliases = {client.user.name, "neuro", "neuro-sama", "neurosama"}
        if message.guild is not None and message.guild.me is not None:
            aliases.add(message.guild.me.display_name)
        aliases.update(
            value.strip()
            for value in os.getenv("UNICORN_AGENT_ALIASES", "").split(",")
            if value.strip()
        )
        lexically_addressed = is_lexically_addressed(message.content, aliases)
        reply_to_message_id = (
            str(message.reference.message_id)
            if message.reference and message.reference.message_id
            else None
        )
        replied_to_target = False
        if message.reference and message.reference.resolved:
            parent = message.reference.resolved
            replied_to_target = getattr(parent, "author", None) == client.user
        occurred_at = message.created_at.isoformat()
        policy_metadata = tracker.metadata(
            channel=str(message.channel.id),
            actor=str(message.author.id),
            content=message.content,
            occurred_at=occurred_at,
            direct=direct,
            mentioned=mentioned,
            lexically_addressed=lexically_addressed,
            replied_to_target=replied_to_target,
            display_name=message.author.display_name,
        )
        event = Experience(
            content=message.content,
            actor=str(message.author.id),
            source="discord",
            channel=str(message.channel.id),
            occurred_at=occurred_at,
            metadata={
                **policy_metadata,
                "author_is_bot": is_bot,
                "author_is_agent": is_agent,
                "display_name": message.author.display_name,
                "discord_message_id": str(message.id),
                "reply_to_message_id": reply_to_message_id,
            },
        )
        writer_allowed = mode == "active" and not is_bot and (
            direct
            or mentioned
            or (
                allow_unsolicited
                and (
                    not unsolicited_channel_ids
                    or event.channel in unsolicited_channel_ids
                )
            )
        )
        result = await agent.ingest(event, allow_writer=writer_allowed)
        tracker.observe(
            channel=event.channel,
            actor=event.actor,
            content=event.content,
            occurred_at=event.occurred_at,
            is_bot=is_bot,
            is_agent=is_agent,
            message_id=str(message.id),
            reply_to_message_id=reply_to_message_id,
            display_name=message.author.display_name,
        )
        print(
            f"event={event.event_id} actor={event.actor} "
            f"action={result.decision.action} confidence={result.decision.confidence:.3f} "
            f"surprise={result.surprise:.3f}"
        )
        discord_reply_sent = False
        discord_send_error = None
        if not is_agent and writer_allowed and result.reply:
            try:
                await message.reply(result.reply, mention_author=False)
                discord_reply_sent = True
            except Exception as exc:
                discord_send_error = f"{type(exc).__name__}: {exc}"
                print(f"discord reply failed event={event.event_id}: {discord_send_error}")

        should_audit = (
            decision_audit_channel_id
            and not is_bot
            and (
                not decision_audit_source_channel_ids
                or event.channel in decision_audit_source_channel_ids
            )
        )
        if should_audit:
            try:
                audit_channel = client.get_channel(int(decision_audit_channel_id))
                if audit_channel is None:
                    audit_channel = await client.fetch_channel(
                        int(decision_audit_channel_id)
                    )
                payload = decision_audit_payload(
                    agent,
                    event,
                    result,
                    writer_allowed=writer_allowed,
                    discord_reply_sent=discord_reply_sent,
                    discord_send_error=discord_send_error,
                    jump_url=message.jump_url,
                )
                probability = payload["decision"]["reply_probability"]
                threshold = payload["decision"]["threshold"]
                margin = probability - threshold
                metadata = payload["event_metadata"]
                did_reply = payload["decision"]["passed_gate"]
                embed = discord.Embed(
                    title=(
                        "🟢 Controller selected REPLY"
                        if did_reply
                        else "🔴 Controller selected SILENCE"
                    ),
                    url=message.jump_url,
                    description=(
                        f"> {message.content[:700] or '*No text content*'}\n\n"
                        f"**Reason:** {result.decision.reason}"
                    ),
                    color=(0x2ECC71 if did_reply else 0xE74C3C),
                    timestamp=message.created_at,
                )
                embed.add_field(
                    name="Reply score",
                    value=(
                        f"**{probability * 100:.1f}%**\n"
                        f"Threshold: {threshold * 100:.1f}%\n"
                        f"Margin: {margin * 100:+.1f} points"
                    ),
                    inline=True,
                )
                embed.add_field(
                    name="Neural state",
                    value=(
                        f"Confidence: {result.decision.confidence * 100:.1f}%\n"
                        f"Surprise: {result.surprise:.3f}\n"
                        f"Compute tier: {result.decision.compute_tier}"
                    ),
                    inline=True,
                )
                ponder = metadata.get("ponder")
                if isinstance(ponder, dict):
                    path = " → ".join(
                        f"{float(value) * 100:.1f}%"
                        for value in ponder.get("probability_path", [])
                    )
                    embed.add_field(
                        name="Recursive controller",
                        value=(
                            f"Steps: **{int(ponder.get('steps', 0))}/"
                            f"{int(ponder.get('max_steps', 0))}**\n"
                            f"Halt: **{float(ponder.get('halt_probability', 0)) * 100:.1f}%** "
                            f"(needs {float(ponder.get('halt_threshold', 0)) * 100:.1f}%)\n"
                            f"Reply path: {path or 'n/a'}"
                        )[:1024],
                        inline=False,
                    )
                embed.add_field(
                    name="Output path",
                    value=(
                        f"Provider: **{payload['runtime']['writer_provider']}**\n"
                        f"Model: **{payload['runtime']['writer_model'] or 'none'}**\n"
                        f"Writer allowed: **{writer_allowed}**\n"
                        f"Writer called: **{payload['runtime']['writer_attempted']}**\n"
                        f"Reply sent: **{discord_reply_sent}**\n"
                        f"Memory selector: **{payload['runtime']['memory_selector']}** "
                        f"({'applied' if payload['runtime']['memory_selector_applied'] else 'gated off'})"
                    ),
                    inline=True,
                )
                embed.add_field(
                    name="Message signals",
                    value=(
                        f"Mentioned: **{bool(metadata.get('mentioned'))}**\n"
                        f"Text address: **{bool(metadata.get('lexically_addressed'))}**\n"
                        f"Reply to Neuro: **{bool(metadata.get('replied_to_target'))}**\n"
                        f"Question: **{bool(metadata.get('question'))}**\n"
                        f"Messages since Neuro: **{int(metadata.get('messages_since_agent', 0))}**"
                    ),
                    inline=True,
                )
                embed.add_field(
                    name="Historical rates",
                    value=(
                        f"This author: {float(metadata.get('author_reply_rate_past', 0)) * 100:.1f}%\n"
                        f"This channel: {float(metadata.get('channel_reply_rate_past', 0)) * 100:.1f}%\n"
                        f"Global: {float(metadata.get('global_reply_rate_past', 0)) * 100:.1f}%\n"
                        f"Recent activity: {float(metadata.get('recent_agent_activity', 0)) * 100:.1f}%"
                    ),
                    inline=True,
                )
                embed.add_field(
                    name="Memory",
                    value=(
                        f"Writer memories: **{len(result.memories)}**\n"
                        f"Gate best cosine: **"
                        f"{float(metadata.get('gate_memory_signals', {}).get('max_memory_similarity', 0)):.3f}**\n"
                        f"Writer top cosine: **"
                        f"{(result.memories[0].similarity if result.memories else 0):.3f}**\n"
                        f"Answered match: **"
                        f"{float(metadata.get('gate_memory_signals', {}).get('max_answered_similarity', 0)):.3f}**\n"
                        f"Recent answered: **"
                        f"{float(metadata.get('gate_memory_signals', {}).get('recent_answered_similarity', 0)):.3f}**\n"
                        f"Near-repeat cooldown: **"
                        f"{bool(metadata.get('gate_memory_signals', {}).get('recent_answered_near_duplicate'))}**\n"
                        f"Answered age: **"
                        f"{float(metadata.get('gate_memory_signals', {}).get('seconds_since_answered_match', 86400)):.0f}s**\n"
                        f"Exact repeat: **"
                        f"{bool(metadata.get('gate_memory_signals', {}).get('exact_duplicate'))}**\n"
                        f"Facts updated: **{len(result.facts_updated)}**"
                    ),
                    inline=True,
                )
                reranker = metadata.get("memory_reranker")
                if isinstance(reranker, dict) and reranker.get("applied"):
                    selected_probabilities = [
                        float(value)
                        for value in reranker.get("selected_probabilities", [])
                    ]
                    embed.add_field(
                        name="Answer-aware memory",
                        value=(
                            f"Candidates: **{int(reranker.get('candidates', 0))}**\n"
                            f"Changed cosine top 5: **"
                            f"{bool(reranker.get('changed_top5'))}**\n"
                            f"Top selection weight: **"
                            f"{(max(selected_probabilities) if selected_probabilities else 0) * 100:.1f}%**\n"
                            f"Selector margin: **"
                            f"{float(reranker.get('selector_margin', 0)):.3f}**"
                        ),
                        inline=True,
                    )
                embed.set_footer(text=f"event {event.event_id} · full metadata attached")
                encoded = json.dumps(
                    payload, ensure_ascii=False, indent=2, default=str
                ).encode("utf-8")
                await audit_channel.send(
                    embed=embed,
                    file=discord.File(
                        io.BytesIO(encoded),
                        filename=f"decision-{message.id}.json",
                    ),
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            except Exception as exc:
                print(
                    f"decision audit failed event={event.event_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

    try:
        client.run(bot_token, log_handler=None)
    finally:
        agent.close()
