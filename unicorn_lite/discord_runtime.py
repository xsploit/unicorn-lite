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


def is_speech_eligible(
    *,
    mode: str,
    is_bot: bool,
    direct: bool,
    mentioned: bool,
    lexically_addressed: bool,
    replied_to_target: bool,
    allow_unsolicited: bool,
    channel: str,
    unsolicited_channel_ids: set[str] | None,
) -> bool:
    """Return whether this turn may reach a writer if the learned gate says REPLY."""
    directed = direct or mentioned or lexically_addressed or replied_to_target
    unsolicited = allow_unsolicited and (
        not unsolicited_channel_ids or channel in unsolicited_channel_ids
    )
    return mode == "active" and not is_bot and (directed or unsolicited)


def audit_delivery_mode(
    *, can_send: bool = True, can_embed: bool, can_attach: bool
) -> str:
    if not can_send:
        return "none"
    if not can_embed:
        return "text"
    return "embed_file" if can_attach else "embed"


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
                "discord-emdr2-v1"
                if getattr(agent, "emdr2_retriever", None) is not None
                else (
                    "answer-aware-v1"
                    if agent.memory_reranker is not None
                    else "cosine-top5"
                )
            ),
            "memory_selector_applied": bool(
                event.metadata.get("emdr2", {}).get("applied")
                or event.metadata.get("memory_reranker", {}).get("applied")
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
    decision_audit_same_channel: bool = False,
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
            message.author.id == client.user.id
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
        speech_eligible = is_speech_eligible(
            mode=mode,
            is_bot=is_bot,
            direct=direct,
            mentioned=mentioned,
            lexically_addressed=lexically_addressed,
            replied_to_target=replied_to_target,
            allow_unsolicited=allow_unsolicited,
            channel=event.channel,
            unsolicited_channel_ids=unsolicited_channel_ids,
        )
        can_send_response = True
        if message.guild is not None and message.guild.me is not None:
            permissions_for = getattr(message.channel, "permissions_for", None)
            if permissions_for is not None:
                channel_permissions = permissions_for(message.guild.me)
                can_send_response = bool(channel_permissions.send_messages)
                if isinstance(message.channel, discord.Thread):
                    can_send_response = can_send_response and bool(
                        channel_permissions.send_messages_in_threads
                    )
        writer_allowed = speech_eligible and can_send_response
        event.metadata["discord_send_allowed"] = can_send_response
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

        should_audit_same_channel = decision_audit_same_channel and writer_allowed
        should_audit_fixed_channel = bool(
            decision_audit_channel_id
            and not is_bot
            and (
                not decision_audit_source_channel_ids
                or event.channel in decision_audit_source_channel_ids
            )
        )
        should_audit = should_audit_same_channel or should_audit_fixed_channel
        if should_audit:
            try:
                if should_audit_same_channel:
                    audit_channel = message.channel
                else:
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
                ponder = metadata.get("ponder")
                ponder_steps = (
                    int(ponder.get("steps", 0)) if isinstance(ponder, dict) else 0
                )
                reason = result.decision.reason
                if len(reason) > 180:
                    reason = reason[:177] + "..."
                content = message.content[:240] or "*No text content*"
                if len(message.content) > 240:
                    content += "..."
                embed = discord.Embed(
                    title=(
                        f"🟢 REPLY · {probability * 100:.1f}%"
                        if did_reply
                        else f"🔴 SILENCE · {probability * 100:.1f}%"
                    ),
                    url=message.jump_url,
                    description=(
                        f"> {content}\n"
                        f"Threshold **{threshold * 100:.1f}%** · "
                        f"margin **{margin * 100:+.1f}** · "
                        f"tier **{result.decision.compute_tier}**"
                        + (f" · **{ponder_steps}** steps" if ponder_steps else "")
                        + f"\n{reason}"
                    ),
                    color=(0x2ECC71 if did_reply else 0xE74C3C),
                    timestamp=message.created_at,
                )
                embed.add_field(
                    name="Signals",
                    value=(
                        f"mention {'✓' if metadata.get('mentioned') else '–'} · "
                        f"address {'✓' if metadata.get('lexically_addressed') else '–'}\n"
                        f"reply {'✓' if metadata.get('replied_to_target') else '–'} · "
                        f"question {'✓' if metadata.get('question') else '–'}"
                    ),
                    inline=True,
                )
                gate_signals = metadata.get("gate_memory_signals", {})
                reranker = metadata.get("memory_reranker")
                emdr2 = metadata.get("emdr2")
                reranker_applied = isinstance(reranker, dict) and reranker.get("applied")
                memory_lines = (
                    f"cosine **{float(gate_signals.get('max_memory_similarity', 0)):.3f}** · "
                    f"answered **{float(gate_signals.get('max_answered_similarity', 0)):.3f}**"
                )
                if reranker_applied:
                    selected_probabilities = [
                        float(value)
                        for value in reranker.get("selected_probabilities", [])
                    ]
                    memory_lines += (
                        f"\n{int(reranker.get('candidates', 0))}→{len(result.memories)} · "
                        f"top weight **{(max(selected_probabilities) if selected_probabilities else 0) * 100:.1f}%** · "
                        f"changed {'✓' if reranker.get('changed_top5') else '–'}"
                    )
                elif isinstance(emdr2, dict) and emdr2.get("applied"):
                    memory_lines += (
                        f"\nEMDR² index **{int(emdr2.get('indexed_memories', 0))}** · "
                        f"changed {'✓' if emdr2.get('changed_top5') else '–'}"
                    )
                else:
                    memory_lines += "\nselector gated off"
                embed.add_field(
                    name="Memory",
                    value=memory_lines,
                    inline=True,
                )
                model = payload["runtime"]["writer_model"] or "none"
                if len(model) > 28:
                    model = "…" + model[-27:]
                embed.add_field(
                    name="Output",
                    value=(
                        f"model **{model}**\n"
                        f"called {'✓' if payload['runtime']['writer_attempted'] else '–'} · "
                        f"sent {'✓' if discord_reply_sent else '–'}"
                    ),
                    inline=True,
                )
                can_embed = True
                can_attach = True
                can_send = True
                if message.guild is not None and message.guild.me is not None:
                    permissions_for = getattr(audit_channel, "permissions_for", None)
                    if permissions_for is not None:
                        permissions = permissions_for(message.guild.me)
                        can_send = bool(permissions.send_messages)
                        can_embed = bool(permissions.embed_links)
                        can_attach = bool(permissions.attach_files)
                footer = f"schema v2 · event {event.event_id[:8]}"
                embed.set_footer(
                    text=(
                        f"{footer} · full JSON attached"
                        if can_attach
                        else f"{footer} · JSON omitted (Attach Files unavailable)"
                    )
                )
                encoded = json.dumps(
                    payload, ensure_ascii=False, indent=2, default=str
                ).encode("utf-8")
                allowed_mentions = discord.AllowedMentions.none()
                delivery_mode = audit_delivery_mode(
                    can_send=can_send,
                    can_embed=can_embed,
                    can_attach=can_attach,
                )
                if delivery_mode == "none":
                    print(
                        f"decision audit skipped event={event.event_id}: "
                        "channel denies Send Messages"
                    )
                elif delivery_mode == "text":
                    await audit_channel.send(
                        content=(
                            f"{'REPLY' if did_reply else 'SILENCE'} "
                            f"{probability * 100:.1f}% · threshold "
                            f"{threshold * 100:.1f}% · event {event.event_id[:8]}"
                        ),
                        allowed_mentions=allowed_mentions,
                    )
                elif delivery_mode == "embed":
                    await audit_channel.send(
                        embed=embed, allowed_mentions=allowed_mentions
                    )
                else:
                    try:
                        await audit_channel.send(
                            embed=embed,
                            file=discord.File(
                                io.BytesIO(encoded),
                                filename=f"decision-{message.id}.json",
                            ),
                            allowed_mentions=allowed_mentions,
                        )
                    except discord.Forbidden:
                        # Channel overrides can make cached attachment permissions
                        # stale. Preserve the receipt even when the JSON upload is
                        # rejected atomically by Discord.
                        embed.set_footer(
                            text=f"{footer} · JSON omitted (upload forbidden)"
                        )
                        await audit_channel.send(
                            embed=embed, allowed_mentions=allowed_mentions
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
