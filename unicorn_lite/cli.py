from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

import torch
import httpx

from .agent import UnicornAgent
from .attribution import build_contextual_policy_dataset
from .credentials import load_external_env, verify_discord_identity
from .discord_data import backfill_discord, build_discord_labels, discord_inventory
from .discord_runtime import run_discord
from .encoder import make_encoder
from .external_turntaking import convert_ishiki, convert_when2speak
from .gate_training import train_response_gate
from .gate_evaluation import evaluate_response_gate
from .learned_policy import LearnedResponsePolicy
from .models import Experience
from .persona import DEFAULT_WEBWAIFU_BACKUP, import_webwaifu_persona
from .training import train_neuro_pairs, train_replay
from .writer import OllamaWriter, VercelWriter


ENCODER_CHOICES = ("local", "hash", "bert-base", "mpc-bert")


def _device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _default_checkpoint() -> str | None:
    configured = os.getenv("UNICORN_CHECKPOINT")
    if configured:
        return configured
    trained = Path(__file__).resolve().parents[1] / "data" / "neuro-controller.pt"
    return str(trained) if trained.exists() else None


def _gate_threshold_overrides(values: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for value in values:
        channel, separator, raw_threshold = value.partition("=")
        if not separator or not channel.strip():
            raise ValueError(
                "gate threshold overrides must use CHANNEL_ID=PROBABILITY"
            )
        threshold = float(raw_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("gate threshold probability must be between 0 and 1")
        overrides[channel.strip()] = threshold
    return overrides


def _agent(args: argparse.Namespace) -> UnicornAgent:
    device = _device(args.device)
    encoder = make_encoder(args.encoder, device=device)
    writer_enabled = bool(
        getattr(args, "allow_writer", False)
        or getattr(args, "mode", "shadow") == "active"
    )
    writer_provider = getattr(args, "writer", os.getenv("UNICORN_WRITER", "ollama"))
    writer = None
    if writer_enabled:
        writer = OllamaWriter() if writer_provider == "ollama" else VercelWriter()
    checkpoint = getattr(args, "checkpoint", None)
    if args.encoder == "hash" and checkpoint == _default_checkpoint():
        checkpoint = None
    agent = UnicornAgent(
        db_path=args.db,
        encoder=encoder,
        device=device,
        writer=writer,
        checkpoint=checkpoint,
    )
    if getattr(args, "command", None) == "discord":
        gate_checkpoint = Path(args.gate_checkpoint)
        if not gate_checkpoint.exists():
            agent.close()
            raise RuntimeError(
                f"Discord runtime requires an accepted learned gate: {gate_checkpoint}"
            )
        gate_encoder_name = getattr(args, "gate_encoder", None)
        gate_encoder = (
            make_encoder(gate_encoder_name, device=device)
            if gate_encoder_name
            else encoder
        )
        agent.policy = LearnedResponsePolicy(
            gate_checkpoint,
            gate_encoder,
            agent.store,
            device=device,
            threshold_overrides=_gate_threshold_overrides(
                getattr(args, "gate_threshold_override", [])
            ),
        )
    return agent


def _result_dict(result: object) -> dict[str, object]:
    data = asdict(result)
    data["memories"] = [
        {
            "event_id": hit["event_id"],
            "actor": hit["actor"],
            "similarity": round(hit["similarity"], 3),
            "content": hit["content"][:100],
        }
        for hit in data["memories"]
    ]
    data["surprise"] = round(float(data["surprise"]), 3)
    return data


def doctor(args: argparse.Namespace) -> None:
    ollama_model = os.getenv("UNICORN_OLLAMA_MODEL", "neuro-gemma4-rp:latest")
    try:
        ollama_models = httpx.get(
            "http://127.0.0.1:11434/api/tags", timeout=5
        ).json().get("models", [])
        ollama_names = {str(item.get("name")) for item in ollama_models}
        ollama_available = True
    except (httpx.HTTPError, ValueError):
        ollama_names = set()
        ollama_available = False
    report = {
        "env_source": str(getattr(args, "env_source", None) or "not found"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gib": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )
        if torch.cuda.is_available()
        else None,
        "vercel_writer_configured": bool(os.getenv("AI_GATEWAY_API_KEY")),
        "default_writer": os.getenv("UNICORN_WRITER", "ollama"),
        "ollama_available": ollama_available,
        "ollama_model": ollama_model,
        "ollama_model_installed": ollama_model in ollama_names,
        "discord_configured": bool(os.getenv("DISCORD_BOT_TOKEN")),
    }
    if args.verify_discord:
        report["discord_identity"] = asyncio.run(verify_discord_identity())
    print(json.dumps(report, indent=2))


async def _demo(args: argparse.Namespace) -> None:
    agent = _agent(args)
    samples = [
        Experience("i prefer local models", actor="subsect"),
        Experience("random background chatter", actor="someone"),
        Experience(
            "can you remember what kind of models i prefer?",
            actor="subsect",
            metadata={"direct": True},
        ),
    ]
    try:
        for event in samples:
            result = await agent.ingest(event, allow_writer=args.allow_writer)
            print(json.dumps(_result_dict(result), indent=2))
        print(
            json.dumps(
                {
                    "events_persisted": agent.store.event_count(),
                    "current_facts": agent.store.current_facts(),
                    "writer_was_allowed": args.allow_writer,
                },
                indent=2,
            )
        )
    finally:
        agent.close()


async def _ingest(args: argparse.Namespace) -> None:
    agent = _agent(args)
    try:
        result = await agent.ingest(
            Experience(
                args.text,
                actor=args.actor,
                source="cli",
                metadata={"direct": args.direct},
            ),
            allow_writer=args.allow_writer,
        )
        print(json.dumps(_result_dict(result), indent=2))
    finally:
        agent.close()


async def _chat(args: argparse.Namespace) -> None:
    agent = _agent(args)
    print("Unicorn Lite is listening. Type /quit to stop.")
    try:
        while True:
            text = await asyncio.to_thread(input, "> ")
            if text.strip() in {"/quit", "/exit"}:
                break
            result = await agent.ingest(
                Experience(text, actor=args.actor, metadata={"direct": True}),
                allow_writer=args.allow_writer,
            )
            if result.reply:
                print(result.reply)
            elif result.writer_error:
                print(f"[{result.decision.action}; {result.writer_error}]")
            else:
                print(f"[{result.decision.action}: {result.decision.reason}]")
    finally:
        agent.close()


def _train(args: argparse.Namespace) -> None:
    device = _device(args.device)
    encoder = make_encoder(args.encoder, device=device)
    report = train_replay(
        args.jsonl,
        encoder,
        args.output,
        device=device,
        epochs=args.epochs,
    )
    print(json.dumps(report, indent=2))


def _train_neuro(args: argparse.Namespace) -> None:
    device = _device(args.device)
    encoder = make_encoder(args.encoder, device=device)
    report = train_neuro_pairs(
        args.train_jsonl,
        args.validation_jsonl,
        encoder,
        args.output,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(report, indent=2))


def _train_gate(args: argparse.Namespace) -> None:
    device = _device(args.device)
    encoder = make_encoder(args.encoder, device=device)
    report = train_response_gate(
        args.policy_jsonl,
        encoder,
        args.output,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_checkpoint_path=args.init_checkpoint,
        architecture=args.architecture,
        max_ponder_steps=args.max_ponder_steps,
        min_ponder_steps=args.min_ponder_steps,
        halt_threshold=args.halt_threshold,
        compute_penalty=args.compute_penalty,
        stability_penalty=args.stability_penalty,
        hard_negative_paths=args.hard_negative_jsonl,
        hard_negative_max_words=args.hard_negative_max_words,
        hard_negative_exclude_questions=args.hard_negative_exclude_questions,
        hard_negative_exclude_requests=args.hard_negative_exclude_requests,
        positive_anchor_paths=args.positive_anchor_jsonl,
        positive_anchor_count=args.positive_anchor_count,
        refractory_paths=args.refractory_jsonl,
    )
    print(json.dumps(report, indent=2))


def _contextual_labels(args: argparse.Namespace) -> None:
    device = _device(args.device)
    encoder = make_encoder(args.encoder, device=device)
    report = build_contextual_policy_dataset(
        args.history,
        args.target_author_id,
        encoder,
        args.output,
        lookback_hours=args.lookback_hours,
        context_messages=args.context_messages,
    )
    print(json.dumps(report, indent=2))


def _evaluate_gate(args: argparse.Namespace) -> None:
    device = _device(args.device)
    encoder = make_encoder(args.encoder, device=device)
    report = evaluate_response_gate(
        args.policy_jsonl,
        args.checkpoint,
        encoder,
        device=device,
        output_path=args.output,
        forced_ponder_steps=args.forced_ponder_steps,
        include_examples=args.include_examples,
        threshold_override=args.threshold,
    )
    print(json.dumps(report, indent=2))


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default=os.getenv("UNICORN_DB", "./data/unicorn.db"))
    parser.add_argument(
        "--encoder", choices=ENCODER_CHOICES, default=os.getenv("UNICORN_ENCODER", "local")
    )
    parser.add_argument("--device", default=os.getenv("UNICORN_DEVICE", "cuda"))
    parser.add_argument("--checkpoint", default=_default_checkpoint())
    parser.add_argument(
        "--writer",
        choices=("ollama", "vercel"),
        default=os.getenv("UNICORN_WRITER", "ollama"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unicorn-lite")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor_parser = commands.add_parser("doctor", help="show usable local hardware")
    doctor_parser.add_argument("--verify-discord", action="store_true")
    doctor_parser.set_defaults(func=doctor)

    demo_parser = commands.add_parser("demo", help="run a three-event persistence demo")
    _common(demo_parser)
    demo_parser.add_argument(
        "--allow-writer", "--call-api", dest="allow_writer", action="store_true"
    )
    demo_parser.set_defaults(func=lambda args: asyncio.run(_demo(args)))

    ingest_parser = commands.add_parser("ingest", help="process one experience")
    _common(ingest_parser)
    ingest_parser.add_argument("text")
    ingest_parser.add_argument("--actor", default="user")
    ingest_parser.add_argument("--direct", action="store_true")
    ingest_parser.add_argument(
        "--allow-writer",
        action="store_true",
        help="explicitly permit one selected writer call if policy selects REPLY",
    )
    ingest_parser.set_defaults(func=lambda args: asyncio.run(_ingest(args)))

    chat_parser = commands.add_parser("chat", help="interactive local session")
    _common(chat_parser)
    chat_parser.add_argument("--actor", default="user")
    chat_parser.add_argument(
        "--allow-writer",
        action="store_true",
        help="explicitly permit selected writer calls for replies",
    )
    chat_parser.set_defaults(func=lambda args: asyncio.run(_chat(args)))

    train_parser = commands.add_parser("train", help="train the core on replay JSONL")
    train_parser.add_argument("jsonl")
    train_parser.add_argument("--output", default="./data/controller.pt")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--encoder", choices=ENCODER_CHOICES, default="local")
    train_parser.add_argument("--device", default=os.getenv("UNICORN_DEVICE", "cuda"))
    train_parser.set_defaults(func=_train)

    neuro_dir = Path.home() / "Downloads" / "neuro-merged"
    neuro_parser = commands.add_parser(
        "train-neuro", help="train and validate the controller on Neuro Q/A pairs"
    )
    neuro_parser.add_argument(
        "--train-jsonl", default=str(neuro_dir / "openai-train.jsonl")
    )
    neuro_parser.add_argument(
        "--validation-jsonl", default=str(neuro_dir / "openai-val.jsonl")
    )
    neuro_parser.add_argument("--output", default="./data/neuro-controller.pt")
    neuro_parser.add_argument("--epochs", type=int, default=12)
    neuro_parser.add_argument("--batch-size", type=int, default=32)
    neuro_parser.add_argument("--encoder", choices=ENCODER_CHOICES, default="local")
    neuro_parser.add_argument("--device", default=os.getenv("UNICORN_DEVICE", "cuda"))
    neuro_parser.set_defaults(func=_train_neuro)

    discord_parser = commands.add_parser("discord", help="listen to all Discord messages")
    _common(discord_parser)
    discord_parser.add_argument(
        "--mode", choices=("shadow", "active"), default="shadow"
    )
    discord_parser.add_argument(
        "--run-seconds",
        type=float,
        help="close cleanly after a bounded live validation window",
    )
    discord_parser.add_argument(
        "--allow-unsolicited",
        action="store_true",
        help="permit active-mode replies without a DM or explicit mention",
    )
    discord_parser.add_argument(
        "--gate-checkpoint",
        default="./data/context-response-gate.pt",
        help="accepted learned Discord action-policy checkpoint",
    )
    discord_parser.add_argument(
        "--gate-encoder",
        choices=ENCODER_CHOICES,
        help=(
            "optional dedicated local encoder for the response gate; keeps the "
            "memory/core encoder and its persisted state unchanged"
        ),
    )
    discord_parser.add_argument(
        "--gate-threshold-override",
        action="append",
        default=[],
        metavar="CHANNEL_ID=PROBABILITY",
        help="apply a measured channel-specific writer-cost threshold",
    )
    discord_parser.add_argument(
        "--unsolicited-channel-id",
        action="append",
        default=[],
        help="limit learned unsolicited replies to specific channels",
    )
    discord_parser.add_argument(
        "--decision-audit-channel-id",
        help="send a no-LLM decision summary and full JSON metadata to this channel",
    )
    discord_parser.add_argument(
        "--decision-audit-source-channel-id",
        action="append",
        default=[],
        help="limit Discord decision audits to messages from specific channels",
    )
    discord_parser.set_defaults(
        func=lambda args: run_discord(
            _agent(args),
            mode=args.mode,
            run_seconds=args.run_seconds,
            allow_unsolicited=args.allow_unsolicited,
            unsolicited_channel_ids=set(args.unsolicited_channel_id) or None,
            decision_audit_channel_id=args.decision_audit_channel_id,
            decision_audit_source_channel_ids=(
                set(args.decision_audit_source_channel_id) or None
            ),
        )
    )

    inventory_parser = commands.add_parser(
        "discord-inventory", help="list readable Discord guild text channels"
    )
    inventory_parser.add_argument(
        "--output", default="./data/discord-inventory.json"
    )
    inventory_parser.set_defaults(
        func=lambda args: discord_inventory(output_path=args.output)
    )

    backfill_parser = commands.add_parser(
        "discord-backfill", help="read Discord history into local chronological JSONL"
    )
    backfill_parser.add_argument(
        "--output", default="./data/discord-history.jsonl"
    )
    backfill_parser.add_argument("--guild-id")
    backfill_parser.add_argument("--channel-id", action="append", default=[])
    backfill_parser.add_argument(
        "--limit-per-channel",
        type=int,
        default=2000,
        help="messages per channel; use 0 for complete available history",
    )
    backfill_parser.set_defaults(
        func=lambda args: backfill_discord(
            args.output,
            guild_id=args.guild_id,
            channel_ids=set(args.channel_id) or None,
            limit_per_channel=(None if args.limit_per_channel == 0 else args.limit_per_channel),
        )
    )

    labels_parser = commands.add_parser(
        "discord-labels", help="derive conservative reply/observe labels from history"
    )
    labels_parser.add_argument(
        "history", help="JSONL produced by discord-backfill"
    )
    labels_parser.add_argument("--target-author-id", required=True)
    labels_parser.add_argument("--output", default="./data/discord-policy.jsonl")
    labels_parser.add_argument("--mention-lookback-hours", type=float, default=24.0)
    labels_parser.set_defaults(
        func=lambda args: build_discord_labels(
            args.history,
            args.target_author_id,
            args.output,
            mention_lookback_hours=args.mention_lookback_hours,
        )
    )

    contextual_parser = commands.add_parser(
        "discord-context-labels",
        help="attribute delayed responses using semantics and conversation structure",
    )
    contextual_parser.add_argument("history", nargs="+")
    contextual_parser.add_argument("--target-author-id", required=True)
    contextual_parser.add_argument("--output", default="./data/discord-context-policy.jsonl")
    contextual_parser.add_argument("--lookback-hours", type=float, default=24.0)
    contextual_parser.add_argument("--context-messages", type=int, default=10)
    contextual_parser.add_argument("--encoder", choices=ENCODER_CHOICES, default="local")
    contextual_parser.add_argument("--device", default=os.getenv("UNICORN_DEVICE", "cuda"))
    contextual_parser.set_defaults(func=_contextual_labels)

    gate_parser = commands.add_parser(
        "train-discord-gate", help="train a recurrent local reply/silence classifier"
    )
    gate_parser.add_argument("policy_jsonl")
    gate_parser.add_argument("--output", default="./data/discord-response-gate.pt")
    gate_parser.add_argument("--epochs", type=int, default=50)
    gate_parser.add_argument("--batch-size", type=int, default=32)
    gate_parser.add_argument("--learning-rate", type=float, default=8e-4)
    gate_parser.add_argument("--encoder", choices=ENCODER_CHOICES, default="local")
    gate_parser.add_argument("--device", default=os.getenv("UNICORN_DEVICE", "cuda"))
    gate_parser.add_argument(
        "--init-checkpoint",
        help="initialize from another compatible gate before fine-tuning",
    )
    gate_parser.add_argument(
        "--architecture", choices=("single", "ponder"), default="single"
    )
    gate_parser.add_argument("--max-ponder-steps", type=int, default=8)
    gate_parser.add_argument("--min-ponder-steps", type=int, default=1)
    gate_parser.add_argument("--halt-threshold", type=float, default=0.5)
    gate_parser.add_argument("--compute-penalty", type=float, default=0.015)
    gate_parser.add_argument("--stability-penalty", type=float, default=0.01)
    gate_parser.add_argument(
        "--hard-negative-jsonl",
        action="append",
        default=[],
        help=(
            "training-only policy JSONL; only unanswered direct OBSERVE rows "
            "at or before the base training cutoff are admitted"
        ),
    )
    gate_parser.add_argument(
        "--hard-negative-max-words",
        type=int,
        default=0,
        help="admit only hard negatives at or below this word count; 0 disables",
    )
    gate_parser.add_argument(
        "--hard-negative-exclude-questions",
        action="store_true",
        help="exclude question-like turns from hard-negative augmentation",
    )
    gate_parser.add_argument(
        "--hard-negative-exclude-requests",
        action="store_true",
        help="exclude imperative/request-like turns from hard-negative augmentation",
    )
    gate_parser.add_argument(
        "--positive-anchor-jsonl",
        action="append",
        default=[],
        help="training-only source of concise direct REPLY anchors",
    )
    gate_parser.add_argument(
        "--positive-anchor-count",
        type=int,
        default=0,
        help="number of chronologically spread positive anchors to replay",
    )
    gate_parser.add_argument(
        "--refractory-jsonl",
        action="append",
        default=[],
        help=(
            "training-only paired repeat negatives and legitimate follow-up controls"
        ),
    )
    gate_parser.set_defaults(func=_train_gate)

    evaluation_parser = commands.add_parser(
        "evaluate-discord-gate",
        help="evaluate a learned gate on local or external policy JSONL",
    )
    evaluation_parser.add_argument("policy_jsonl")
    evaluation_parser.add_argument("checkpoint")
    evaluation_parser.add_argument("--output")
    evaluation_parser.add_argument("--encoder", choices=ENCODER_CHOICES, default="local")
    evaluation_parser.add_argument("--device", default=os.getenv("UNICORN_DEVICE", "cuda"))
    evaluation_parser.add_argument(
        "--forced-ponder-steps",
        type=int,
        help="ablation: bypass learned halting and use exactly this many local steps",
    )
    evaluation_parser.add_argument(
        "--include-examples",
        action="store_true",
        help="include per-event probabilities and decisions in the report",
    )
    evaluation_parser.add_argument(
        "--threshold",
        type=float,
        help="evaluate at this decision threshold instead of the checkpoint default",
    )
    evaluation_parser.set_defaults(func=_evaluate_gate)

    external_parser = commands.add_parser(
        "import-turntaking-data",
        help="convert external SPEAK/SILENT datasets into the local policy schema",
    )
    external_parser.add_argument("format", choices=("ishiki", "when2speak"))
    external_parser.add_argument("input", nargs="+")
    external_parser.add_argument("--output", required=True)
    external_parser.set_defaults(
        func=lambda args: print(
            json.dumps(
                convert_ishiki(args.input, args.output)
                if args.format == "ishiki"
                else convert_when2speak(args.input[0], args.output),
                indent=2,
            )
        )
    )

    persona_parser = commands.add_parser(
        "import-persona", help="extract one persona from a WebWaifu local backup"
    )
    persona_parser.add_argument("--backup", default=str(DEFAULT_WEBWAIFU_BACKUP))
    persona_parser.add_argument("--persona-id", default="neuro-sama")
    persona_parser.add_argument("--output", default="./data/neuro-persona.txt")
    persona_parser.set_defaults(
        func=lambda args: print(
            json.dumps(
                import_webwaifu_persona(
                    args.backup, args.output, persona_id=args.persona_id
                ),
                indent=2,
            )
        )
    )
    return parser


def main() -> None:
    source = load_external_env()
    args = build_parser().parse_args()
    if getattr(args, "command", None) == "doctor":
        args.env_source = str(source) if source else None
    args.func(args)
