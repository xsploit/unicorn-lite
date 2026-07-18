from __future__ import annotations

import json
from pathlib import Path


DEFAULT_WEBWAIFU_BACKUP = (
    Path.home() / "Downloads" / "yourwifey-local-backup-2026-05-22T08-16-55.json"
)
DEFAULT_PERSONA_PATH = Path(__file__).resolve().parents[1] / "data" / "neuro-persona.txt"


def import_webwaifu_persona(
    backup_path: str | Path = DEFAULT_WEBWAIFU_BACKUP,
    output_path: str | Path = DEFAULT_PERSONA_PATH,
    persona_id: str = "neuro-sama",
) -> dict[str, object]:
    source = Path(backup_path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    personas = payload.get("state", {}).get("personas", [])
    selected = next(
        (item for item in personas if str(item.get("id")) == persona_id), None
    )
    if selected is None:
        raise ValueError(f"persona {persona_id!r} was not found in {source}")
    prompt = str(selected.get("systemPrompt", "")).strip()
    if not prompt:
        raise ValueError(f"persona {persona_id!r} has an empty system prompt")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(prompt + "\n", encoding="utf-8")
    return {
        "persona_id": persona_id,
        "persona_name": selected.get("name"),
        "characters": len(prompt),
        "source": str(source.resolve()),
        "output": str(destination.resolve()),
        "provider_secrets_copied": False,
    }


def load_persona(path: str | Path = DEFAULT_PERSONA_PATH) -> str:
    source = Path(path)
    if not source.is_file():
        return (
            "You are Neuro, an entertaining AI VTuber with a chaotic, dry, playful "
            "personality. Be concise, conversational, and never use customer-service phrasing."
        )
    return source.read_text(encoding="utf-8").strip()
