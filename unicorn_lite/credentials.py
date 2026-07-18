from __future__ import annotations

import os
from pathlib import Path

import httpx


DEFAULT_ENV_SOURCE = Path.cwd() / ".env"


def load_external_env(path: str | Path | None = None) -> Path | None:
    """Load an env file without overwriting values already present in the process."""
    source = Path(path or os.getenv("UNICORN_ENV_FILE") or DEFAULT_ENV_SOURCE)
    if not source.is_file():
        return None
    for raw_line in source.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)
    return source.resolve()


async def verify_discord_identity(token: str | None = None) -> dict[str, object]:
    bot_token = token or os.getenv("DISCORD_BOT_TOKEN")
    if not bot_token:
        return {"configured": False, "valid": False}
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            "https://discord.com/api/v10/users/@me",
            headers={"Authorization": f"Bot {bot_token}"},
        )
    if response.status_code != 200:
        return {
            "configured": True,
            "valid": False,
            "status": response.status_code,
        }
    body = response.json()
    discriminator = str(body.get("discriminator", "0"))
    username = str(body.get("username", "unknown"))
    tag = username if discriminator == "0" else f"{username}#{discriminator}"
    return {
        "configured": True,
        "valid": True,
        "bot": tag,
        "application_id": str(body.get("id", "")),
    }
