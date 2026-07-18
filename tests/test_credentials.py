from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from unicorn_lite.credentials import load_external_env


class CredentialTests(unittest.TestCase):
    def test_external_env_loads_without_overwriting_process_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / ".env"
            source.write_text(
                "DISCORD_BOT_TOKEN=from-file\nAI_GATEWAY_API_KEY='gateway-file'\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"DISCORD_BOT_TOKEN": "already-set"},
                clear=False,
            ):
                os.environ.pop("AI_GATEWAY_API_KEY", None)
                loaded = load_external_env(source)
                self.assertEqual(loaded, source.resolve())
                self.assertEqual(os.environ["DISCORD_BOT_TOKEN"], "already-set")
                self.assertEqual(os.environ["AI_GATEWAY_API_KEY"], "gateway-file")


if __name__ == "__main__":
    unittest.main()
