from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from unicorn_lite.persona import import_webwaifu_persona


class PersonaTests(unittest.TestCase):
    def test_import_copies_only_selected_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "backup.json"
            output = Path(directory) / "persona.txt"
            source.write_text(
                json.dumps(
                    {
                        "providerSecrets": [{"secret": "must-not-copy"}],
                        "state": {
                            "personas": [
                                {
                                    "id": "neuro-sama",
                                    "name": "Neuro-sama",
                                    "systemPrompt": "chaotic neuro prompt",
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = import_webwaifu_persona(source, output)
            self.assertEqual(output.read_text(encoding="utf-8").strip(), "chaotic neuro prompt")
            self.assertNotIn("must-not-copy", output.read_text(encoding="utf-8"))
            self.assertFalse(report["provider_secrets_copied"])


if __name__ == "__main__":
    unittest.main()
