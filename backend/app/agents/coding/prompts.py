"""Prompt library: versioned defaults in ``app/agents/prompts/``, user overrides in data.

Defaults ship with the code (``system.md``, ``tools_text.md``, ``tools_native.md``,
``modes.md``, ``compact.md``, ``init.md``, ``variants.json``). Editing a prompt in
Settings writes an override to ``<data>/prompts/<name>``; "reset" deletes the override.
Templates use ``{{name}}`` placeholders.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

DEFAULTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_FILES = (
    "system.md", "chat.md", "tools_text.md", "tools_native.md", "tools_json.md", "modes.md", "compact.md", "init.md",
    "variants.json",
)
_VERSION = re.compile(r"<!--\s*prompt:\s*(\S+)\s*\|\s*version:\s*(\d+)\s*-->")
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


class PromptLibrary:
    def __init__(self, override_dir: Path, defaults_dir: Path = DEFAULTS_DIR) -> None:
        self.override_dir = Path(override_dir)
        self.defaults_dir = Path(defaults_dir)

    # --------------------------------------------------------------- files
    def _check(self, name: str) -> None:
        if name not in PROMPT_FILES:
            raise KeyError(f"Unknown prompt '{name}'. Known prompts: {', '.join(PROMPT_FILES)}")

    def default(self, name: str) -> str:
        self._check(name)
        return (self.defaults_dir / name).read_text(encoding="utf-8")

    def get(self, name: str) -> str:
        self._check(name)
        override = self.override_dir / name
        if override.is_file():
            return override.read_text(encoding="utf-8")
        return self.default(name)

    def is_overridden(self, name: str) -> bool:
        return (self.override_dir / name).is_file()

    def set(self, name: str, text: str) -> None:
        self._check(name)
        if name.endswith(".json"):
            json.loads(text)  # refuse invalid JSON with a clear error
        self.override_dir.mkdir(parents=True, exist_ok=True)
        (self.override_dir / name).write_text(text, encoding="utf-8")

    def reset(self, name: str) -> None:
        self._check(name)
        override = self.override_dir / name
        if override.is_file():
            override.unlink()

    def describe(self) -> list[dict[str, Any]]:
        items = []
        for name in PROMPT_FILES:
            text = self.get(name)
            match = _VERSION.search(text)
            if name.endswith(".json"):
                try:
                    version = json.loads(text).get("version")
                except json.JSONDecodeError:
                    version = None
            else:
                version = int(match.group(2)) if match else None
            items.append({"name": name, "version": version, "overridden": self.is_overridden(name), "content": text})
        return items

    # ------------------------------------------------------------ helpers
    def mode_instructions(self, mode: str) -> str:
        sections: dict[str, str] = {}
        current: Optional[str] = None
        for line in self.get("modes.md").splitlines():
            if line.startswith("## "):
                current = line[3:].strip()
                sections[current] = ""
            elif current is not None:
                sections[current] += line + "\n"
        return sections.get(mode, sections.get("agent", "")).strip()

    def variant_for(self, model_name: str) -> dict[str, Any]:
        try:
            data = json.loads(self.get("variants.json"))
        except json.JSONDecodeError:
            data = json.loads(self.default("variants.json"))
        name = (model_name or "").lower()
        for variant in data.get("variants", []):
            if str(variant.get("match", "")).lower() in name:
                return variant
        return {"tool_format": "auto"}

    @staticmethod
    def render(template: str, values: dict[str, Any]) -> str:
        text = _VERSION.sub("", template, count=1).lstrip("\n")
        rendered = _PLACEHOLDER.sub(lambda m: str(values.get(m.group(1), "")), text)
        return re.sub(r"\n{3,}", "\n\n", rendered).strip() + "\n"
