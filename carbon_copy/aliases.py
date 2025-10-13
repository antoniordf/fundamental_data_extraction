from __future__ import annotations

from pathlib import Path
from typing import Dict

from selector.models import StatementType


def _default_alias_map() -> Dict[StatementType, Dict[str, str]]:
    return {statement: {} for statement in StatementType}


def load_aliases(path: Path | None) -> Dict[StatementType, Dict[str, str]]:
    alias_map = _default_alias_map()
    if path is None:
        path = Path(__file__).with_name("aliases.yaml")
    if not path.exists():
        return alias_map
    current: StatementType | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            section = line.strip().strip(":")
            try:
                current = StatementType(section.lower())
            except ValueError:
                current = None
            continue
        if current is None:
            continue
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = [part.strip() for part in stripped.split(":", 1)]
        alias_map[current][key.lower()] = value
    return alias_map


def normalize_label(
    statement: StatementType,
    label: str,
    alias_map: Dict[StatementType, Dict[str, str]],
) -> str:
    cleaned = canonicalize(label)
    aliases = alias_map.get(statement, {})
    if cleaned.lower() in aliases:
        return aliases[cleaned.lower()]
    return cleaned


def canonicalize(label: str) -> str:
    text = (label or "").strip()
    text = text.replace("\t", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    if text.endswith(":") and text.count(":") == 1:
        text = text[:-1].strip()
    return text
