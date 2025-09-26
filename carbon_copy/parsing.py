from __future__ import annotations

import re
from typing import Optional, Tuple

from .constants import (
    DASH_VALUES,
    NON_MONETARY_KEYWORDS,
    PAREN_NUMBER,
)


def parse_number(cell: object) -> Optional[float]:
    if cell is None:
        return None
    text = str(cell).strip()
    if text in DASH_VALUES:
        return None
    cleaned = text.replace(",", "")
    if PAREN_NUMBER.match(cleaned):
        cleaned = "-" + cleaned.strip("()")
    try:
        return float(cleaned)
    except ValueError:
        return None


def detect_units(text_block: str) -> Tuple[str, float, Optional[str]]:
    lower = (text_block or "").lower()
    if "in millions" in lower:
        return "In millions", 1.0, None
    if "in thousands" in lower:
        return "In thousands", 1.0 / 1000.0, None
    if "in billions" in lower:
        return "In billions", 1000.0, None
    return (
        "Units not stated",
        1.0,
        "Units Assumption: assumed dollars; normalized to USD millions",
    )


def is_non_monetary_label(label: str) -> bool:
    lowered = (label or "").lower()
    return any(keyword in lowered for keyword in NON_MONETARY_KEYWORDS)
