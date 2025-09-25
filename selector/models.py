"""Data models for financial statement extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List


class StatementType(str, Enum):
    BALANCE = "balance"
    INCOME = "income"
    CASH = "cash"


@dataclass(slots=True)
class StatementSelectionResult:
    file: Path
    total_pages: int
    selected_pages: Dict[StatementType, List[int]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def selected_page_numbers(self) -> List[int]:
        pages: List[int] = []
        for idxs in self.selected_pages.values():
            pages.extend(idxs)
        return sorted(set(pages))
