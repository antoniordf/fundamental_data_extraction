from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from selector.models import StatementType


@dataclass(slots=True)
class HeaderInfo:
    statement_name: str
    original_units: str
    saved_units: str
    units_assumption: Optional[str]
    audited_status: str
    source: str

    def as_dict(self) -> Dict[str, str]:
        data = {
            "Statement Name": self.statement_name,
            "Original Units": self.original_units,
            "Saved Units": self.saved_units,
            "Audited/Unaudited": self.audited_status,
            "Source (page number[s])": self.source,
        }
        if self.units_assumption:
            data["Units Assumption"] = self.units_assumption
        return data


@dataclass(slots=True)
class StatementBlock:
    statement_type: StatementType
    header: HeaderInfo
    wide_table: pd.DataFrame
    column_metadata: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.header.statement_name


@dataclass(slots=True)
class CarbonCopyResult:
    source_pdf: Path
    blocks: Dict[StatementType, StatementBlock]
    long_table: pd.DataFrame
    certification: pd.DataFrame

    def get_block(self, statement: StatementType) -> StatementBlock:
        return self.blocks[statement]
