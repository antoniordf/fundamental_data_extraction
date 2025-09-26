from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd

from selector.models import StatementType


def reorder_period_columns(columns: Iterable[str]) -> List[str]:
    cols = list(columns)

    def parse_key(label: str) -> tuple[int, str]:
        label_strip = label.strip()
        int_match = re.search(r"(\d{4})", label_strip)
        if int_match:
            return (int(int_match.group(1)), label_strip)
        month_match = re.search(r"(\w+)\s+\d{1,2},\s+(\d{4})", label_strip)
        if month_match:
            return (int(month_match.group(2)), label_strip)
        return (9999, label_strip)

    cols.sort(key=parse_key)
    return cols


def assemble_long(statement: StatementType, header: dict[str, str], wide: pd.DataFrame, page: str) -> pd.DataFrame:
    payload = []
    period_cols = wide.columns[3:]
    for _, row in wide.iterrows():
        for period in period_cols:
            value = row[period]
            processed_value = None
            if not pd.isna(value):
                if isinstance(value, str):
                    cleaned = value.replace(",", "").strip()
                    if cleaned.startswith("(") and cleaned.endswith(")"):
                        cleaned = f"-{cleaned[1:-1]}"
                    try:
                        processed_value = float(cleaned)
                    except ValueError:
                        processed_value = None
                else:
                    processed_value = float(value)

            payload.append(
                {
                    "Statement": header["Statement Name"],
                    "Order Index": row["Order Index"],
                    "Level": row["Level"],
                    "Line Item (as printed)": row["Line Item (as printed)"],
                    "Column Header (as printed)": period,
                    "Value (in millions)": processed_value,
                    "Original Units": header["Original Units"],
                    "Saved Units": header["Saved Units"],
                    "Units Assumption": header.get("Units Assumption", ""),
                    "Audited/Unaudited": header["Audited/Unaudited"],
                    "Page": page,
                }
            )
    return pd.DataFrame(payload)
