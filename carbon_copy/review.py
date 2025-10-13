from __future__ import annotations

from typing import Dict, List

from .models import CarbonCopyResult, LabelReviewRequest, StatementBlock


def _value_columns(block: StatementBlock) -> List[str]:
    return [
        column
        for column in block.wide_table.columns
        if column not in {"Order Index", "Level", "Line Item (as printed)"}
    ]


def build_label_review_requests(block: StatementBlock) -> List[LabelReviewRequest]:
    """Construct per-row review requests for lines flagged as suspect."""
    if not block.suspect_labels:
        return []

    columns = _value_columns(block)
    requests: List[LabelReviewRequest] = []
    row_count = len(block.wide_table)
    skip_labels = {
        "common stock",
        "preferred stock",
        "treasury stock",
        "shares outstanding",
        "shares used - basic",
        "shares used - diluted",
    }
    for row_index, suspect in block.suspect_labels.items():
        if row_index >= row_count:
            continue
        row = block.wide_table.iloc[row_index]
        values = {column: row.get(column) for column in columns}
        cleaned_label = str(row["Line Item (as printed)"])
        if cleaned_label.lower() in skip_labels:
            continue
        requests.append(
            LabelReviewRequest(
                statement=block.statement_type,
                row_index=row_index,
                cleaned_label=cleaned_label,
                raw_label=suspect.original,
                reasons=list(suspect.reasons),
                values=values,
            )
        )
    return requests


def collect_label_review_requests(result: CarbonCopyResult) -> List[LabelReviewRequest]:
    """Flatten suspect rows across all statements for downstream LLM cleanup."""
    payloads: List[LabelReviewRequest] = []
    for block in result.blocks.values():
        payloads.extend(build_label_review_requests(block))
    return payloads
