from __future__ import annotations

from typing import Optional

from .constants import (
    CASH_NEGATIVE_KEYWORDS,
    CASH_POSITIVE_KEYWORDS,
    INCOME_EXPENSE_KEYWORDS,
    TAX_BENEFIT_KEYWORDS,
    TAX_EXPENSE_KEYWORDS,
)


def normalize_income(label: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    lower = label.lower()
    keep_sign_keywords = (
        "total ",
        "gross profit",
        "operating income",
        "income before",
        "net income",
        "net revenue",
        "other (income)",
        "income (expense)",
        "earnings per share",
    )
    if any(keyword in lower for keyword in keep_sign_keywords) or lower.endswith(" net"):
        return value
    if "tax" in lower:
        benefit = bool(TAX_BENEFIT_KEYWORDS.search(lower))
        expense = bool(TAX_EXPENSE_KEYWORDS.search(lower))
        if benefit and not expense:
            return abs(value)
        if expense and not benefit:
            return -abs(value)
        return abs(value) if value < 0 else -abs(value)
    if INCOME_EXPENSE_KEYWORDS.search(lower):
        return -abs(value)
    return value


def normalize_cashflow(label: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    lower = label.lower()
    if CASH_NEGATIVE_KEYWORDS.search(lower):
        return -abs(value)
    if CASH_POSITIVE_KEYWORDS.search(lower):
        return abs(value)
    return value
