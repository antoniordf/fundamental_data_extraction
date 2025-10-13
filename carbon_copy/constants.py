from __future__ import annotations

import re
from typing import Pattern

from selector.models import StatementType


STATEMENT_TITLE_PATTERNS: dict[StatementType, Pattern[str]] = {
    StatementType.INCOME: re.compile(r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+(OPERATIONS|INCOME|EARNINGS)", re.I),
    StatementType.BALANCE: re.compile(r"CONSOLIDATED\s+BALANCE\s+SHEETS?", re.I),
    StatementType.CASH: re.compile(r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+CASH\s+FLOWS?", re.I),
}


DEMO_AUDITED_LABEL = "Audited (10-K)"

SAVED_UNITS_LABEL = "USD millions (normalized)"

NON_MONETARY_KEYWORDS = (
    "per share",
    "per common share",
    "earnings per share",
    "weighted-average shares",
    "shares outstanding",
    "percent",
    "%",
    "basis points",
    "ratio",
    "effective tax rate",
    "days",
    "headcount",
    "units",
)

INCOME_EXPENSE_KEYWORDS = re.compile(
    r"(cost|cost of sales|cost of revenue|expense|selling|general|administrative|research and development|"
    r"\br&d\b|restructuring|impairment|acquisition-related|amortization of intangible|interest expense|provision)",
    re.I,
)

CASH_NEGATIVE_KEYWORDS = re.compile(
    r"(payment|paid|purchase|additions|acquisition|repayments|extinguishment|settlement|capital expenditure|"
    r"repurchase|dividend|taxes paid|withholding|lease payment)",
    re.I,
)

CASH_POSITIVE_KEYWORDS = re.compile(
    r"(proceeds|issuance|collections|maturities|disposals|sale of|borrowings|draw|capital contribution)",
    re.I,
)

TAX_BENEFIT_KEYWORDS = re.compile(r"(benefit|credit|relief)", re.I)
TAX_EXPENSE_KEYWORDS = re.compile(r"(expense|provision|charge)", re.I)

PAREN_NUMBER = re.compile(r"^\s*\(.*\)\s*$")

DASH_VALUES = {"", "-", "–", "—", "— —"}
