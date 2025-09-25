"""Static configuration for financial statement detection."""
from __future__ import annotations

import re

OPS_HEAD = re.compile(
    r"(consolidated\s+)?statements?\s*[\W_]*\s*of\s*[\W_]*\s*"
    r"(operations|income|earnings)(?!.*comprehensive)",
    re.I,
)
BAL_HEAD = re.compile(
    r"(consolidated\s+)?balance\s*[\W_]*\s*sheets?"
    r"|statements?\s*[\W_]*\s*of\s*[\W_]*\s*financial\s*[\W_]*\s*position",
    re.I,
)
CF_HEAD = re.compile(
    r"(consolidated\s+)?statements?\s*[\W_]*\s*of\s*[\W_]*\s*cash\s*[\W_]*\s*flows?",
    re.I,
)
EXCLUDE = re.compile(r"\bItem\s+7A\b.*Market\s+Risk\b", re.I)
CONTINUED = re.compile(r"\(continued\)", re.I)
INDEX_PAGE = re.compile(r"Index\s+to\s+Consolidated\s+Financial\s+Statements", re.I)
TABLE_OF_CONTENTS = re.compile(r"Table\s+of\s+Contents", re.I)

OTHER_STATEMENT_PATS = [
    re.compile(r"statements?\s+of\s+comprehensive\s+income", re.I),
    re.compile(r"statements?\s+of\s+shareholders'\s+equity", re.I),
    re.compile(r"statements?\s+of\s+stockholders'\s+equity", re.I),
    re.compile(r"statements?\s+of\s+equity", re.I),
]

OPS_ANCH = [
    "revenue",
    "net revenue",
    "sales",
    "cost of revenue",
    "gross profit",
    "selling and marketing",
    "marketing and sales",
    "research and development",
    "general and administrative",
    "operating income",
    "income from operations",
    "earnings per share",
    "net income",
    "net loss",
]
BAL_ANCH = [
    "total assets",
    "total liabilities",
    "cash and cash equivalents",
    "accounts receivable",
    "deferred revenue",
    "stockholders’ equity",
    "stockholders' equity",
    "total liabilities and stockholders",
    "total liabilities and shareholders",
]
CF_ANCH = [
    "net cash provided by operating activities",
    "net cash used in investing activities",
    "net cash provided by financing activities",
    "operating activities",
    "investing activities",
    "financing activities",
    "ending cash and cash equivalents",
]

ANCHOR_CONFIG = {
    "balance": (BAL_HEAD, BAL_ANCH, 4),
    "income": (OPS_HEAD, OPS_ANCH, 4),
    "cash": (CF_HEAD, CF_ANCH, 3),
}

DEFAULT_MAX_SCAN = 12
DEFAULT_SEARCH_WINDOW = 20
DEFAULT_TOC_DELTA = 6
DEFAULT_LOW_TEXT_THRESHOLD = 2000
DEFAULT_LOW_TEXT_SAMPLE = 8

MAX_STATEMENT_EXTENSION_PAGES = 4
MIN_CONTINUATION_DENSITY = 12
CONTINUATION_ANCHOR_BONUS = 1
