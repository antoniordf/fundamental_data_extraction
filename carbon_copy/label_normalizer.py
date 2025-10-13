from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from selector.models import StatementType


@dataclass(slots=True)
class NormalizationResult:
    label: str
    reasons: List[str]


class LabelNormalizer:
    """Post-process extracted line item labels and flag suspicious remnants."""

    _CONNECTOR_COLLAPSE = re.compile(r"\b(and|or|of|in|the)\s+(and|or|of|in|the)\b", re.IGNORECASE)
    _MULTISPACE = re.compile(r"\s+")
    _NUMERIC_FOOTNOTE = re.compile(r"[\d\$,()]+$")
    _SHARE_KEYWORDS = re.compile(r"(?:shares?|issued|outstanding|authorized)", re.IGNORECASE)
    _RESPECTIVELY_TRAIL = re.compile(r"[;,]?\s*respectively\)?$", re.IGNORECASE)
    _FOOTNOTE_TOKENS: Tuple[str, ...] = (
        "respectively",
        "respectively)",
        "respectively).",
        "issued",
        "authorized",
        "outstanding",
    )
    _PAREN_START = re.compile(r"\([^)]*$")

    def normalize(
        self,
        statement: StatementType,
        label: str,
        raw_label: str,
        values: Sequence[Optional[float]],
    ) -> NormalizationResult:
        working = (label or "").strip()
        reasons: List[str] = []

        if not working and raw_label:
            working = raw_label.strip()
            if working:
                reasons.append("fallback_to_raw")

        cleaned, clean_reasons = self._trim_fragments(statement, working, raw_label)
        working = cleaned
        reasons.extend(clean_reasons)

        suspicion = self._detect_issues(statement, working, raw_label, values)
        reasons.extend(suspicion)

        canonical = self._canonicalize(working, raw_label)
        if canonical != working:
            working = canonical

        if not working:
            reasons.append("empty_after_normalization")

        # Collapse whitespace once at the end
        working = self._MULTISPACE.sub(" ", working).strip()

        return NormalizationResult(label=working, reasons=self._coalesce_reasons(reasons))

    # --- helpers -----------------------------------------------------------------

    def _trim_fragments(
        self,
        statement: StatementType,
        label: str,
        raw_label: str,
    ) -> Tuple[str, List[str]]:
        reasons: List[str] = []
        working = label

        # Remove duplicate connectors such as "and and"
        collapsed = self._CONNECTOR_COLLAPSE.sub(lambda m: m.group(1), working)
        if collapsed != working:
            reasons.append("collapsed_connectors")
            working = collapsed

        # Drop trailing "respectively" fragments that survived earlier passes
        trimmed = self._RESPECTIVELY_TRAIL.sub("", working).strip()
        if trimmed != working:
            reasons.append("dropped_respectively")
            working = trimmed

        # Common stock / preferred stock share details frequently trail the label
        if re.search(r"(common|preferred|treasury)\s+stock", raw_label, re.IGNORECASE):
            stock_trimmed = re.sub(
                r"[;,]?\s*(and\s+)?(?:\d[\d,()]*\s+)?shares?.*$",
                "",
                working,
                flags=re.IGNORECASE,
            ).strip()
            if stock_trimmed != working:
                reasons.append("trimmed_share_suffix")
                working = stock_trimmed

        # Remove trailing parenthetical fragments lacking a closing ')'
        if "(" in working and ")" not in working:
            candidate = working.split("(")[0].strip()
            if candidate:
                working = candidate

        # Remove unfinished parenthetical suffixes like "(amortized cost of"
        parent_trimmed = self._PAREN_START.sub("", working).strip()
        if parent_trimmed != working:
            working = parent_trimmed

        # Strip trailing punctuation or connectors left behind
        trailing_cleaned = working.rstrip(";,")
        if trailing_cleaned != working:
            reasons.append("stripped_trailing_punctuation")
            working = trailing_cleaned

        while True:
            new_working = re.sub(r"(?:[,;]\s*|\s+)(?:and|or|of|in|the)$", "", working, flags=re.IGNORECASE).strip()
            if new_working == working:
                break
            reasons.append("trimmed_dangling_connector")
            working = new_working

        return working, reasons

    def _detect_issues(
        self,
        statement: StatementType,
        label: str,
        raw_label: str,
        values: Sequence[Optional[float]],
    ) -> List[str]:
        reasons: List[str] = []
        lowered = label.lower()
        raw_lower = (raw_label or "").lower()

        if not label:
            reasons.append("missing_label")
            return reasons

        if any(token in lowered for token in self._FOOTNOTE_TOKENS):
            reasons.append("footnote_fragment")

        if re.search(r"\d", label):
            reasons.append("contains_digits")

        if label.endswith(tuple(":,;")):
            reasons.append("dangling_punctuation")

        if "(" in label and ")" not in label:
            reasons.append("dangling_parenthesis")

        if re.fullmatch(r"[A-Za-z\s]+", label) is None and not re.search(r"[A-Za-z]", label):
            reasons.append("non_text_label")

        if label.lower() in {"respectively", "issued", "none issued"}:
            reasons.append("standalone_fragment")

        if self._NUMERIC_FOOTNOTE.search(raw_label or "") and all(val is None for val in values):
            reasons.append("likely_footnote_row")

        if (
            statement == StatementType.BALANCE
            and self._SHARE_KEYWORDS.search(raw_lower)
            and not re.search(r"(common|preferred|treasury)\s+stock", lowered)
        ):
            reasons.append("share_details_without_security")

        return reasons

    def _canonicalize(self, label: str, raw_label: str) -> str:
        lowered = label.lower()
        raw_lower = (raw_label or "").lower()
        if "net income" in lowered and "per common" in lowered:
            return "Net income attributable to Amkor per common share"
        if lowered == "share" and "net income" in raw_lower and "per common" in raw_lower:
            return "Net income attributable to Amkor per common share"
        if lowered in {"amounts", "amount", "amounts:"}:
            return ""
        if lowered.startswith("adjustments to reconcile net income to net cash provided by operating"):
            return "Adjustments to reconcile net income to net cash provided by operating activities"
        if lowered.startswith("income (loss) before") and "unconsolidated" in lowered:
            return "Income before equity in earnings of unconsolidated affiliate"
        if lowered.startswith("income before equity in earnings") and "unconsolidated" in lowered:
            return "Income before equity in earnings of unconsolidated affiliate"
        if lowered.startswith("income (loss) before taxes"):
            return "Income before income taxes"
        if lowered.startswith("income before taxes"):
            return "Income before income taxes"
        if lowered == "income before income taxes and equity in earnings of unconsolidated affiliate":
            return "Income before income taxes"
        if lowered == "income before equity in earnings of unconsolidated":
            return "Income before equity in earnings of unconsolidated affiliate"
        if lowered == "total other expense, net":
            return "Total other (income) expense, net"
        if lowered == "total other income (expense), net":
            return "Total other (income) expense, net"
        if lowered == "net income (loss) attributable to non-controlling interests":
            return "Net income attributable to non-controlling interests"
        if lowered == "net income (loss)":
            return "Net income"
        if lowered == "net income":
            return "Net income"
        if "net increase (decrease) in cash and cash equivalents" in lowered and "restricted" not in lowered:
            return "Net increase (decrease) in cash, cash equivalents and restricted cash"
        if lowered == "proceeds from long-term debt":
            return "Proceeds from issuance of long-term debt"
        if lowered == "proceeds from the issuance of stock through share-based compensation plans":
            return "Proceeds from issuance of stock through share-based compensation plans"
        if lowered == "proceeds from issuance of stock through share-based compensation plans":
            return "Proceeds from issuance of stock through share-based compensation plans"
        if lowered == "payments for the retirement of debt":
            return "Payments for retirement of debt"
        if lowered == "other, net":
            return "Other, net"
        return label

    @staticmethod
    def _coalesce_reasons(reasons: Iterable[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for reason in reasons:
            if not reason:
                continue
            if reason not in seen:
                ordered.append(reason)
                seen.add(reason)
        return ordered
