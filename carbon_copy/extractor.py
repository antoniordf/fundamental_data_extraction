from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from selector.models import StatementType

from .certification import build_certification
from .constants import (
    SAVED_UNITS_LABEL,
    STATEMENT_TITLE_PATTERNS,
)
from .models import CarbonCopyResult, HeaderInfo, StatementBlock
from .normalization import normalize_cashflow, normalize_income
from .parsing import detect_units, is_non_monetary_label, parse_number
from .utils import assemble_long, reorder_period_columns


try:  # pdfplumber is optional until extraction runs
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    pdfplumber = None


class CarbonCopyError(RuntimeError):
    pass


@dataclass(slots=True)
class ExtractionConfig:
    snap_tolerance: float = 3.0
    text_tolerance: float = 2.0
    intersection_tolerance: float = 2.0


class CarbonCopyExtractor:
    def __init__(self, config: ExtractionConfig | None = None) -> None:
        self.config = config or ExtractionConfig()
        if pdfplumber is None:
            raise CarbonCopyError(
                "pdfplumber is required for carbon copy extraction. Install via 'pip install pdfplumber'."
            )

    def extract(self, pdf_path: Path) -> CarbonCopyResult:
        if not pdf_path.exists():
            raise CarbonCopyError(f"Input PDF not found: {pdf_path}")

        with pdfplumber.open(str(pdf_path)) as pdf:
            page_texts = [page.extract_text() or "" for page in pdf.pages]
            statement_pages = self._assign_statement_pages(page_texts)
            if any(not pages for pages in statement_pages.values()):
                raise CarbonCopyError(
                    "Failed to detect all three statements. Ensure the input PDF contains the income statement, "
                    "balance sheet, and cash flow statement pages."
                )

            blocks: Dict[StatementType, StatementBlock] = {}
            long_tables: List[pd.DataFrame] = []

            for statement, pages in statement_pages.items():
                page_metadatas = [page_texts[idx] for idx in pages]
                wide_df, column_metadata = self._build_wide_table(statement, pdf, pages)

                # Use first page metadata for headers
                header_text = page_metadatas[0]
                original_units, scale_to_millions, assumption = detect_units(header_text)
                audited_status = "Unaudited" if "unaudited" in header_text.lower() else "Audited/Unaudited: See filing"
                source_pages = ", ".join(f"p.{idx + 1}" for idx in pages)

                period_columns = wide_df.columns[3:]

                # Scale monetary values to millions
                for column in period_columns:
                    wide_df[column] = [
                        self._scale_value(label, value, scale_to_millions)
                        for label, value in zip(wide_df["Line Item (as printed)"], wide_df[column])
                    ]

                header = HeaderInfo(
                    statement_name=self._derive_statement_name(statement, header_text),
                    original_units=original_units,
                    saved_units=SAVED_UNITS_LABEL,
                    units_assumption=assumption,
                    audited_status=audited_status,
                    source=source_pages,
                )

                block = StatementBlock(statement, header, wide_df, column_metadata)
                blocks[statement] = block
                long_tables.append(
                    assemble_long(statement, header.as_dict(), wide_df, header.source)
                )

        certification = build_certification(blocks)
        long_df = pd.concat(long_tables, ignore_index=True)
        return CarbonCopyResult(pdf_path, blocks, long_df, certification)

    # --- helpers ---
    def _assign_statement_pages(self, page_texts: Sequence[str]) -> Dict[StatementType, List[int]]:
        assignments: Dict[StatementType, List[int]] = {
            StatementType.INCOME: [],
            StatementType.BALANCE: [],
            StatementType.CASH: [],
        }
        current: Optional[StatementType] = None
        for idx, text in enumerate(page_texts):
            hits = [stype for stype, pattern in STATEMENT_TITLE_PATTERNS.items() if pattern.search(text)]
            if hits:
                current = hits[0]
                assignments[current].append(idx)
            elif current is not None:
                assignments[current].append(idx)
        return assignments

    def _build_wide_table(
        self,
        statement: StatementType,
        pdf,
        pages: Sequence[int],
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
        lines: List[Dict[str, object]] = []
        for page_index in pages:
            lines.extend(self._collect_lines(pdf.pages[page_index], page_index))

        period_labels: List[str] | None = None
        parsed_rows: List[Dict[str, object]] = []
        pending_numbers: List[str] | None = None
        waiting_index: int | None = None
        header_buffer: List[Dict[str, object]] = []
        column_centers: List[float] | None = None

        for line in lines:
            text = str(line["text"]).strip()
            if not text:
                continue
            if self._should_skip_line(text):
                continue

            tokens: List[str] = line["tokens"]  # type: ignore[assignment]
            positions: List[float] = line["x0s"]  # type: ignore[assignment]
            centers: List[float] = line.get("centers", positions)  # type: ignore[assignment]
            if not tokens:
                continue

            label_parts: List[str] = []
            label_positions: List[float] = []
            numeric_tokens: List[str] = []
            numeric_positions: List[float] = []
            numeric_originals: List[str] = []
            numeric_x0s: List[float] = []
            encountered_numeric = False

            for token, center, x0 in zip(tokens, centers, positions):
                cleaned = token.strip()
                if not cleaned:
                    continue
                if cleaned in {"$", "US$", "U.S.$"}:
                    continue
                if self._is_numeric_token(cleaned):
                    encountered_numeric = True
                    numeric_tokens.append(self._clean_numeric_token(cleaned))
                    numeric_positions.append(center)
                    numeric_originals.append(token)
                    numeric_x0s.append(x0)
                else:
                    if not encountered_numeric:
                        label_parts.append(cleaned)
                        label_positions.append(x0)
                    else:
                        continue

            if period_labels is None:
                if numeric_tokens:
                    if any(len(token) >= 4 for token in numeric_tokens):
                        date_matches = [
                            match.group(0)
                            for match in re.finditer(
                                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}",
                                text,
                                re.IGNORECASE,
                            )
                        ]
                        if date_matches and len(date_matches) <= len(numeric_tokens):
                            period_labels = date_matches
                        else:
                            period_labels = self._compose_column_labels(
                                header_buffer,
                                numeric_tokens,
                                numeric_positions,
                            )
                        column_centers = list(numeric_positions)
                        pending_numbers = None
                        waiting_index = None
                        header_buffer = []
                    else:
                        header_buffer.append(line)
                    continue
                header_buffer.append(line)
                continue

            if numeric_tokens and "in millions" in text.lower():
                pending_numbers = None
                waiting_index = None
                header_buffer = []
                continue

            if column_centers and numeric_tokens:
                filtered_tokens: List[str] = []
                filtered_positions: List[float] = []
                filtered_originals: List[str] = []
                filtered_x0s: List[float] = []
                for token, center, original, x0 in zip(
                    numeric_tokens,
                    numeric_positions,
                    numeric_originals,
                    numeric_x0s,
                ):
                    distance = min(abs(center - ref) for ref in column_centers)
                    if distance <= 45:
                        filtered_tokens.append(token)
                        filtered_positions.append(center)
                        filtered_originals.append(original)
                        filtered_x0s.append(x0)
                    else:
                        label_parts.append(original.strip())
                        label_positions.append(x0)
                numeric_tokens = filtered_tokens
                numeric_positions = filtered_positions
                numeric_originals = filtered_originals
                numeric_x0s = filtered_x0s

            has_label = bool(label_parts)
            has_numbers = any(token for token in numeric_tokens)

            if has_label and has_numbers:
                label = " ".join(label_parts).strip()
                if not label:
                    continue
                label = self._clean_line_item_label(label)
                x0 = min(label_positions) if label_positions else (positions[0] if positions else 0.0)
                values = self._numeric_tokens_to_values(numeric_tokens, period_labels)
                parsed_rows.append({"label": label, "x0": x0, "values": values})
                pending_numbers = None
                waiting_index = None
                continue

            if has_label and not has_numbers:
                label = " ".join(label_parts).strip()
                if not label:
                    continue
                label = self._clean_line_item_label(label)
                x0 = min(label_positions) if label_positions else (positions[0] if positions else 0.0)
                if pending_numbers is not None:
                    values = self._numeric_tokens_to_values(pending_numbers, period_labels)
                    pending_numbers = None
                    parsed_rows.append({"label": label, "x0": x0, "values": values})
                    waiting_index = None
                else:
                    values = [None] * len(period_labels)
                    parsed_rows.append({"label": label, "x0": x0, "values": values})
                    waiting_index = None if label.endswith(":") else len(parsed_rows) - 1
                continue

            if not has_label and has_numbers:
                if waiting_index is not None:
                    values = self._numeric_tokens_to_values(numeric_tokens, period_labels)
                    parsed_rows[waiting_index]["values"] = values
                    waiting_index = None
                else:
                    pending_numbers = list(numeric_tokens)
                continue

            # neither label nor numbers
            continue

        if period_labels is None:
            raise CarbonCopyError("Unable to detect period columns for statement.")
        if not parsed_rows:
            raise CarbonCopyError("No rows extracted for statement.")

        cleaned_period_labels = [self._clean_period_label(label) for label in period_labels]
        ordered_labels = reorder_period_columns(cleaned_period_labels)
        label_lookup: Dict[str, List[int]] = {}
        for idx, label in enumerate(cleaned_period_labels):
            label_lookup.setdefault(label, []).append(idx)
        ordered_indices: List[int] = []
        for label in ordered_labels:
            ordered_indices.append(label_lookup[label].pop(0))
        period_columns = ordered_labels
        level_centers = self._cluster_levels([float(row["x0"]) for row in parsed_rows])

        data: List[List[object]] = []
        for order_index, row in enumerate(parsed_rows, start=1):
            level = self._map_level(float(row["x0"]), level_centers)
            ordered_values: List[Optional[float]] = []
            for idx in ordered_indices:
                ordered_values.append(row["values"][idx] if idx < len(row["values"]) else None)
            data.append([order_index, level, row["label"], *ordered_values])

        wide = pd.DataFrame(
            data,
            columns=[
                "Order Index",
                "Level",
                "Line Item (as printed)",
                *period_columns,
            ],
        )

        column_metadata = self._build_column_metadata(
            statement,
            period_columns,
            [period_labels[idx] for idx in ordered_indices],
        )

        reset_prefixes = (
            "total ",
            "net ",
            "gross profit",
            "operating income",
            "income before",
            "beginning cash",
            "ending cash",
            "increase (decrease)",
        )
        wide["Level"] = [
            0
            if any(label.lower().startswith(prefix) for prefix in reset_prefixes)
            else level
            for label, level in zip(wide["Line Item (as printed)"], wide["Level"])
        ]

        if statement == StatementType.INCOME:
            for column in wide.columns[3:]:
                wide[column] = [
                    normalize_income(label, value) if not is_non_monetary_label(label) else value
                    for label, value in zip(wide["Line Item (as printed)"], wide[column])
                ]
        elif statement == StatementType.CASH:
            for column in wide.columns[3:]:
                wide[column] = [
                    normalize_cashflow(label, value)
                    for label, value in zip(wide["Line Item (as printed)"], wide[column])
                ]
        return wide, column_metadata

    def _collect_lines(self, page, page_index: int) -> List[Dict[str, object]]:
        words = page.extract_words(use_text_flow=True)
        buckets: Dict[float, List[dict]] = {}
        for word in words:
            key = round(word["top"], 1)
            buckets.setdefault(key, []).append(word)

        lines: List[Dict[str, object]] = []
        for key in sorted(buckets):
            line_words = sorted(buckets[key], key=lambda w: w["x0"])
            tokens = [w["text"] for w in line_words]
            x0s = [float(w["x0"]) for w in line_words]
            centers = [
                (float(w["x0"]) + float(w.get("x1", w["x0"])) ) / 2 for w in line_words
            ]
            text = " ".join(tokens)
            lines.append(
                {
                    "text": text,
                    "tokens": tokens,
                    "x0s": x0s,
                    "centers": centers,
                    "page": page_index,
                }
            )
        return lines

    def _should_skip_line(self, text: str) -> bool:
        lowered = text.lower()
        if not lowered:
            return True
        if "table of contents" in lowered:
            return True
        if "https://" in lowered or "http://" in lowered:
            return True
        if "see accompanying" in lowered:
            return True
        if lowered.startswith("page "):
            return True
        if "ea-" in lowered and ":" in lowered:
            return True
        if lowered.startswith("year ended"):
            return True
        if "consolidated " in lowered and "statements" in lowered:
            return True
        if "electronic arts" in lowered:
            return True
        return False

    _PERIOD_PATTERN = re.compile(r"([A-Za-z]+\s+\d{1,2},\s+\d{4}|\d{4})")

    def _extract_period_labels(self, text: str) -> List[str]:
        matches = [m.group(0).strip() for m in self._PERIOD_PATTERN.finditer(text)]
        if len(matches) >= 2:
            return matches
        return []

    def _is_numeric_token(self, token: str) -> bool:
        candidate = token.replace(",", "")
        candidate = candidate.replace("$", "")
        candidate = candidate.replace("%", "")
        candidate = candidate.replace("€", "")
        candidate = candidate.replace("£", "")
        candidate = candidate.replace("–", "-").replace("—", "-").replace("−", "-")
        candidate = candidate.strip()
        if candidate in {"-", "–", "—", "−"}:
            return True
        return bool(re.search(r"\d", candidate)) or ("(" in candidate and ")" in candidate)

    def _clean_numeric_token(self, token: str) -> str:
        cleaned = token.replace("$", "")
        cleaned = cleaned.replace("€", "").replace("£", "")
        cleaned = cleaned.replace("%", "")
        cleaned = cleaned.replace("\u00a0", "")
        cleaned = cleaned.replace("–", "-").replace("—", "-").replace("−", "-")
        cleaned = cleaned.strip()
        cleaned = re.sub(r"[^0-9(),.-]", "", cleaned)
        return cleaned

    def _numeric_tokens_to_values(
        self,
        tokens: Sequence[str],
        period_labels: Sequence[str],
    ) -> List[Optional[float]]:
        values: List[Optional[float]] = []
        for token in tokens:
            cleaned_token = token.strip()
            if not cleaned_token:
                values.append(None)
                continue
            values.append(parse_number(cleaned_token))
        if len(values) > len(period_labels):
            values = values[-len(period_labels):]
        if len(values) < len(period_labels):
            values.extend([None] * (len(period_labels) - len(values)))
        return values

    def _compose_column_labels(
        self,
        header_lines: List[Dict[str, object]],
        numeric_tokens: Sequence[str],
        numeric_positions: Sequence[float],
    ) -> List[str]:
        centers = list(numeric_positions) if numeric_positions else list(range(len(numeric_tokens)))
        parts: List[List[str]] = [[] for _ in numeric_tokens]

        for line in header_lines:
            tokens: List[str] = line.get("tokens", [])  # type: ignore[assignment]
            line_centers: List[float] = line.get("centers", [])  # type: ignore[assignment]
            if not tokens:
                continue
            if any("unaudited" in token.lower() for token in tokens):
                continue
            groups: List[List[str]] = []
            group_centers: List[List[float]] = []
            current_tokens: List[str] = [tokens[0]]
            current_centers: List[float] = [line_centers[0]]
            gap_threshold = 40.0
            for token, center in zip(tokens[1:], line_centers[1:]):
                if abs(center - current_centers[-1]) > gap_threshold:
                    groups.append(current_tokens)
                    group_centers.append(current_centers)
                    current_tokens = [token]
                    current_centers = [center]
                else:
                    current_tokens.append(token)
                    current_centers.append(center)
            groups.append(current_tokens)
            group_centers.append(current_centers)

            if groups:
                per_group = len(parts) // len(groups) if len(groups) and len(parts) % len(groups) == 0 else None
            else:
                per_group = None

            if per_group:
                for idx, group in enumerate(groups):
                    start = idx * per_group
                    end = min(start + per_group, len(parts))
                    for col in range(start, end):
                        parts[col].extend(group)
            else:
                for token, center in zip(tokens, line_centers):
                    col_idx = self._nearest_column_index(center, centers)
                    parts[col_idx].append(token)

        for token, x0 in zip(numeric_tokens, centers):
            idx = self._nearest_column_index(x0, centers)
            parts[idx].append(token)

        labels = [" ".join(segment).strip() for segment in parts]
        labels = self._merge_period_fragments(labels)
        for idx, label in enumerate(labels):
            if not label:
                labels[idx] = str(numeric_tokens[idx])
        return labels

    def _nearest_column_index(self, x0: float, centers: Sequence[float]) -> int:
        if not centers:
            return 0
        return min(range(len(centers)), key=lambda idx: abs(centers[idx] - x0))

    def _cluster_levels(self, positions: Iterable[float]) -> List[float]:
        unique = sorted(set(round(pos, 1) for pos in positions))
        clusters: List[float] = []
        for pos in unique:
            if not clusters or abs(pos - clusters[-1]) > 3.0:
                clusters.append(pos)
        if not clusters:
            clusters.append(0.0)
        return clusters

    def _map_level(self, position: float, clusters: Sequence[float]) -> int:
        if not clusters:
            return 0
        best_idx = min(range(len(clusters)), key=lambda idx: abs(clusters[idx] - position))
        return min(best_idx, 2)

    def _build_column_metadata(
        self,
        statement: StatementType,
        display_labels: Sequence[str],
        raw_labels: Sequence[str],
    ) -> Dict[str, Dict[str, str]]:
        metadata: Dict[str, Dict[str, str]] = {}
        for display, raw in zip(display_labels, raw_labels):
            metadata[display] = {
                "raw_label": raw,
                "duration": self._infer_duration(statement, display),
                "period_end": self._extract_period_end(display, raw),
            }
        return metadata

    def _infer_duration(self, statement: StatementType, label: str) -> str:
        lowered = label.lower()
        if statement == StatementType.BALANCE:
            return "point"
        if "three months" in lowered:
            return "quarter"
        if any(keyword in lowered for keyword in {"six months", "nine months", "year-to-date", "year to date"}):
            return "ytd"
        if "year ended" in lowered or "twelve months" in lowered:
            return "annual"
        if re.search(r"(19|20)\d{2}", lowered) and "month" not in lowered:
            return "annual"
        if statement == StatementType.CASH:
            return "ytd"
        return "unknown"

    def _extract_period_end(self, display_label: str, raw_label: str) -> str:
        for source in (display_label, raw_label):
            match = re.search(
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}",
                source,
                re.IGNORECASE,
            )
            if match:
                return match.group(0)
        year_match = re.search(r"(19|20)\d{2}", display_label)
        if not year_match:
            year_match = re.search(r"(19|20)\d{2}", raw_label)
        return year_match.group(0) if year_match else ""

    def _clean_period_label(self, label: str) -> str:
        label_norm = " ".join(label.split())
        label_norm = re.sub(r"\(in .*?\)", "", label_norm, flags=re.IGNORECASE).strip()
        month_match = re.search(
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}",
            label_norm,
            re.IGNORECASE,
        )
        if month_match:
            date_text = month_match.group(0)
            prefix = label_norm[: month_match.start()].strip()
            if prefix:
                return f"{prefix} {date_text}".strip()
            return date_text
        year_match = re.search(r"(19|20)\d{2}", label_norm)
        if year_match:
            year = year_match.group(0)
            prefix = label_norm[: year_match.start()].strip()
            if prefix and any(
                word.lower() in {"three", "six", "nine", "twelve", "months", "ended", "year"}
                for word in prefix.split()
            ):
                return f"{prefix} {year}".strip()
            return year
        return label_norm

    def _merge_period_fragments(self, labels: Sequence[str]) -> List[str]:
        merged: List[str] = []
        skip_next = False
        for idx, label in enumerate(labels):
            if skip_next:
                skip_next = False
                continue
            current = label.strip()
            if idx + 1 < len(labels):
                next_label = labels[idx + 1].strip()
                if re.fullmatch(r"(19|20)\d{2}", next_label):
                    if current.endswith(",") or any(
                        term in current.lower()
                        for term in [
                            "ended",
                            "months",
                            "march",
                            "june",
                            "september",
                            "december",
                            "april",
                            "may",
                            "july",
                            "august",
                            "october",
                            "november",
                            "january",
                            "february",
                        ]
                    ):
                        merged.append(f"{current} {next_label}".strip())
                        skip_next = True
                        continue
            merged.append(current)
        return merged

    def _clean_line_item_label(self, label: str) -> str:
        lowered = label.lower()
        if "preferred stock" in lowered:
            return "Preferred stock"
        if "common stock" in lowered:
            return "Common stock"
        return label

    def _scale_value(self, label: str, value: Optional[float], scale: float) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return None
        if is_non_monetary_label(label):
            return value
        return value * scale

    def _derive_statement_name(self, statement: StatementType, page_text: str) -> str:
        pattern = STATEMENT_TITLE_PATTERNS[statement]
        for line in (page_text or "").splitlines():
            if pattern.search(line):
                return line.strip()
        defaults = {
            StatementType.INCOME: "CONSOLIDATED STATEMENTS OF OPERATIONS",
            StatementType.BALANCE: "CONSOLIDATED BALANCE SHEETS",
            StatementType.CASH: "CONSOLIDATED STATEMENTS OF CASH FLOWS",
        }
        return defaults[statement]
