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
from .aliases import load_aliases, normalize_label
from .label_normalizer import LabelNormalizer
from .models import CarbonCopyResult, HeaderInfo, StatementBlock, SuspectLabel
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
        self.alias_map = load_aliases(None)
        self.label_normalizer = LabelNormalizer()

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
                wide_df, column_metadata, suspect_labels = self._build_wide_table(statement, pdf, pages)

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

                block = StatementBlock(statement, header, wide_df, column_metadata, suspect_labels)
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
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]], Dict[int, SuspectLabel]]:
        lines: List[Dict[str, object]] = []
        for page_index in pages:
            lines.extend(self._collect_lines(pdf.pages[page_index], page_index))

        period_labels: List[str] | None = None
        parsed_rows: List[Dict[str, object]] = []
        pending_numbers: List[str] | None = None
        waiting_index: int | None = None
        header_buffer: List[Dict[str, object]] = []
        column_centers: List[float] | None = None
        period_center_map: Dict[str, float] = {}
        in_header = True

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
            label_centers: List[float] = []
            trailing_parts: List[str] = []
            trailing_positions: List[float] = []
            trailing_centers: List[float] = []
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
                        label_centers.append(center)
                    else:
                        if re.search(r"[A-Za-z]", cleaned):
                            trailing_parts.append(cleaned)
                            trailing_positions.append(x0)
                            trailing_centers.append(center)

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
                        centers_source = [float(pos) for pos in numeric_positions]
                        assigned_centers = self._assign_label_centers(centers_source, len(period_labels))
                        column_centers = assigned_centers
                        for label, center in zip(period_labels, assigned_centers):
                            period_center_map[label] = center
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
                    # DEBUG
                    # if 'Cash and cash equivalents' in text:
                    #     print('TOKEN', original, 'center', center, 'dist', distance, 'centers', column_centers)
                    if distance <= 45:
                        filtered_tokens.append(token)
                        filtered_positions.append(center)
                        filtered_originals.append(original)
                        filtered_x0s.append(x0)
                    else:
                        if not self._is_footnote_token(original):
                            label_parts.append(original.strip())
                            label_positions.append(x0)
                            label_centers.append(center)
                numeric_tokens = filtered_tokens
                numeric_positions = filtered_positions
                numeric_originals = filtered_originals
                numeric_x0s = filtered_x0s

            if trailing_parts:
                label_parts.extend(trailing_parts)
                label_positions.extend(trailing_positions)
                label_centers.extend(trailing_centers)

            if label_parts:
                filtered = []
                for text, pos, cen in zip(label_parts, label_positions, label_centers):
                    stripped = text.strip()
                    core = stripped.strip(",;:")
                    if not core:
                        continue
                    if re.fullmatch(r"[\d\$,().-]+", core):
                        continue
                    filtered.append((stripped, pos, cen))
                if filtered:
                    label_parts = [item[0] for item in filtered]
                    label_positions = [item[1] for item in filtered]
                    label_centers = [item[2] for item in filtered]
                    connectors = {"and", "of", "the"}
                    while len(label_parts) > 1 and label_parts[0].lower() in connectors:
                        label_parts.pop(0)
                        label_positions.pop(0)
                        label_centers.pop(0)

            if in_header:
                period_matches = self._extract_period_labels(text)
                if period_matches:
                    centers_source: List[float]
                    if numeric_positions:
                        centers_source = [float(pos) for pos in numeric_positions]
                    elif label_centers:
                        centers_source = [float(pos) for pos in label_centers]
                    else:
                        centers_source = [float(pos) for pos in positions]
                    assigned_centers = self._assign_label_centers(centers_source, len(period_matches))
                    for label, center in zip(period_matches, assigned_centers):
                        if center is None:
                            continue
                        period_center_map[label] = center
                    if period_center_map:
                        ordered = sorted(period_center_map.items(), key=lambda item: item[1])
                        period_labels = [label for label, _ in ordered]
                        column_centers = [center for _, center in ordered]
                    header_buffer.append(line)
                    pending_numbers = None
                    waiting_index = None
                    continue

            has_label = bool(label_parts)
            has_numbers = any(token for token in numeric_tokens)

            if has_label and has_numbers:
                raw_label_text = line["text"]
                merged_label = " ".join(label_parts).strip() or raw_label_text
                if waiting_index is not None and re.search(r"(share|shares|issued|outstanding|respectively)", raw_label_text, re.IGNORECASE):
                    values = self._numeric_tokens_to_values(numeric_tokens, period_labels)
                    base = parsed_rows[waiting_index]["label"]
                    combined = self._clean_line_item_label(f"{base} {merged_label}")
                    parsed_rows[waiting_index]["label"] = combined or base
                    parsed_rows[waiting_index]["values"] = values
                    existing_raw = parsed_rows[waiting_index].get("raw_label", "")
                    parsed_rows[waiting_index]["raw_label"] = (f"{existing_raw} {raw_label_text}").strip()
                    waiting_index = None
                    pending_numbers = None
                    continue
                in_header = False
                label = self._clean_line_item_label(merged_label)
                if not label:
                    continue
                x0 = min(label_positions) if label_positions else (positions[0] if positions else 0.0)
                values = self._numeric_tokens_to_values(numeric_tokens, period_labels)
                parsed_rows.append({
                    "label": label,
                    "x0": x0,
                    "values": values,
                    "raw_label": raw_label_text,
                })
                pending_numbers = None
                waiting_index = None
                continue

            if has_label and not has_numbers:
                label = " ".join(label_parts).strip()
                label = self._clean_line_item_label(label)
                if not label:
                    continue
                x0 = min(label_positions) if label_positions else (positions[0] if positions else 0.0)
                if label.endswith(":") and parsed_rows:
                    last_row = parsed_rows[-1]
                    if all(val is None for val in last_row["values"]):
                        combined = self._clean_line_item_label(f"{last_row['label']} {label}")
                        if combined:
                            last_row["label"] = combined
                            existing_raw = last_row.get("raw_label", "")
                            last_row["raw_label"] = (f"{existing_raw} {line['text']}").strip()
                            continue
                should_append = (
                    parsed_rows
                    and pending_numbers is None
                    and waiting_index is None
                    and not label.endswith(":")
                )
                if should_append and re.fullmatch(r"[A-Z&\s]+", label):
                    should_append = False
                if should_append:
                    last_row = parsed_rows[-1]
                    has_numbers_in_last = any(val is not None for val in last_row["values"])
                    append_to_data_row = (
                        has_numbers_in_last
                        and re.search(r"(per common share|per common|per share|shares?|amounts?)$", label, re.IGNORECASE)
                    )
                    if has_numbers_in_last and not append_to_data_row:
                        should_append = False
                    elif has_numbers_in_last and append_to_data_row:
                        last_row["label"] = self._clean_line_item_label(f"{last_row['label']} {label}")
                        existing_raw = last_row.get("raw_label", "")
                        last_row["raw_label"] = (f"{existing_raw} {line['text']}").strip()
                        continue
                    else:
                        if re.search(r"accompanying notes", label, re.IGNORECASE):
                            continue
                        last_row["label"] = self._clean_line_item_label(f"{last_row['label']} {label}")
                        existing_raw = last_row.get("raw_label", "")
                        last_row["raw_label"] = (f"{existing_raw} {line['text']}").strip()
                        continue
                if pending_numbers is not None:
                    values = self._numeric_tokens_to_values(pending_numbers, period_labels)
                    pending_numbers = None
                    parsed_rows.append({"label": label, "x0": x0, "values": values, "raw_label": line["text"]})
                    waiting_index = None
                else:
                    values = [None] * len(period_labels)
                    parsed_rows.append({"label": label, "x0": x0, "values": values, "raw_label": line["text"]})
                    waiting_index = None if label.endswith(":") else len(parsed_rows) - 1
                continue

            if not has_label and has_numbers:
                if waiting_index is not None:
                    values = self._numeric_tokens_to_values(numeric_tokens, period_labels)
                    parsed_rows[waiting_index]["values"] = values
                    existing_raw = parsed_rows[waiting_index].get("raw_label", "")
                    parsed_rows[waiting_index]["raw_label"] = (f"{existing_raw} {line['text']}").strip()
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

        def _duration_label(idx: int, cleaned: str, raw: str, duration: str) -> str:
            date_text = self._extract_period_end(cleaned, raw) or self._extract_period_end(raw, raw)
            label_text = duration
            if date_text:
                label_text = f"{duration} {date_text}"
            return " ".join(label_text.split())

        adjusted_labels: List[str] = []
        total_cols = len(cleaned_period_labels)
        for idx, (cleaned, raw) in enumerate(zip(cleaned_period_labels, period_labels)):
            lower = raw.lower()
            normalized = cleaned
            if "three months" in lower and "six months" in lower and total_cols >= 4:
                normalized = _duration_label(idx, cleaned, raw, "Three Months Ended" if idx < total_cols / 2 else "Six Months Ended")
            elif "three months" in lower and "nine months" in lower and total_cols >= 4:
                normalized = _duration_label(idx, cleaned, raw, "Three Months Ended" if idx < total_cols / 2 else "Nine Months Ended")
            elif "three months" in lower and "twelve months" in lower and total_cols >= 4:
                normalized = _duration_label(idx, cleaned, raw, "Three Months Ended" if idx < total_cols / 2 else "Twelve Months Ended")
            elif "three months" in lower:
                normalized = _duration_label(idx, cleaned, raw, "Three Months Ended")
            elif "six months" in lower:
                normalized = _duration_label(idx, cleaned, raw, "Six Months Ended")
            elif "nine months" in lower:
                normalized = _duration_label(idx, cleaned, raw, "Nine Months Ended")
            elif "twelve months" in lower:
                normalized = _duration_label(idx, cleaned, raw, "Twelve Months Ended")
            elif "year ended" in lower or "years ended" in lower:
                normalized = _duration_label(idx, cleaned, raw, "Year Ended")
            elif "quarter ended" in lower or "quarters ended" in lower:
                normalized = _duration_label(idx, cleaned, raw, "Quarter Ended")
            adjusted_labels.append(normalized)

        cleaned_period_labels = adjusted_labels
        ordered_labels = reorder_period_columns(cleaned_period_labels)
        label_lookup: Dict[str, List[int]] = {}
        for idx, label in enumerate(cleaned_period_labels):
            label_lookup.setdefault(label, []).append(idx)
        ordered_indices: List[int] = []
        for label in ordered_labels:
            ordered_indices.append(label_lookup[label].pop(0))
        # DEBUG
        # print('DEBUG columns', cleaned_period_labels, ordered_labels, ordered_indices)
        period_columns = ordered_labels
        level_centers = self._cluster_levels([float(row["x0"]) for row in parsed_rows])

        data: List[List[object]] = []
        raw_labels: List[str] = []
        for order_index, row in enumerate(parsed_rows, start=1):
            level = self._map_level(float(row["x0"]), level_centers)
            ordered_values: List[Optional[float]] = []
            for idx in ordered_indices:
                ordered_values.append(row["values"][idx] if idx < len(row["values"]) else None)
            data.append([order_index, level, row["label"], *ordered_values])
            raw_labels.append(row.get("raw_label", ""))

        wide = pd.DataFrame(
            data,
            columns=[
                "Order Index",
                "Level",
                "Line Item (as printed)",
                *period_columns,
            ],
        )
        wide["_raw_label"] = raw_labels

        value_columns = list(wide.columns[3:])
        if value_columns:
            mask_all_na = wide[value_columns].isna().all(axis=1)
            footnote_regex = re.compile(
                r"(?:accompanying notes|condensed consolidated statements|consolidated statements|\(in thousands|\(in millions|shares issued|issued(?:[,;]|$)|respectively|the accompanying notes|shares used in computing per common share amounts|shares used\s+-|net income attributable to amkor per common share)",
                re.IGNORECASE,
            )
            drop_mask = mask_all_na & wide["Line Item (as printed)"].str.contains(footnote_regex, na=False)
            if drop_mask.any():
                wide = wide[~drop_mask].reset_index(drop=True)

        post_drop_labels = {
            "net income attributable to amkor",
            "amounts",
            "shares used in computing per common share",
            "shares used",
        }

        while True:
            suspects = self._apply_label_normalization(statement, wide)
            if value_columns:
                mask_all_na = wide[value_columns].isna().all(axis=1)
                drop_mask_post = mask_all_na & wide["Line Item (as printed)"].str.lower().isin(post_drop_labels)
            else:
                drop_mask_post = wide["Line Item (as printed)"].str.lower().isin(post_drop_labels)
            if not drop_mask_post.any():
                break
            wide = wide[~drop_mask_post].reset_index(drop=True)

        if "_raw_label" in wide.columns:
            wide.drop(columns=["_raw_label"], inplace=True)

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
        return wide, column_metadata, suspects

    def _apply_label_normalization(
        self,
        statement: StatementType,
        frame: pd.DataFrame,
    ) -> Dict[int, SuspectLabel]:
        suspects: Dict[int, SuspectLabel] = {}
        value_columns = [
            column
            for column in frame.columns
            if column not in {"Order Index", "Level", "Line Item (as printed)", "_raw_label"}
        ]
        seen_labels: Dict[str, List[int]] = {}

        def _dedupe(reasons: List[str]) -> List[str]:
            seen: set[str] = set()
            ordered: List[str] = []
            for reason in reasons:
                if not reason:
                    continue
                if reason not in seen:
                    ordered.append(reason)
                    seen.add(reason)
            return ordered

        drop_indices: List[int] = []

        section_stack: Dict[int, str] = {}

        for idx in range(len(frame)):
            row = frame.iloc[idx]
            raw_label = str(row.get("_raw_label", "") or "")
            current_label = str(row.get("Line Item (as printed)", "") or "")
            values = [row[col] for col in value_columns]
            level = int(row.get("Level", 0))

            raw_lower = raw_label.lower()
            has_values = any(val is not None for val in values)
            if not has_values:
                if "per common" in raw_lower and "net income" in raw_lower:
                    drop_indices.append(idx)
                    continue
                if raw_lower.strip() in {"amounts", "amounts:"}:
                    drop_indices.append(idx)
                    continue
                if "shares used in computing per common share" in raw_lower:
                    drop_indices.append(idx)
                    continue
                if raw_lower.strip() in {"activities", "activity", "affiliate", "unconsolidated affiliate"}:
                    drop_indices.append(idx)
                    continue
            if "preferred stock" in raw_lower and "common stock" in raw_lower and "shares outstanding" in raw_lower:
                drop_indices.append(idx)
                continue
            if re.search(r"per common share:", raw_lower) or re.search(
                r"shares used in computing per common share amounts", raw_lower
            ):
                drop_indices.append(idx)
                continue

            normalization = self.label_normalizer.normalize(statement, current_label, raw_label, values)
            normalized = normalize_label(statement, normalization.label, self.alias_map)

            reasons = list(normalization.reasons)
            if normalized and normalized != normalization.label:
                reasons.append("alias_applied")

            final_label = (normalized or normalization.label or raw_label).strip()
            if "per common share" in raw_lower and "net income" in raw_lower and "per share" not in final_label.lower():
                final_label = "Net income attributable to Amkor per common share"
            if not final_label.strip():
                if has_values:
                    final_label = current_label or raw_label or f"Line {idx + 1}"
                else:
                    drop_indices.append(idx)
                    continue

            parent_label = section_stack.get(level - 1, "") if level > 0 else ""
            prev_label = ""
            if idx > 0:
                prev_label = str(frame.at[idx - 1, "Line Item (as printed)"])

            adjusted_label = final_label
            lowered_prev = prev_label.lower()
            numeric_values = [
                float(value)
                for value in values
                if isinstance(value, (int, float))
            ]
            max_magnitude = max((abs(val) for val in numeric_values), default=None)
            lowered_label = final_label.lower()
            if lowered_label == "basic":
                if max_magnitude is not None and max_magnitude <= 10:
                    adjusted_label = "Earnings per share - Basic"
                else:
                    adjusted_label = "Shares used - Basic"
            elif lowered_label == "diluted":
                if max_magnitude is not None and max_magnitude <= 10:
                    adjusted_label = "Earnings per share - Diluted"
                else:
                    adjusted_label = "Shares used - Diluted"
            elif lowered_label.startswith("shares used -"):
                # already canonical
                pass
            elif "shares used" in lowered_label and "per common share" in lowered_label:
                if "diluted" in lowered_label:
                    adjusted_label = "Shares used - Diluted"
                elif "basic" in lowered_label:
                    adjusted_label = "Shares used - Basic"
            elif "per common share" in lowered_prev and lowered_label in {"basic", "diluted"}:
                if lowered_label == "basic":
                    adjusted_label = "Earnings per share - Basic"
                else:
                    adjusted_label = "Earnings per share - Diluted"

            frame.at[idx, "Line Item (as printed)"] = adjusted_label
            final_label = adjusted_label

            parent_lower = parent_label.lower()
            if final_label.lower() == "restricted cash":
                if "current" in parent_lower:
                    final_label = "Restricted cash (current)"
                elif level == 0:
                    final_label = "Restricted cash (non-current)"
                frame.at[idx, "Line Item (as printed)"] = final_label

            if final_label.lower() == "long-term debt, related party":
                final_label = "Long-term debt (related party)"
                frame.at[idx, "Line Item (as printed)"] = final_label

            if final_label.lower() == "shares outstanding":
                reasons = []

            if final_label.lower() == "share":
                final_label = "Net income attributable to Amkor per common share"
                frame.at[idx, "Line Item (as printed)"] = final_label

            key = final_label.lower()
            if key:
                seen_labels.setdefault(key, []).append(idx)

            security_labels = {"common stock", "preferred stock", "treasury stock"}
            if final_label.lower() in security_labels:
                reasons = [
                    reason
                    for reason in reasons
                    if reason
                    not in {
                        "footnote_fragment",
                        "share_details_without_security",
                        "standalone_fragment",
                        "missing_label",
                        "empty_after_normalization",
                    }
                ]

            if reasons:
                deduped = _dedupe(reasons)
                core_reasons = [reason for reason in deduped if reason != "alias_applied"]
                if core_reasons:
                    suspects[idx] = SuspectLabel(
                        row_index=idx,
                        original=raw_label or current_label,
                        cleaned=final_label,
                        reasons=core_reasons,
                    )

            # update section stack hierarchy
            keys_to_delete = [lvl for lvl in list(section_stack.keys()) if lvl >= level]
            for lvl in keys_to_delete:
                section_stack.pop(lvl, None)
            section_stack[level] = final_label

        for indices in seen_labels.values():
            if len(indices) <= 1:
                continue
            for idx in indices:
                entry = suspects.get(idx)
                if entry:
                    if "duplicate_after_normalization" not in entry.reasons:
                        entry.reasons.append("duplicate_after_normalization")
                else:
                    row = frame.iloc[idx]
                    raw_label = str(row.get("_raw_label", "") or "")
                    suspects[idx] = SuspectLabel(
                        row_index=idx,
                        original=raw_label or str(row.get("Line Item (as printed)", "")),
                        cleaned=str(row.get("Line Item (as printed)", "")),
                        reasons=["duplicate_after_normalization"],
                    )

        if drop_indices:
            frame.drop(index=drop_indices, inplace=True)
            frame.reset_index(drop=True, inplace=True)
            # Recurse to rebuild suspects on the trimmed frame
            return self._apply_label_normalization(statement, frame)

        if value_columns:
            mask_all_na = frame[value_columns].isna().all(axis=1)
            duplicate_labels = frame["Line Item (as printed)"].str.lower().duplicated(keep=False)
            redundant = frame.index[mask_all_na & duplicate_labels]
            if len(redundant) > 0:
                frame.drop(index=redundant, inplace=True)
                frame.reset_index(drop=True, inplace=True)
                return self._apply_label_normalization(statement, frame)

        post_drop_labels = {
            "shares used in computing per common share",
            "amounts",
            "amounts:",
        }
        mask_post = frame["Line Item (as printed)"].str.lower().isin(post_drop_labels)
        if mask_post.any():
            frame.drop(index=frame.index[mask_post], inplace=True)
            frame.reset_index(drop=True, inplace=True)
            return self._apply_label_normalization(statement, frame)

        label_counts = frame["Line Item (as printed)"].str.lower().value_counts()
        for idx in list(suspects.keys()):
            cleaned_lower = suspects[idx].cleaned.lower()
            if label_counts.get(cleaned_lower, 0) <= 1:
                reasons = [reason for reason in suspects[idx].reasons if reason != "duplicate_after_normalization"]
                if reasons:
                    suspects[idx].reasons = reasons
                else:
                    del suspects[idx]

        return suspects

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
        if re.match(r"\d{1,2}/\d{1,2}/\d{4}", text):
            return True
        if "table of contents" in lowered:
            return True
        if "https://" in lowered or "http://" in lowered:
            return True
        if "see accompanying" in lowered:
            return True
        if lowered.startswith("page "):
            return True
        if " document" in lowered and re.search(r"\d{1,2}:\d{2}", lowered):
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
        return matches

    def _is_footnote_token(self, token: str) -> bool:
        stripped = token.strip()
        if not stripped:
            return True
        stripped = stripped.strip("()[]{}")
        stripped = stripped.replace("–", "-").replace("—", "-").replace("−", "-")
        stripped = stripped.strip("-.,")
        if not stripped:
            return True
        if stripped.isdigit():
            return True
        if len(stripped) == 1 and stripped.isalpha():
            return True
        if re.fullmatch(r"\d+[a-zA-Z]?", stripped):
            return True
        return False

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

    def _assign_label_centers(self, positions: Sequence[float], count: int) -> List[float]:
        if count <= 0:
            return []
        if not positions:
            return [float(index) for index in range(count)]
        numeric_positions = [float(pos) for pos in positions]
        total = len(numeric_positions)
        if total == 1:
            return [numeric_positions[0] for _ in range(count)]
        results: List[float] = []
        for idx in range(count):
            start = int(round(idx * total / count))
            end = int(round((idx + 1) * total / count))
            if start >= total:
                start = total - 1
            if end <= start:
                end = min(total, start + 1)
            chunk = numeric_positions[start:end]
            if not chunk:
                chunk = [numeric_positions[min(start, total - 1)]]
            results.append(sum(chunk) / len(chunk))
        return results

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
            raw_prefix = label_norm[: month_match.start()].strip()
            if raw_prefix:
                duration_patterns = [
                    re.compile(r"(?:for\s+the\s+)?((?:three|six|nine|twelve)\s+months?\s+ended)", re.IGNORECASE),
                    re.compile(r"(?:for\s+the\s+)?((?:one|two)\s+months?\s+ended)", re.IGNORECASE),
                    re.compile(r"(?:for\s+the\s+)?((?:quarter|quarters)\s+ended)", re.IGNORECASE),
                    re.compile(r"(?:for\s+the\s+)?((?:year|years)\s+ended)", re.IGNORECASE),
                ]
                candidates: List[tuple[int, str]] = []
                for pattern in duration_patterns:
                    for match in pattern.finditer(raw_prefix):
                        qualifier = " ".join(match.group(1).split())
                        candidates.append((match.start(), qualifier))
                if candidates:
                    _, qualifier = max(candidates, key=lambda item: item[0])
                    return f"{qualifier} {date_text}".strip()
            return date_text
        year_match = re.search(r"(19|20)\d{2}", label_norm)
        if year_match:
            year = year_match.group(0)
            raw_prefix = label_norm[: year_match.start()].strip()
            if raw_prefix:
                duration_patterns = [
                    re.compile(r"(?:for\s+the\s+)?((?:three|six|nine|twelve)\s+months?\s+ended)", re.IGNORECASE),
                    re.compile(r"(?:for\s+the\s+)?((?:year|years)\s+ended)", re.IGNORECASE),
                    re.compile(r"(?:for\s+the\s+)?((?:quarter|quarters)\s+ended)", re.IGNORECASE),
                ]
                candidates: List[tuple[int, str]] = []
                for pattern in duration_patterns:
                    for match in pattern.finditer(raw_prefix):
                        qualifier = " ".join(match.group(1).split())
                        candidates.append((match.start(), qualifier))
                if candidates:
                    _, qualifier = max(candidates, key=lambda item: item[0])
                    return f"{qualifier} {year}".strip()
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
        parts = label.split()
        while parts and self._is_footnote_token(parts[-1]):
            parts.pop()
        label = " ".join(parts).strip()
        label = re.sub(r"\s+", " ", label)
        raw_lower = label.lower()
        # Remove trailing footnote phrases
        for pattern in [
            r",?\s*respectively\)?$",
            r",?\s*respectively\s*\([^)]*\)$",
            r",?\s*in thousands\)?$",
            r",?\s*in millions\)?$",
        ]:
            label = re.sub(pattern, "", label, flags=re.IGNORECASE).strip()
        label = label.rstrip(":")
        if "total assets" in raw_lower and "liabilities and equity" in raw_lower:
            label = re.sub(r"\bLIABILITIES AND EQUITY\b", "", label, flags=re.IGNORECASE).strip()
        label = re.sub(r"\bCommitments and contingencies.*$", "", label, flags=re.IGNORECASE).strip(" ,;")
        if re.search(r"(common|preferred|treasury)\s+stock", label, re.IGNORECASE):
            label = re.sub(r"[;,]?\s*(?:and\s+)?shares?.*$", "", label, flags=re.IGNORECASE).strip()
        label = re.sub(r",?\s*shares issued.*$", "", label, flags=re.IGNORECASE).strip()
        if "shares outstanding" in raw_lower:
            return "Shares outstanding"
        label = re.sub(r"\b(and|or)\s+(and|or)\b", r"\1", label, flags=re.IGNORECASE)
        # Remove dangling connectors
        while True:
            new_label = re.sub(r"(?:[,;]\s*|\s+)(?:and|or|of|in|the)$", "", label, flags=re.IGNORECASE).strip()
            if new_label == label:
                break
            label = new_label
        lowered = label.lower()
        if not label or lowered in {"(", ")", "-"}:
            return ""
        if "accompanying notes" in lowered:
            return ""
        if lowered.startswith("except per share"):
            return ""
        if lowered.startswith("(in ") or lowered.startswith("in "):
            return ""
        if "net income attributable to amk net income attributable to amkor per common share" in lowered:
            return "Net income attributable to Amkor per common share"
        if "net income (loss) attributable to amk" in lowered and "per common share" in lowered:
            return "Net income attributable to Amkor per common share"
        if lowered.startswith("net income attributable to amk"):
            return "Net income attributable to Amkor"
        if "net income attributable to non-controlling interests" in lowered:
            return "Net income attributable to non-controlling interests"
        if "net loss (income) attributable to non-controlling interests" in lowered:
            return "Net income attributable to non-controlling interests"
        if "proceeds from long-term debt" in lowered and "issuance" in lowered:
            return "Proceeds from issuance of long-term debt"
        if lowered in {"retained earnings", "retained earnings (accumulated deficit)"}:
            return "Retained earnings (accumulated deficit)"
        if "restricted cash" in lowered and "short-term investments" in lowered:
            return "Restricted cash"
        if lowered == "shares outstanding":
            return "Shares outstanding"
        label = re.sub(r"\bamk\b", "Amkor", label, flags=re.IGNORECASE)
        lowered = label.lower()
        if "shares used in computing per common share amounts" in lowered:
            if "diluted" in lowered and "basic" not in lowered:
                return "Diluted"
            if "basic" in lowered and "diluted" not in lowered:
                return "Basic"
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
                return self._canonical_statement_title(statement, line.strip())
        defaults = {
            StatementType.INCOME: "CONSOLIDATED STATEMENTS OF OPERATIONS",
            StatementType.BALANCE: "CONSOLIDATED BALANCE SHEETS",
            StatementType.CASH: "CONSOLIDATED STATEMENTS OF CASH FLOWS",
        }
        return defaults[statement]

    def _canonical_statement_title(self, statement: StatementType, title: str) -> str:
        lowered = title.lower()
        if statement == StatementType.CASH or ("cash" in lowered and "flow" in lowered):
            return "CONSOLIDATED STATEMENTS OF CASH FLOWS"
        if statement == StatementType.BALANCE or "balance" in lowered:
            return "CONSOLIDATED BALANCE SHEETS"
        if statement == StatementType.INCOME:
            if "operations" in lowered:
                return "CONSOLIDATED STATEMENTS OF OPERATIONS"
            if "comprehensive" in lowered and "income" in lowered:
                return "CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME"
            if "income" in lowered:
                return "CONSOLIDATED STATEMENTS OF INCOME"
        return title
