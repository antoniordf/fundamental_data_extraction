"""High-level financial statement detection logic."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .constants import (
    ANCHOR_CONFIG,
    DEFAULT_MAX_SCAN,
    DEFAULT_SEARCH_WINDOW,
    DEFAULT_TOC_DELTA,
    EXCLUDE,
    INDEX_PAGE,
    TABLE_OF_CONTENTS,
)
from .exceptions import OutputError
from .extractor import PDFTextExtractor
from .models import StatementSelectionResult, StatementType
from .utils import (
    dedup_order,
    has_anchors,
    include_continuations,
    near_top,
    numeric_density,
    write_subset,
)


class FinancialStatementSelector:
    """Coordinates extraction and heuristic selection of statement pages."""

    def __init__(
        self,
        extractor: PDFTextExtractor,
        search_window: int = DEFAULT_SEARCH_WINDOW,
        toc_delta: int = DEFAULT_TOC_DELTA,
    ) -> None:
        self.extractor = extractor
        self.search_window = search_window
        self.toc_delta = toc_delta

    def run(self, input_pdf: Path, output_pdf: Path | None = None) -> StatementSelectionResult:
        extraction = self.extractor.extract(input_pdf)
        texts = extraction.texts
        total_pages = len(texts)

        main_hint = self._find_main_fs_hint(texts)
        inner_idx = self._locate_inner_index(texts, main_hint)
        printed_map = self._build_printed_map(texts, inner_idx, main_hint)
        mapped = self._map_to_actual(texts, printed_map)
        with_cont = self._expand_with_continuations(texts, mapped)
        with_cont = self._ensure_fallback_hits(texts, with_cont)

        selected_indices = sorted(
            set(idx for idxs in with_cont.values() for idx in idxs)
        )

        if not selected_indices:
            selected_indices = self._cluster_fallback(texts)

        output_path = output_pdf or input_pdf.with_name(
            f"{input_pdf.stem} - FS only (robust).pdf"
        )
        if selected_indices:
            try:
                write_subset(extraction.source_path, selected_indices, output_path)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise OutputError(f"Failed to write output PDF: {output_path}") from exc

        statement_pages: Dict[StatementType, List[int]] = {}
        for stype, idxs in with_cont.items():
            statement_pages[stype] = [idx + 1 for idx in sorted(idxs)]

        metadata = {
            "file": input_pdf.name,
            "total_pages": total_pages,
            "main_fs_printed_hint": main_hint,
            "inner_index_actual_page_1_based": None if inner_idx is None else inner_idx + 1,
            "printed_map": printed_map,
            "mapped_actual_1_based": {
                stype.value: [idx + 1 for idx in idxs]
                for stype, idxs in with_cont.items()
            },
            "selected_pages_1_based": [idx + 1 for idx in selected_indices],
            "output_pdf": str(output_path) if selected_indices else None,
            "used_pdfplumber": extraction.used_pdfplumber,
            "used_ocr": extraction.used_ocr,
        }

        return StatementSelectionResult(
            file=input_pdf,
            total_pages=total_pages,
            selected_pages=statement_pages,
            metadata=metadata,
        )

    # --- Heuristics ---
    def _find_main_fs_hint(
        self,
        texts: Sequence[str],
        max_scan: int = DEFAULT_MAX_SCAN,
    ) -> int | None:
        trailing = re.compile(r"(\d{1,3})\s*$")
        for idx in range(min(max_scan, len(texts))):
            text = texts[idx]
            if not TABLE_OF_CONTENTS.search(text):
                continue
            for line in (line.strip() for line in text.split("\n") if line.strip()):
                if "financial statements" in line.lower():
                    match = trailing.search(line)
                    if match:
                        return int(match.group(1))
        return None

    def _locate_inner_index(self, texts: Sequence[str], hint_printed: int | None) -> int | None:
        candidates = [i for i, text in enumerate(texts) if INDEX_PAGE.search(text)]
        if not candidates:
            return None
        if hint_printed is None:
            return candidates[0]
        target = hint_printed - 1
        return min(candidates, key=lambda idx: abs(idx - target))

    def _parse_inner_index_printed(self, texts: Sequence[str], idx: int) -> Dict[str, List[int]]:
        trailing = re.compile(r"(\d{1,3})\s*$")
        result = {"balance": [], "income": [], "cash": []}
        page_text = texts[idx]
        for line in (line.strip() for line in page_text.split("\n") if line.strip()):
            match = trailing.search(line)
            if not match:
                continue
            try:
                number = int(match.group(1))
            except ValueError:
                continue
            lower = line.lower()
            if "balance sheet" in lower:
                result["balance"].append(number)
            if re.search(r"statements?\s+of\s+(operations|income|earnings)", lower):
                result["income"].append(number)
            if "statements of cash flows" in lower:
                result["cash"].append(number)
        return result

    def _build_printed_map(
        self,
        texts: Sequence[str],
        inner_idx: int | None,
        main_hint: int | None,
    ) -> Dict[str, List[int]]:
        printed_map = {"balance": [], "income": [], "cash": []}
        if inner_idx is not None:
            return self._parse_inner_index_printed(texts, inner_idx)

        if main_hint is None:
            return printed_map

        approx = main_hint - 1
        start = max(0, approx - self.toc_delta)
        end = min(len(texts), approx + self.toc_delta + 1)

        for idx in range(start, end):
            text = texts[idx]
            for key, (heading, anchors, anchor_min) in ANCHOR_CONFIG.items():
                if self._valid_stmt_page(text, heading, anchors, anchor_min):
                    printed_map[key].append(idx + 1)
        return printed_map

    def _map_to_actual(
        self,
        texts: Sequence[str],
        printed_map: Dict[str, List[int]],
    ) -> Dict[StatementType, List[int]]:
        mapped: Dict[StatementType, List[int]] = {}
        for stype, (heading, anchors, anchor_min) in ANCHOR_CONFIG.items():
            printed = printed_map.get(stype, [])
            actual = self._find_heading_hits(texts, heading)
            valid_actual = [
                idx for idx in actual if self._valid_stmt_page(texts[idx], heading, anchors, anchor_min)
            ]
            chosen: List[int] = []
            for printed_page in printed:
                best: tuple[int, int] | None = None
                for idx in valid_actual:
                    diff = abs((idx + 1) - printed_page)
                    if diff > self.search_window:
                        continue
                    if best is None or diff < best[0]:
                        best = (diff, idx)
                if best:
                    chosen.append(best[1])
            if not chosen:
                if valid_actual:
                    chosen.append(valid_actual[0])
                elif actual:
                    chosen.append(actual[0])
            mapped[StatementType(stype)] = dedup_order(chosen)
        return mapped

    def _expand_with_continuations(
        self,
        texts: Sequence[str],
        mapped: Dict[StatementType, List[int]],
    ) -> Dict[StatementType, List[int]]:
        expanded: Dict[StatementType, List[int]] = {}
        for stype, idxs in mapped.items():
            heading, anchors, anchor_min = ANCHOR_CONFIG[stype.value]
            pages: List[int] = []
            for idx in idxs:
                pages.extend(include_continuations(texts, idx, heading, anchors, anchor_min))
            expanded[stype] = dedup_order(pages)
        return expanded

    def _ensure_fallback_hits(
        self,
        texts: Sequence[str],
        current: Dict[StatementType, List[int]],
    ) -> Dict[StatementType, List[int]]:
        result = dict(current)
        for stype, (heading, anchors, anchor_min) in ANCHOR_CONFIG.items():
            stmt_type = StatementType(stype)
            if result.get(stmt_type):
                continue
            hits = [
                idx
                for idx in self._find_heading_hits(texts, heading)
                if self._valid_stmt_page(texts[idx], heading, anchors, anchor_min)
            ]
            if hits:
                result[stmt_type] = dedup_order(
                    include_continuations(texts, hits[0], heading, anchors, anchor_min)
                )
            else:
                result.setdefault(stmt_type, [])
        return result

    def _cluster_fallback(self, texts: Sequence[str]) -> List[int]:
        buckets = {}
        for stype, (heading, anchors, anchor_min) in ANCHOR_CONFIG.items():
            buckets[stype] = [
                idx
                for idx in self._find_heading_hits(texts, heading)
                if self._valid_stmt_page(texts[idx], heading, anchors, anchor_min)
            ]
        if not all(buckets.values()):
            pool = sorted(idx for values in buckets.values() for idx in values)
            if not pool:
                return []
            center = pool[len(pool) // 2]
            start = max(0, center - 4)
            end = min(len(texts), center + 5)
            return dedup_order(range(start, end))

        combos: List[tuple[int, int, int, int]] = []
        for b_idx in buckets["balance"]:
            for i_idx in buckets["income"]:
                for c_idx in buckets["cash"]:
                    span = max(b_idx, i_idx, c_idx) - min(b_idx, i_idx, c_idx)
                    combos.append((span, b_idx, i_idx, c_idx))
        if not combos:
            return []
        _, b_idx, i_idx, c_idx = min(combos)
        window = range(max(0, min(b_idx, i_idx, c_idx) - 4), min(len(texts), max(b_idx, i_idx, c_idx) + 6))
        collected: List[int] = []
        for idx in window:
            text = texts[idx]
            for stype, (heading, anchors, anchor_min) in ANCHOR_CONFIG.items():
                if self._valid_stmt_page(text, heading, anchors, anchor_min):
                    collected.extend(include_continuations(texts, idx, heading, anchors, anchor_min))
        return dedup_order(collected)

    # --- Helpers ---
    def _find_heading_hits(self, texts: Sequence[str], heading) -> List[int]:
        return [idx for idx, text in enumerate(texts) if heading.search(text)]

    def _valid_stmt_page(
        self,
        text: str,
        heading,
        anchors: Sequence[str],
        anchor_min: int,
        min_numeric: int = 6,
    ) -> bool:
        if not text:
            return False
        if EXCLUDE.search(text):
            return False
        if INDEX_PAGE.search(text):
            return False
        if not near_top(text, heading):
            return False
        if not has_anchors(text, anchors, min_hits=anchor_min):
            return False
        if numeric_density(text) < min_numeric:
            return False
        return True
