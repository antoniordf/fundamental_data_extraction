"""PDF text extraction utilities with optional OCR."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from .constants import (
    DEFAULT_LOW_TEXT_SAMPLE,
    DEFAULT_LOW_TEXT_THRESHOLD,
)
from .exceptions import ExtractionError
from .utils import (
    OCRResult,
    extract_texts,
    have_pdfplumber,
    have_ocr,
    low_text_quality,
    run_ocr,
)


@dataclass
class ExtractionOutput:
    texts: List[str]
    source_path: Path
    used_pdfplumber: bool
    used_ocr: bool


class PDFTextExtractor:
    def __init__(
        self,
        low_text_threshold: int = DEFAULT_LOW_TEXT_THRESHOLD,
        sample_pages: int = DEFAULT_LOW_TEXT_SAMPLE,
        enable_ocr: bool = False,
    ) -> None:
        self.low_text_threshold = low_text_threshold
        self.sample_pages = sample_pages
        self.enable_ocr = enable_ocr

    def extract(self, path: Path) -> ExtractionOutput:
        if not path.exists():
            raise ExtractionError(f"Input PDF not found: {path}")

        texts = extract_texts(path)
        used_pdfplumber = have_pdfplumber()

        if (
            self.enable_ocr
            and have_ocr()
            and low_text_quality(texts, self.low_text_threshold, self.sample_pages)
        ):
            ocr_result = self._perform_ocr(path)
            if ocr_result:
                return ExtractionOutput(
                    texts=ocr_result.texts,
                    source_path=ocr_result.path,
                    used_pdfplumber=have_pdfplumber(),
                    used_ocr=True,
                )

        return ExtractionOutput(
            texts=texts,
            source_path=path,
            used_pdfplumber=used_pdfplumber,
            used_ocr=False,
        )

    def _perform_ocr(self, path: Path) -> OCRResult | None:
        try:
            return run_ocr(path)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ExtractionError("OCR process failed") from exc
