"""Utility helpers for PDF text processing."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from PyPDF2 import PdfReader, PdfWriter

from .constants import (
    CONTINUED,
    CONTINUATION_ANCHOR_BONUS,
    INDEX_PAGE,
    MAX_STATEMENT_EXTENSION_PAGES,
    MIN_CONTINUATION_DENSITY,
    OTHER_STATEMENT_PATS,
)


def have_pdfplumber() -> bool:
    try:
        import pdfplumber  # noqa: F401
        return True
    except Exception:
        return False


def have_ocr() -> bool:
    return shutil.which("ocrmypdf") is not None


def norm(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\s*\n", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def dedup_order(seq: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def near_top(text: str, pat, max_chars: int = 500) -> bool:
    match = pat.search(text)
    return bool(match and match.start() <= max_chars)


def numeric_density(text: str) -> int:
    if not text:
        return 0
    dollar = text.count("$")
    comma_nums = len(re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", text))
    return dollar + comma_nums


def has_anchors(text: str, anchors: Sequence[str], min_hits: int = 3) -> bool:
    lowered = (text or "").lower()
    return sum(1 for anchor in anchors if anchor in lowered) >= min_hits


def write_subset(src: Path, idxs: Iterable[int], out_path: Path) -> None:
    reader = PdfReader(str(src))
    writer = PdfWriter()
    for idx in sorted(set(idxs)):
        if 0 <= idx < len(reader.pages):
            writer.add_page(reader.pages[idx])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        writer.write(handle)


@dataclass
class OCRResult:
    path: Path
    texts: List[str]


def run_ocr(src: Path) -> OCRResult | None:
    if not have_ocr():
        return None
    with tempfile.TemporaryDirectory() as tmp_dir:
        ocr_out = Path(tmp_dir) / "ocr.pdf"
        cmd = [
            "ocrmypdf",
            "--force-ocr",
            "--deskew",
            "--clean",
            "--optimize",
            "2",
            str(src),
            str(ocr_out),
        ]
        subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not ocr_out.exists():
            return None
        texts = extract_texts_via_pdfplumber(ocr_out) if have_pdfplumber() else extract_texts_via_pypdf(ocr_out)
        return OCRResult(path=ocr_out, texts=texts)


def extract_texts(path: Path) -> List[str]:
    if have_pdfplumber():
        return extract_texts_via_pdfplumber(path)
    return extract_texts_via_pypdf(path)


def extract_texts_via_pdfplumber(path: Path) -> List[str]:
    import pdfplumber

    texts: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            texts.append(norm(text))
    return texts


def extract_texts_via_pypdf(path: Path) -> List[str]:
    reader = PdfReader(str(path))
    texts: List[str] = []
    for i in range(len(reader.pages)):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:
            text = ""
        texts.append(norm(text))
    return texts


def low_text_quality(texts: Sequence[str], threshold_chars: int, sample_pages: int) -> bool:
    sample = texts[: min(sample_pages, len(texts))]
    total = sum(len(page) for page in sample)
    return total < threshold_chars


def continuation_limits(text: str, heading_pat) -> bool:
    if not text:
        return False
    if INDEX_PAGE.search(text):
        return True
    if any(pat.search(text) for pat in OTHER_STATEMENT_PATS if pat is not heading_pat):
        return True
    return False


def include_continuations(
    texts: Sequence[str],
    start_idx: int,
    heading_pat,
    anchors: Sequence[str],
    anchor_min: int,
    max_extension: int = MAX_STATEMENT_EXTENSION_PAGES,
) -> List[int]:
    pages = [start_idx]
    idx = start_idx + 1
    extensions = 0
    while idx < len(texts) and extensions < max_extension:
        text = texts[idx]
        if not text:
            break
        if continuation_limits(text, heading_pat):
            break
        if CONTINUED.search(text) or heading_pat.search(text):
            if numeric_density(text) >= 4:
                pages.append(idx)
                idx += 1
                extensions += 1
                continue
        if numeric_density(text) >= MIN_CONTINUATION_DENSITY and has_anchors(
            text,
            anchors,
            min_hits=max(1, anchor_min - CONTINUATION_ANCHOR_BONUS),
        ):
            pages.append(idx)
            idx += 1
            extensions += 1
            continue
        break
    return pages
