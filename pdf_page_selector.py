#!/usr/bin/env python3
"""Command line interface for the financial statement page selector."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from selector.constants import DEFAULT_SEARCH_WINDOW, DEFAULT_TOC_DELTA
from selector.exceptions import ExtractionError, OutputError, SelectorError
from selector.extractor import PDFTextExtractor
from selector.models import StatementType
from selector.selector import FinancialStatementSelector

DEFAULT_INPUT_DIR = Path("input_pdfs")
DEFAULT_OUTPUT_DIR = Path("output_pdfs")


def find_first_pdf(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    for path in sorted(directory.rglob("*.pdf")):
        if path.is_file():
            return path
    return None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract financial statement pages from an SEC filing PDF.",
    )
    parser.add_argument("input_pdf", nargs="?", help="Input PDF to scan")
    parser.add_argument("output_pdf", nargs="?", help="Optional output path for the extracted pages")
    parser.add_argument("--auto-ocr", action="store_true", help="Run OCR if the initial extraction yields low text quality")
    parser.add_argument("--input-dir", dest="input_dir", help="Directory to search for a PDF when input path is not provided")
    parser.add_argument("--output-dir", dest="output_dir", help="Directory where extracted PDFs should be written")
    parser.add_argument(
        "--scan-dir",
        dest="scan_dir",
        help="Directory to search for a PDF when input path is not provided (deprecated; use --input-dir)",
    )
    parser.add_argument(
        "--search-window",
        type=int,
        default=None,
        help="Maximum distance (in pages) between printed index entries and actual statements",
    )
    parser.add_argument(
        "--toc-delta",
        type=int,
        default=None,
        help="Range of pages around the TOC hint to inspect when the index is unavailable",
    )
    return parser


def resolve_input_path(args: argparse.Namespace) -> Path:
    input_pdf = args.input_pdf
    search_dir_opt = args.input_dir or args.scan_dir

    if input_pdf:
        return Path(input_pdf)

    search_dir = Path(search_dir_opt) if search_dir_opt else DEFAULT_INPUT_DIR
    if not search_dir.exists():
        if search_dir_opt:
            raise SelectorError(f"Input directory does not exist: {search_dir}")
        search_dir.mkdir(parents=True, exist_ok=True)

    found_pdf = find_first_pdf(search_dir)
    if not found_pdf:
        raise SelectorError(f"No PDF files found in directory: {search_dir}")
    return found_pdf


def resolve_output_path(input_path: Path, args: argparse.Namespace) -> Path:
    if args.output_pdf:
        out_path = Path(args.output_pdf)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem} - FS only (robust).pdf"


def serialize_metadata(metadata: Dict[str, object]) -> Dict[str, object]:
    serializable: Dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        elif isinstance(value, dict):
            serializable[key] = serialize_metadata(value)  # type: ignore[arg-type]
        elif isinstance(value, list):
            serializable[key] = [serialize_metadata(item) if isinstance(item, dict) else item for item in value]  # type: ignore[list-item]
        else:
            serializable[key] = value
    return serializable


def main(argv: List[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.output_dir and args.output_pdf:
        parser.error("Specify either an explicit output PDF path or --output-dir, not both.")

    try:
        input_path = resolve_input_path(args)
        output_path = resolve_output_path(input_path, args)

        extractor = PDFTextExtractor(enable_ocr=args.auto_ocr)
        selector = FinancialStatementSelector(
            extractor,
            search_window=args.search_window or DEFAULT_SEARCH_WINDOW,
            toc_delta=args.toc_delta or DEFAULT_TOC_DELTA,
        )
        result = selector.run(input_path, output_pdf=output_path)

        metadata = serialize_metadata(result.metadata)
        metadata.setdefault("selected_pages", {})
        metadata["selected_pages"] = {
            stype.value: result.selected_pages.get(stype, []) for stype in StatementType
        }
        print(json.dumps(metadata, indent=2))
        return 0
    except (ExtractionError, OutputError, SelectorError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
