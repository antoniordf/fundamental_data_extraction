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
from selector.planner import plan_minimal_filings
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
    parser.add_argument(
        "--plan-latest",
        dest="plan_latest",
        help="Latest available filing period (e.g. FY2024, Q1 2026) for coverage planning",
    )
    parser.add_argument(
        "--plan-years",
        dest="plan_years",
        type=int,
        default=None,
        help="Number of fiscal years to cover when planning the minimal filing set",
    )
    parser.add_argument(
        "--plan-json",
        action="store_true",
        help="Emit plan output as JSON instead of a simple filing list",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every PDF found under the resolved input directory",
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


def resolve_input_paths(args: argparse.Namespace) -> List[Path]:
    if args.input_pdf:
        input_path = Path(args.input_pdf)
        if input_path.is_dir():
            pdfs = sorted(input_path.rglob("*.pdf"))
            if not pdfs:
                raise SelectorError(f"No PDF files found in directory: {input_path}")
            return pdfs
        return [input_path]

    if args.all or args.input_dir or args.scan_dir:
        base_dir = Path(args.input_dir or args.scan_dir or DEFAULT_INPUT_DIR)
        if not base_dir.exists():
            if base_dir == DEFAULT_INPUT_DIR and not (args.input_dir or args.scan_dir):
                base_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise SelectorError(f"Input directory does not exist: {base_dir}")
        pdfs = sorted(base_dir.rglob("*.pdf"))
        if not pdfs:
            raise SelectorError(f"No PDF files found in directory: {base_dir}")
        return pdfs

    return [resolve_input_path(args)]


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
    if args.all and args.output_pdf:
        parser.error("--all cannot be combined with an explicit output PDF path.")

    if args.plan_latest:
        if args.auto_ocr or args.all or args.input_pdf or args.scan_dir or args.output_pdf:
            parser.error("Planning mode cannot be combined with extraction options.")
        plan_years = args.plan_years or 10
        plan = plan_minimal_filings(args.plan_latest, plan_years)
        if args.plan_json:
            print(json.dumps(plan.to_json(), indent=2))
        else:
            sorted_filings = sorted(
                plan.filings,
                key=lambda f: (
                    0 if f.filing_type == "10-K" else 1,
                    -f.year,
                    f.quarter or 0,
                ),
            )
            for filing in sorted_filings:
                label = f"{filing.filing_type} {filing.year}"
                if filing.quarter is not None:
                    label += f" Q{filing.quarter}"
                print(label)
        return 0

    try:
        input_paths = resolve_input_paths(args)
    except (ExtractionError, OutputError, SelectorError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    input_is_dir = bool(args.input_pdf and Path(args.input_pdf).is_dir())
    batch_mode = len(input_paths) > 1 or args.all or input_is_dir

    extractor = PDFTextExtractor(enable_ocr=args.auto_ocr)
    selector = FinancialStatementSelector(
        extractor,
        search_window=args.search_window or DEFAULT_SEARCH_WINDOW,
        toc_delta=args.toc_delta or DEFAULT_TOC_DELTA,
    )

    output_dir_path: Path | None = None
    if batch_mode:
        output_dir_path = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
        output_dir_path.mkdir(parents=True, exist_ok=True)

    outputs: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    for input_path in input_paths:
        if batch_mode:
            assert output_dir_path is not None
            output_path = output_dir_path / f"{input_path.stem} - FS only (robust).pdf"
        else:
            output_path = resolve_output_path(input_path, args)

        try:
            result = selector.run(input_path, output_pdf=output_path)
        except (ExtractionError, OutputError, SelectorError) as exc:
            errors.append({"file": str(input_path), "error": str(exc)})
            continue

        metadata = serialize_metadata(result.metadata)
        metadata.setdefault("selected_pages", {})
        metadata["selected_pages"] = {
            stype.value: result.selected_pages.get(stype, []) for stype in StatementType
        }
        outputs.append(metadata)

    if errors or len(outputs) != 1 or batch_mode:
        payload: Dict[str, object] = {"results": outputs}
        if errors:
            payload["errors"] = errors
        print(json.dumps(payload, indent=2))
        return 1 if errors else 0

    # single successful run
    print(json.dumps(outputs[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
