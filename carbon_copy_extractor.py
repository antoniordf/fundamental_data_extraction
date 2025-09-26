#!/usr/bin/env python3
"""Extract consolidated financial statements into WIDE/LONG Excel outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from carbon_copy import (
    CarbonCopyError,
    CarbonCopyExtractor,
    ExtractionConfig,
    build_demo_result,
    write_excel_workbook,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Excel-ready income statement, balance sheet, and cash flow tables "
            "from a PDF produced by pdf_page_selector."
        )
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Write the curated demo workbook (EA FY2025) without reading a PDF.",
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_flag",
        type=Path,
        metavar="PATH",
        help="Input PDF containing the three statements (legacy flag; positional argument preferred).",
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        help="Input PDF containing the three statements (e.g., output from pdf_page_selector).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination Excel file (*.xlsx). If omitted, defaults to carbon_copy_excels/<pdf-stem>_carbon_copy.xlsx.",
    )
    parser.add_argument(
        "--snap-tol",
        type=float,
        default=3.0,
        help="pdfplumber snap tolerance in points (default: 3.0).",
    )
    parser.add_argument(
        "--text-tol",
        type=float,
        default=2.0,
        help="pdfplumber text tolerance in points (default: 2.0).",
    )
    parser.add_argument(
        "--intersection-tol",
        type=float,
        default=2.0,
        help="pdfplumber intersection tolerance in points (default: 2.0).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.pdf_flag and args.pdf and args.pdf_flag != args.pdf:
            parser.error("Provide the PDF only once (positional or --pdf, not both).")
        pdf_path = args.pdf or args.pdf_flag
        if args.demo:
            if pdf_path is not None:
                parser.error("Do not supply a PDF when using --demo.")
            result = build_demo_result()
            default_output = Path("carbon_copy_excels/demo_carbon_copy.xlsx")
        else:
            if pdf_path is None:
                parser.error("Provide a PDF path (positional argument) or use --demo.")
            config = ExtractionConfig(
                snap_tolerance=args.snap_tol,
                text_tolerance=args.text_tol,
                intersection_tolerance=args.intersection_tol,
            )
            extractor = CarbonCopyExtractor(config)
            result = extractor.extract(pdf_path)
            default_output = Path("carbon_copy_excels") / f"{pdf_path.stem}_carbon_copy.xlsx"

        output_path = args.output or default_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_excel_workbook(result, output_path)
        print(f"Wrote workbook to {output_path}")
        return 0
    except CarbonCopyError as exc:
        print(f"Carbon copy extraction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
