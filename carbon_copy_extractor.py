#!/usr/bin/env python3
"""Extract and aggregate financial statements into structured Excel workbooks."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from carbon_copy import (
    CarbonCopyError,
    CarbonCopyExtractor,
    ExtractionConfig,
    CarbonCopyResult,
    LabelLLMProcessor,
    LabelSuggestion,
    apply_label_suggestions,
    build_demo_result,
    write_excel_workbook,
    write_long_aggregation_workbook,
)
from carbon_copy.batch_certification import build_batch_certification


DEFAULT_PER_FILE_DIR = Path("carbon_copy_excels")
DEFAULT_DEMO_PER_FILE = Path("carbon_copy_excels/demo_carbon_copy.xlsx")
DEFAULT_LONG_OUTPUT = Path("carbon_copy_excels/carbon_copy_long.xlsx")
DEFAULT_DEMO_LONG = Path("carbon_copy_excels/demo_long.xlsx")

LONG_COLUMNS = [
    "Source PDF",
    "Statement",
    "Order Index",
    "Level",
    "Line Item (as printed)",
    "Column Header (as printed)",
    "Value (in millions)",
    "Original Units",
    "Saved Units",
    "Units Assumption",
    "Audited/Unaudited",
    "Page",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract statement tables from PDFs and optionally stack all LONG_All rows "
            "into a single workbook for spreadsheet modeling."
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
        help="Input PDF (legacy flag; positional argument preferred).",
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        help="Input PDF produced by pdf_page_selector (aggregated when combined with --batch).",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        help="Directory containing PDFs to process and aggregate into a single workbook.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination Excel file (*.xlsx) for the aggregated long-format workbook. "
            "Defaults to carbon_copy_excels/carbon_copy_long.xlsx."
        ),
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Also emit an individual workbook for each processed PDF (original wide/long layout).",
    )
    parser.add_argument(
        "--per-file-dir",
        type=Path,
        help="Destination directory for per-file outputs when --per-file is used (default: carbon_copy_excels).",
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
    parser.add_argument(
        "--llm-normalize",
        action="store_true",
        help="Normalize suspect labels with an LLM (requires OPENAI_API_KEY and the openai package).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model name for label normalization (overrides OPENAI_LABEL_MODEL/env default).",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for label normalization requests (default: 0.0).",
    )
    parser.add_argument(
        "--llm-min-confidence",
        type=float,
        default=0.0,
        help="Apply suggestions only when confidence is at least this value (default: 0.0). Ignored in --llm-dry-run mode.",
    )
    parser.add_argument(
        "--llm-dry-run",
        action="store_true",
        help="Do not mutate labels; log suggested normalizations for review.",
    )
    return parser


def resolve_pdfs(args: argparse.Namespace) -> List[Path]:
    pdfs: List[Path] = []
    for candidate in [args.pdf_flag, args.pdf]:
        if candidate:
            pdfs.append(candidate)
    if args.batch:
        if not args.batch.exists() or not args.batch.is_dir():
            raise CarbonCopyError(f"Batch directory not found: {args.batch}")
        pdfs.extend(sorted(path for path in args.batch.glob("*.pdf")))
    unique: List[Path] = []
    seen: set[Path] = set()
    for path in pdfs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            raise CarbonCopyError(f"Input PDF not found: {resolved}")
        if resolved.suffix.lower() != ".pdf":
            continue
        seen.add(resolved)
        unique.append(resolved)
    unique.sort()
    return unique


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_per_file(result: CarbonCopyResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result.source_pdf.stem}_carbon_copy.xlsx"
    write_excel_workbook(result, output_path)
    return output_path


def persist_suggestions_log(
    pdf_path: Path,
    suggestions: Sequence[LabelSuggestion],
    applied: Sequence[LabelSuggestion],
    *,
    dry_run: bool,
    min_confidence: float,
) -> Optional[Path]:
    if not suggestions:
        return None
    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pdf": str(pdf_path),
        "dry_run": dry_run,
        "min_confidence": min_confidence,
        "total_suggestions": len(suggestions),
        "applied_suggestions": len(applied),
        "applied_indices": [item.row_index for item in applied],
        "suggestions": [asdict(item) for item in suggestions],
    }
    log_path = out_dir / f"{pdf_path.stem}.label_suggestions.json"
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return log_path


def prepare_long_table(
    results: List[CarbonCopyResult],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    certifications: List[pd.DataFrame] = []
    for result in results:
        frame = result.long_table.copy()
        frame.insert(0, "Source PDF", result.source_pdf.name)
        for column in LONG_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""
        frame = frame[LONG_COLUMNS]
        frames.append(frame)
        cert = result.certification.copy()
        cert.insert(0, "Source PDF", result.source_pdf.name)
        certifications.append(cert)
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        empty = pd.DataFrame(columns=LONG_COLUMNS)
        return empty, pd.DataFrame(), pd.DataFrame(columns=["Source PDF", "Check", "Result", "Details"])
    combined_columns = {
        column: pd.concat([frame[column] for frame in frames], ignore_index=True)
        for column in LONG_COLUMNS
    }
    long_combined = pd.DataFrame(combined_columns)
    certification_combined = (
        pd.concat(certifications, ignore_index=True) if certifications else pd.DataFrame()
    )
    batch_certification = build_batch_certification(results)
    if not batch_certification.empty:
        certification_combined = (
            pd.concat([certification_combined, batch_certification], ignore_index=True)
            if not certification_combined.empty
            else batch_certification
        )
    return long_combined, certification_combined, batch_certification


def collect_certification_issues(
    results: List[CarbonCopyResult],
    batch_certification: pd.DataFrame | None = None,
) -> pd.DataFrame:
    records: List[dict[str, str]] = []
    for result in results:
        df = result.certification
        if df is None or df.empty:
            continue
        if "Result" not in df.columns:
            continue
        flagged = df[~df["Result"].isin(["PASS", "INFO"])]
        if flagged.empty:
            continue
        for _, row in flagged.iterrows():
            records.append(
                {
                    "Source PDF": result.source_pdf.name,
                    "Check": str(row.get("Check", "")),
                    "Result": str(row.get("Result", "")),
                    "Details": str(row.get("Details", "")),
                }
            )
    if batch_certification is not None and not batch_certification.empty:
        flagged = batch_certification[~batch_certification["Result"].isin(["PASS", "INFO"])]
        for _, row in flagged.iterrows():
            records.append(
                {
                    "Source PDF": str(row.get("Source PDF", "")),
                    "Check": str(row.get("Check", "")),
                    "Result": str(row.get("Result", "")),
                    "Details": str(row.get("Details", "")),
                }
            )
    if not records:
        return pd.DataFrame(columns=["Source PDF", "Check", "Result", "Details"])
    return pd.DataFrame(records)


def print_certification_summary(
    results: List[CarbonCopyResult],
    batch_certification: pd.DataFrame | None = None,
) -> None:
    issues = collect_certification_issues(results, batch_certification)
    if issues.empty:
        if results:
            print("Certification summary: all checks passed.")
        return
    print("Certification issues detected:")
    for record in issues.to_dict("records"):
        details = record.get("Details")
        suffix = f" ({details})" if details else ""
        print(
            f" - {record.get('Source PDF', '')}: {record.get('Check', '')} "
            f"→ {record.get('Result', '')}{suffix}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.pdf_flag and args.pdf and args.pdf_flag != args.pdf:
            parser.error("Provide the PDF only once (positional or --pdf, not both).")

        if args.demo:
            if args.pdf or args.pdf_flag or args.batch:
                parser.error("Do not supply PDFs when using --demo.")
            result = build_demo_result()
            demo_results = [result]
            long_table, certification, batch_certification = prepare_long_table(demo_results)
            long_output = args.output or DEFAULT_DEMO_LONG
            ensure_directory(long_output)
            write_long_aggregation_workbook(long_table, certification, long_output)
            print(f"Wrote demo long-format workbook to {long_output}")
            if args.per_file:
                per_file_path = write_per_file(result, args.per_file_dir or DEFAULT_PER_FILE_DIR)
                print(f"Wrote demo per-file workbook to {per_file_path}")
            print_certification_summary(demo_results, batch_certification)
            return 0

        pdf_paths = resolve_pdfs(args)
        if not pdf_paths:
            parser.error("Provide at least one PDF path, --batch directory, or use --demo.")

        config = ExtractionConfig(
            snap_tolerance=args.snap_tol,
            text_tolerance=args.text_tol,
            intersection_tolerance=args.intersection_tol,
        )
        extractor = CarbonCopyExtractor(config)
        per_file_outputs: List[Path] = []
        results: List[CarbonCopyResult] = []
        for pdf_path in pdf_paths:
            result = extractor.extract(pdf_path)
            results.append(result)
            if args.per_file:
                per_file_outputs.append(
                    write_per_file(result, args.per_file_dir or DEFAULT_PER_FILE_DIR)
                )

        if args.llm_normalize and results:
            llm_model = args.llm_model or os.getenv("OPENAI_LABEL_MODEL", "gpt-4.1-mini")
            min_conf = max(0.0, args.llm_min_confidence)
            for result in results:
                try:
                    processor = LabelLLMProcessor(allow_mock=args.llm_dry_run)
                except RuntimeError as exc:
                    raise CarbonCopyError(str(exc)) from exc
                suggestions = processor.review_result(
                    result,
                    model=llm_model,
                    temperature=args.llm_temperature,
                    dry_run=args.llm_dry_run,
                )
                applied: List[LabelSuggestion] = []
                if suggestions and not args.llm_dry_run:
                    apply_label_suggestions(
                        result,
                        suggestions,
                        min_confidence=min_conf,
                    )
                    applied = [
                        suggestion
                        for suggestion in suggestions
                        if suggestion.confidence is None or suggestion.confidence >= min_conf
                    ]
                log_path = persist_suggestions_log(
                    result.source_pdf,
                    suggestions,
                    applied,
                    dry_run=args.llm_dry_run,
                    min_confidence=min_conf,
                )
                usage = processor.summarize_usage(pdf_name=result.source_pdf.name)
                total = len(suggestions)
                applied_count = len(applied) if not args.llm_dry_run else 0
                log_suffix = f" (log: {log_path})" if log_path else ""
                print(
                    f"LLM normalization for {result.source_pdf.name}: "
                    f"suggestions={total}, applied={applied_count}{log_suffix}"
                )
                totals = usage.get("totals") if isinstance(usage, dict) else None
                if isinstance(totals, dict):
                    tokens = totals.get("total_tokens")
                    cost = totals.get("estimated_cost_usd")
                    token_fragments: List[str] = []
                    if tokens:
                        token_fragments.append(f"tokens={tokens}")
                    if isinstance(cost, (int, float)):
                        token_fragments.append(f"cost≈${float(cost):.4f}")
                    if token_fragments:
                        print(f"  Usage ({', '.join(token_fragments)})")

        long_table, certification, batch_certification = prepare_long_table(results)
        long_output = args.output or DEFAULT_LONG_OUTPUT
        ensure_directory(long_output)
        write_long_aggregation_workbook(long_table, certification, long_output)

        print(f"Processed {len(pdf_paths)} PDF(s). Wrote long-format workbook to {long_output}")
        if args.per_file and per_file_outputs:
            print(f"Wrote {len(per_file_outputs)} per-file workbook(s) to {(args.per_file_dir or DEFAULT_PER_FILE_DIR).resolve()}")
        print_certification_summary(results, batch_certification)
        return 0
    except CarbonCopyError as exc:
        print(f"Carbon copy extraction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
