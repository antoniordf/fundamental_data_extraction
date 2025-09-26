# Financial Statement Page Selector

Utility for extracting the balance sheet, income statement, and cash-flow pages from SEC filing PDFs. The extractor locates statements using table-of-contents hints, printed page indices, heading recognition, and fallback clustering, optionally re-running OCR when text extraction is poor.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
  - pdfplumber (preferred extractor)
  - PyPDF2 (fallback and page writing)
  - ocrmypdf (optional; only needed for `--auto-ocr`)

## Repository Layout

```
pdf_page_selector.py   # CLI entry point for PDF reduction
carbon_copy_extractor.py
                      # CLI entry point for Excel carbon copy outputs
selector/              # Library package (selector heuristics)
carbon_copy/           # Carbon-copy Excel builder (parsing, normalization, certification)
input_pdfs/            # Default source directory for batch runs
output_pdfs/           # Default destination for generated PDFs
carbon_copy_excels/    # Default destination for generated Excel workbooks
```

## Usage

### Single PDF

```bash
./myenv/bin/python pdf_page_selector.py path/to/filing.pdf
```

- Generates `path/to/filing - FS only (robust).pdf` beside the input unless `--output-dir` or `--output-pdf` is provided.
- Add `--auto-ocr` to re-run extraction through OCR when the first pass yields very sparse text.

### Batch Processing

Place the filings you wish to process inside `input_pdfs/` and run:

```bash
./myenv/bin/python pdf_page_selector.py --all
```

- Every `*.pdf` under `input_pdfs/` (recursively) is processed.
- Outputs are written to `output_pdfs/` by default. Use `--output-dir /path/to/out` to override.
- Failures for individual filings are reported but do not halt the batch; the final JSON includes an `errors` array alongside `results` when any issues occur.

### Planning Minimal Filing Sets

Estimate the smallest collection of 10-K and 10-Q filings required to rebuild `n` years of quarterly and annual history (accounting for overlapping coverage of prior periods):

```bash
./myenv/bin/python pdf_page_selector.py --plan-latest "Q1 2026" --plan-years 10
```

Acceptable `--plan-latest` formats include `FY2024`, `FY-2024`, `Q3 2025`, and `2025Q3`. Planning mode emits a JSON plan listing the required quarters, fiscal years, and the minimal filings that cover them; it runs independently of extraction flags.

Assumptions baked into the planner:

- Each 10-K supplies income statement & cash-flow history for the current fiscal year plus the prior two, and balance sheet snapshots for the current and immediately preceding year-end.
- A Q2 10-Q provides enough data (quarter and year-to-date columns) to reconstruct Q1 and Q2 for the current and prior fiscal year; a Q3 10-Q provides Q3 for the current and prior fiscal year.
- Only the most recent Q1 filing is required when the latest period is Q1; older Q1s are inferred from later Q2 filings.

### Additional Options

- `--input-dir DIR`: base directory to search for PDFs when `--all` or no explicit file is supplied.
- `--output-pdf FILE`: explicit output path (only valid for single-file mode).
- `--search-window N`: tweak the window used when mapping printed index numbers to actual page indices.
- `--toc-delta N`: adjust the number of pages scanned around a Table-of-Contents hint.

## Carbon Copy Extraction to Excel

Once the financial-statement pages have been isolated (step 1), run the carbon copy extractor to mirror the statements in Excel (step 2):

```bash
./myenv/bin/python carbon_copy_extractor.py 'output_pdfs/filing - FS only (robust).pdf'
```

- The PDF is positional; quotes only matter when the filename includes spaces/parentheses.
- Output defaults to `carbon_copy_excels/<pdf-stem>_carbon_copy.xlsx`. Override with `--output path/to/file.xlsx`.
- `--demo` writes a curated EA FY2025 example workbook without reading a PDF.
- Tuning knobs match pdfplumber’s tolerances (`--snap-tol`, `--text-tol`, `--intersection-tol`).

### Excel Workbook Structure

Each workbook contains:

- `WIDE_Income`, `WIDE_BalanceSheet`, `WIDE_CashFlows`: verbatim statements with line-item order preserved, monetary figures normalized to USD millions, and columns reordered oldest → newest.
- `LONG_All`: a denormalized table (one row per statement/period pair) for downstream analytics.
- `Certification`: automated checks (balance sheet equality, cash roll-through, net income link, cross-footing, coverage, units) with pass/variance details.

Behind the scenes, the `carbon_copy` package handles PDF parsing, monetary-unit detection, sign normalization, and certification logic; it is designed to operate on any company’s statements produced by `pdf_page_selector.py`.

## Programmatic Use

The underlying logic is exposed via the `selector` package:

```python
from pathlib import Path
from selector.extractor import PDFTextExtractor
from selector.selector import FinancialStatementSelector

extractor = PDFTextExtractor(enable_ocr=True)
selector = FinancialStatementSelector(extractor)
result = selector.run(Path("input_pdfs/sample.pdf"))
print(result.selected_pages)
```

`StatementSelectionResult.metadata` contains detailed diagnostics, including mapped page numbers, TOC hints, and extraction flags.

## Error Handling

- Missing inputs or empty directories raise `SelectorError` with a descriptive message.
- OCR failures propagate as `ExtractionError`.
- File-write issues (e.g., permission errors) raise `OutputError`.
- The CLI emits errors as JSON and exits with a non-zero status code.
