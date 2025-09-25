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
pdf_page_selector.py   # CLI entry point
selector/              # Library package (extractor, heuristics, utils)
input_pdfs/            # Default source directory for batch runs
output_pdfs/           # Default destination for generated PDFs
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

### Additional Options

- `--input-dir DIR`: base directory to search for PDFs when `--all` or no explicit file is supplied.
- `--output-pdf FILE`: explicit output path (only valid for single-file mode).
- `--search-window N`: tweak the window used when mapping printed index numbers to actual page indices.
- `--toc-delta N`: adjust the number of pages scanned around a Table-of-Contents hint.

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
