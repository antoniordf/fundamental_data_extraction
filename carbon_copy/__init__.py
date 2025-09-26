"""Carbon copy (verbatim) financial statement extraction."""
from .excel import write_excel_workbook
from .extractor import CarbonCopyError, CarbonCopyExtractor, ExtractionConfig
from .models import CarbonCopyResult, StatementBlock
from .demo import build_demo_result

__all__ = [
    "CarbonCopyError",
    "CarbonCopyExtractor",
    "ExtractionConfig",
    "CarbonCopyResult",
    "StatementBlock",
    "write_excel_workbook",
    "build_demo_result",
]
