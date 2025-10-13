"""Carbon copy (verbatim) financial statement extraction."""
from .aggregation import SeriesAggregator
from .excel import (
    write_excel_workbook,
    write_time_series_workbook,
    write_long_aggregation_workbook,
)
from .extractor import CarbonCopyError, CarbonCopyExtractor, ExtractionConfig
from .models import (
    CarbonCopyResult,
    LabelReviewRequest,
    LabelSuggestion,
    StatementBlock,
    SuspectLabel,
)
from .llm_processor import LabelLLMProcessor, apply_label_suggestions
from .demo import build_demo_result

__all__ = [
    "CarbonCopyError",
    "CarbonCopyExtractor",
    "ExtractionConfig",
    "CarbonCopyResult",
    "StatementBlock",
    "SuspectLabel",
    "LabelReviewRequest",
    "LabelSuggestion",
    "LabelLLMProcessor",
    "apply_label_suggestions",
    "write_excel_workbook",
    "write_time_series_workbook",
    "write_long_aggregation_workbook",
    "build_demo_result",
    "SeriesAggregator",
]
