"""Custom exceptions for the selector package."""
from __future__ import annotations


class SelectorError(RuntimeError):
    """Base error for the financial statement selector."""


class ExtractionError(SelectorError):
    """Raised when a PDF cannot be processed."""


class OutputError(SelectorError):
    """Raised when the output PDF cannot be written."""
