"""
Shared interface for language-specific function extractors.

This exists so extract_functions_multilang.py doesn't need to know anything
language-specific — it just calls .extract_functions(source_text) on
whichever parser matches the file extension, and gets back a uniform record
shape. Adding a third language later means writing one new class here, not
touching the extraction driver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExtractedFunction:
    """
    One extracted function record.

    Field names deliberately match what the paper describes the existing
    C/C++ pipeline producing (source code, function name, CWE identifier,
    programming language, source file path), so downstream code — dedup,
    splitting, label assignment — can treat all languages identically.
    """

    function_name: str
    source_code: str
    start_line: int
    end_line: int


class BaseFunctionParser(ABC):
    """
    Every language parser must implement extract_functions().

    Kept deliberately minimal: parsing responsibility ends at "here are the
    functions in this file, with their names and bodies." CWE-labeling,
    good/bad classification, and sanitisation are handled uniformly,
    upstream, in labeling.py — not duplicated per language.
    """

    language_name: str

    @abstractmethod
    def extract_functions(self, source_text: str) -> list[ExtractedFunction]:
        """Parse one file's source text and return its top-level functions."""
        raise NotImplementedError

    def safe_extract(self, source_text: str, file_path: str) -> list[ExtractedFunction]:
        """
        Wrapper that never raises. A single malformed file should not crash
        a run of tens of thousands of files. Failures are reported, not
        swallowed silently -- caller is responsible for logging the return.
        """
        try:
            return self.extract_functions(source_text)
        except Exception as exc:  # noqa: BLE001 - intentionally broad, logged upstream
            raise ParseFailure(file_path, str(exc)) from exc


class ParseFailure(Exception):
    """Raised (and caught by the driver) when one file fails to parse."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"{file_path}: {reason}")
