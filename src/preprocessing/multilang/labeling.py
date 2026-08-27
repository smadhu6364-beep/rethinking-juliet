"""
Label assignment shared across all languages.

Kept as one module, not duplicated per language, so the labeling rule is
guaranteed identical for C/C++, Java, and C# — this is the thing that makes
cross-language comparison in the paper meaningful at all. If Java used a
different label rule than C#, any difference in results would be confounded
by that instead of reflecting anything about the language itself.

Convention (per the paper's stated methodology and Juliet's own naming
scheme): functions containing "bad" in the name/path are vulnerable (label
1); functions containing "good" (including good1, good2, goodG2B, goodB2G
variants) are safe (label 0). CWE identifiers come from the directory path,
matching Juliet's CWEnnn_Description folder structure across all four
language variants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_CWE_DIR_PATTERN = re.compile(r"CWE(\d+)_", re.IGNORECASE)

# Tokens that would leak the label directly into the training text if left
# in the function name. Originally just "bad"/"good"; extended to also
# catch "G2B"/"B2G" after quantifying that 58.2% of Java/C# rows carry a
# G2B/B2G identifier, 100% of which map to label 0 -- a leak, not a
# disclosed-and-accepted limitation.
_NAME_SANITIZE_PATTERN = re.compile(r"(bad|good|G2B|B2G)", re.IGNORECASE)

# Comment tokens that constitute label leakage in the training text,
# identical list to the one quantified for C/C++ (FLAW, POTENTIAL, FIX,
# INCIDENTAL). Counting these per-language, not removing them, is the
# point: it's what lets this project state whether the same artefact
# recurs in Java/C# or not.
LEAKAGE_TOKENS = ("FLAW", "POTENTIAL", "FIX", "INCIDENTAL")


@dataclass
class LabelResult:
    label: int          # 1 = vulnerable, 0 = safe
    cwe_id: str | None
    sanitized_name: str
    leakage_token_counts: dict[str, int]


def extract_cwe_id(file_path: str) -> str | None:
    """Pull the CWE identifier from the directory path, same convention
    used across all Juliet language variants."""
    match = _CWE_DIR_PATTERN.search(str(file_path))
    return f"CWE{match.group(1)}" if match else None


def assign_label(function_name: str) -> int:
    """
    1 if the function name marks it vulnerable ('bad'), 0 if safe ('good').
    Raises if neither pattern is found rather than silently guessing --
    an unrecognized naming pattern should stop the pipeline, not produce
    a mislabeled sample.
    """
    lowered = function_name.lower()
    if "bad" in lowered:
        return 1
    if "good" in lowered:
        return 0
    raise ValueError(
        f"Cannot determine label from function name: '{function_name}'. "
        f"Does not match Juliet's bad/good naming convention."
    )


def sanitize_name(function_name: str) -> str:
    """Remove the obvious leaking tokens (bad/good/G2B/B2G) from the name.
    See src/preprocessing/sanitize.py for the corresponding fix applied to
    the full source-code body (comments + in-body identifier references),
    which is what training actually reads."""
    return _NAME_SANITIZE_PATTERN.sub("", function_name)


def count_leakage_tokens(source_code: str) -> dict[str, int]:
    """Count occurrences of each known leakage token in the function body.
    This is measurement, not removal -- matches how the C/C++ analysis in
    Section III-D was framed (quantify the leak, don't just delete it and
    move on)."""
    return {
        token: len(re.findall(rf"\b{token}\b", source_code, re.IGNORECASE))
        for token in LEAKAGE_TOKENS
    }


def label_function(function_name: str, file_path: str, source_code: str) -> LabelResult:
    return LabelResult(
        label=assign_label(function_name),
        cwe_id=extract_cwe_id(file_path),
        sanitized_name=sanitize_name(function_name),
        leakage_token_counts=count_leakage_tokens(source_code),
    )
