"""
Canonical source-code sanitizer shared across the C/C++ and Java/C#
(multilang) extraction pipelines.

Strips the two confirmed label-leakage channels found in the Juliet-derived
training text:

1. Naming leakage: "bad"/"good"/"G2B"/"B2G" substrings inside identifiers
   (the function's own name, and any sibling-helper-function references
   inside the body) that deterministically correlate with the label —
   quantified at 58.2% of Java/C# rows carrying a G2B/B2G identifier, 100%
   of which are label 0.
2. Comment leakage: Juliet's own instructional comments containing
   FLAW / POTENTIAL FLAW / FIX / INCIDENTAL markers, which spell out in
   plain English whether the function is the vulnerable or safe half of
   the test case — quantified at a ~17x difference in FIX-token frequency
   between the two classes.

Applied identically to C/C++ and Java/C# so the fix doesn't introduce a
new asymmetry between languages.
"""

from __future__ import annotations

import re

# Matches C-style comments while protecting string/char literal contents,
# so a `//` or `/*` sequence inside a quoted string is never mistaken for
# a comment start. Standard technique: alternate comment patterns with
# string patterns in one regex so re.sub only ever replaces genuine
# comments, leaving string/char literals untouched.
_COMMENT_OR_STRING_RE = re.compile(
    r'//[^\n]*'  # line comment
    r'|/\*.*?\*/'  # block comment
    r'|"(?:\\.|[^"\\])*"'  # double-quoted string/char literal
    r"|'(?:\\.|[^'\\])*'",  # single-quoted char/string literal
    re.DOTALL,
)

_CWE_ID_RE = re.compile(r"CWE\d+_[A-Za-z0-9_]*")
_LEAKY_IDENTIFIER_RE = re.compile(r"\b\w*(?:bad|good|G2B|B2G)\w*\b", re.IGNORECASE)


def _strip_comment_keep_string(match: re.Match) -> str:
    text = match.group(0)
    if text.startswith("//") or text.startswith("/*"):
        # Comment: drop it, but keep line breaks so line numbers don't shift.
        return "\n" * text.count("\n")
    return text  # string/char literal: leave untouched


def sanitize_source_code(code: str) -> str:
    """
    Remove comment-based and naming-based label leakage from a Juliet
    function body. Order matters: strip comments first (so leaky
    identifiers mentioned only inside comments don't need special-casing),
    then normalise leaky identifiers in the remaining code, including the
    function's own name in its signature line.
    """
    code = _COMMENT_OR_STRING_RE.sub(_strip_comment_keep_string, code)
    code = _CWE_ID_RE.sub("FUNC", code)
    code = _LEAKY_IDENTIFIER_RE.sub("FUNC", code)
    return code
