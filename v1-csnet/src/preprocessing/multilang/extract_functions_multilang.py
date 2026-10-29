"""
Extraction driver for Java and C#.

Output schema is deliberately identical to what the paper describes the
existing C/C++ pipeline producing: source code, function name, CWE id,
programming language, source file path, label. Same columns, same dtypes,
so add_splits.py-equivalent logic and the training scripts need to change
only their input path, not their column-handling code.

IMPORTANT — stated plainly, not buried: this driver has not been run
against real Juliet Java/C# source trees, because those files are not yet
available in this environment. It has been smoke-tested against small
synthetic Java/C# snippets (see tests/) to confirm the parsing and
labeling logic executes correctly end-to-end. Running it against the real
NIST SARD Juliet Java/C# suites, and checking the resulting counts, is the
next required step once the source trees are supplied -- do not treat any
row count from this script as real until that run has actually happened.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .config import SUPPORTED_LANGUAGES, resolve_source_root
from .labeling import label_function
from .parsers.base_parser import ParseFailure
from .parsers.csharp_parser import CSharpFunctionParser
from .parsers.java_parser import JavaFunctionParser

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PARSERS = {
    "java": JavaFunctionParser,
    "csharp": CSharpFunctionParser,
}


def discover_source_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Recursively find all source files under root with the given extensions."""
    files: list[Path] = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def extract_language(language: str, source_root: Path) -> pd.DataFrame:
    """
    Run extraction for one language across its entire source tree.
    Returns a DataFrame with the unified schema. Files that fail to parse
    are logged and skipped, not silently dropped -- a summary count of
    failures is logged at the end so a bad run is visible, not hidden.
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    lang_config = SUPPORTED_LANGUAGES[language]
    parser = _PARSERS[language]()

    source_files = discover_source_files(source_root, lang_config.file_extensions)
    logger.info("Found %d %s source files under %s", len(source_files), language, source_root)

    if not source_files:
        raise RuntimeError(
            f"Zero {language} source files found under {source_root}. "
            f"Check the path is correct and points at the actual Juliet "
            f"{language} test suite root, not a parent or sibling directory."
        )

    records: list[dict] = []
    failures: list[ParseFailure] = []

    for file_path in source_files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            failures.append(ParseFailure(str(file_path), f"read error: {exc}"))
            continue

        try:
            functions = parser.safe_extract(text, str(file_path))
        except ParseFailure as failure:
            failures.append(failure)
            continue

        for fn in functions:
            try:
                label_result = label_function(fn.function_name, str(file_path), fn.source_code)
            except ValueError as exc:
                # Function name doesn't match bad/good convention -- likely
                # a helper method, not a Juliet test case. Skip, don't guess.
                logger.debug("Skipping unlabelable function: %s", exc)
                continue

            records.append(
                {
                    "source_code": fn.source_code,
                    "function_name": fn.function_name,
                    "sanitized_function_name": label_result.sanitized_name,
                    "cwe_id": label_result.cwe_id,
                    "language": language,
                    "source_file": str(file_path),
                    "label": label_result.label,
                    "start_line": fn.start_line,
                    "end_line": fn.end_line,
                    **{
                        f"leak_token_{tok.lower()}": count
                        for tok, count in label_result.leakage_token_counts.items()
                    },
                }
            )

    if failures:
        logger.warning(
            "%d/%d %s files failed to parse. First 5 failures:",
            len(failures), len(source_files), language,
        )
        for failure in failures[:5]:
            logger.warning("  %s", failure)

    if not records:
        raise RuntimeError(
            f"Extraction for {language} produced zero labeled functions "
            f"from {len(source_files)} source files. This means either the "
            f"parser found no method-like nodes, or none matched the "
            f"bad/good naming convention -- both indicate something is "
            f"wrong with the source root or the parser, not a real result."
        )

    df = pd.DataFrame.from_records(records)
    logger.info(
        "%s extraction complete: %d functions (%d vulnerable, %d safe) from %d files (%d failed)",
        language, len(df), (df["label"] == 1).sum(), (df["label"] == 0).sum(),
        len(source_files), len(failures),
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Java/C# functions from Juliet")
    parser.add_argument("--language", required=True, choices=list(SUPPORTED_LANGUAGES))
    parser.add_argument("--source-dir", default=None, help="Juliet source root (overrides env var)")
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    source_root = resolve_source_root(args.language, args.source_dir)
    df = extract_language(args.language, source_root)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)


if __name__ == "__main__":
    main()
