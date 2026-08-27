"""
End-to-end orchestrator: extraction -> dedup -> split -> overlap check,
for Java and/or C#. Designed to run in Google Colab as the target
environment (per project decision), but has no Colab-specific code in it --
it's a plain script, so it also runs identically on a laptop or cluster if
that changes later.

Colab usage:

    !pip install tree_sitter tree-sitter-java tree-sitter-c-sharp pandas scikit-learn pyarrow

    from google.colab import drive
    drive.mount('/content/drive')

    !python -m preprocessing.multilang.run_pipeline \\
        --languages java csharp \\
        --java-dir /content/drive/MyDrive/juliet/java \\
        --csharp-dir /content/drive/MyDrive/juliet/csharp \\
        --output-root /content/drive/MyDrive/multilang_output

Nothing here writes to, reads from, or depends on any existing C/C++
output path. All outputs land under --output-root, kept entirely separate.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import resolve_source_root
from .dedup_split_multilang import deduplicate, stratified_split, verify_zero_overlap
from .extract_functions_multilang import extract_language

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_for_language(language: str, source_dir: str | None, output_root: Path) -> None:
    logger.info("=" * 60)
    logger.info("Starting pipeline for: %s", language)
    logger.info("=" * 60)

    source_root = resolve_source_root(language, source_dir)
    df = extract_language(language, source_root)

    lang_output_dir = output_root / language
    lang_output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = lang_output_dir / f"{language}_functions_raw.parquet"
    df.to_parquet(raw_path, index=False)
    logger.info("Wrote raw extraction: %s (%d rows)", raw_path, len(df))

    deduped = deduplicate(df)
    splits = stratified_split(deduped)
    verify_zero_overlap(splits)

    for name, split_df in splits.items():
        out_path = lang_output_dir / f"{name}.parquet"
        split_df.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(split_df))

    logger.info("Completed %s: %d raw -> %d deduped -> "
                "train=%d val=%d test=%d",
                language, len(df), len(deduped),
                len(splits["train"]), len(splits["val"]), len(splits["test"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Java/C# extraction pipeline")
    parser.add_argument("--languages", nargs="+", required=True, choices=["java", "csharp"])
    parser.add_argument("--java-dir", default=None)
    parser.add_argument("--csharp-dir", default=None)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    dir_map = {"java": args.java_dir, "csharp": args.csharp_dir}

    results = {}
    for language in args.languages:
        try:
            run_for_language(language, dir_map[language], output_root)
            results[language] = "SUCCESS"
        except Exception as exc:
            logger.error("Pipeline FAILED for %s: %s", language, exc)
            results[language] = f"FAILED: {exc}"

    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for language, status in results.items():
        logger.info("  %s: %s", language, status)


if __name__ == "__main__":
    main()
