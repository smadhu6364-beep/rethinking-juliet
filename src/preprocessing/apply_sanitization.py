"""
One-off script: apply src/preprocessing/sanitize.py's leakage fix to the
already-extracted raw parquet files for all three languages (C/C++, Java,
C#), in place, and regenerate sanitized_function_name for Java/C# with the
updated (G2B/B2G-aware) pattern.

Does NOT re-run the underlying parsers against the raw Juliet source
trees -- extraction itself (function boundaries, labels, CWE ids) is
unaffected by this fix, only the leaking text within already-extracted
functions changes. Re-running the parsers would be strictly slower for
the same result.

The original, unsanitized text is preserved in a new *_raw column in each
file (not discarded) so the before/after leak measurement stays available
as evidence.

Run once from the repo root: python -m src.preprocessing.apply_sanitization
Then re-run rebuild_splits_clean.py and dedup_split_multilang.py to
regenerate the train/val/test splits from this cleaned data.
"""

import shutil
from pathlib import Path

import pandas as pd

from src.preprocessing.sanitize import sanitize_source_code
from src.preprocessing.multilang.labeling import sanitize_name


def backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".BEFORE_leakage_fix")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
        print(f"  Backed up original to {backup_path}")
    else:
        print(f"  Backup already exists at {backup_path}, not overwriting")


def sanitize_cpp() -> None:
    path = Path("data/processed/juliet_functions_full.parquet")
    print(f"\n=== C/C++: {path} ===")
    backup(path)

    df = pd.read_parquet(path)
    print(f"  Rows: {len(df)}")

    df["code_raw"] = df["code"]
    df["code"] = df["code"].apply(sanitize_source_code)

    df.to_parquet(path, index=False)
    print(f"  Sanitized 'code' column written back to {path}")


def sanitize_multilang(language: str) -> None:
    path = Path(f"data/processed/multilang/{language}/{language}_functions_raw.parquet")
    print(f"\n=== {language}: {path} ===")
    backup(path)

    df = pd.read_parquet(path)
    print(f"  Rows: {len(df)}")

    df["source_code_raw"] = df["source_code"]
    df["source_code"] = df["source_code"].apply(sanitize_source_code)
    df["sanitized_function_name"] = df["function_name"].apply(sanitize_name)

    df.to_parquet(path, index=False)
    print(f"  Sanitized 'source_code' + 'sanitized_function_name' written back to {path}")


def main() -> None:
    sanitize_cpp()
    sanitize_multilang("java")
    sanitize_multilang("csharp")
    print("\nDone. Next: re-run rebuild_splits_clean.py and dedup_split_multilang.py "
          "(both languages) to regenerate train/val/test splits from this cleaned data.")


if __name__ == "__main__":
    main()
