
"""
Rebuild clean train/validation/test splits
using the correct preprocessing order:

1. sanitize
2. deduplicate
3. split
"""

import re
import pandas as pd

from sklearn.model_selection import train_test_split
from pathlib import Path

RANDOM_SEED = 42

INPUT_FILE = "data/processed/juliet_functions_full.parquet"


def sanitize_code(code: str) -> str:

    code = re.sub(r'CWE\d+_[A-Za-z0-9_]*', 'FUNC', code)

    code = re.sub(r'\bgoodG2B\b', 'FUNC', code)
    code = re.sub(r'\bgoodB2G\b', 'FUNC', code)

    code = re.sub(r'\bgood\b', 'FUNC', code)
    code = re.sub(r'\bbad\b', 'FUNC', code)

    code = re.sub(
        r'\b[A-Za-z0-9_]*(Good|Bad|Sink|Source)\b',
        'FUNC',
        code
    )

    return code


def main():

    print("Loading dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Original rows: {len(df)}")

    # -----------------------------------------
    # SANITIZE FIRST
    # -----------------------------------------

    print("\nSanitizing code...")

    df["code"] = df["code"].apply(sanitize_code)

    # -----------------------------------------
    # DEDUPLICATE AFTER SANITIZATION
    # -----------------------------------------

    print("\nRemoving duplicates...")

    before = len(df)

    df = df.drop_duplicates(subset=["code"])

    after = len(df)

    print(f"Removed duplicates: {before - after}")
    print(f"Remaining rows: {after}")

    # -----------------------------------------
    # SPLIT
    # -----------------------------------------

    print("\nCreating train/val/test splits...")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED
    )

    # -----------------------------------------
    # SAVE
    # -----------------------------------------

    Path("data/splits").mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(
        "data/splits/train_clean.parquet",
        index=False
    )

    val_df.to_parquet(
        "data/splits/val_clean.parquet",
        index=False
    )

    test_df.to_parquet(
        "data/splits/test_clean.parquet",
        index=False
    )

    print("\nDONE")

    print(f"\nTrain: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")


if __name__ == "__main__":
    main()

