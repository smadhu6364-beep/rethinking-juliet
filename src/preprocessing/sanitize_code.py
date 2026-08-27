
"""
Sanitize Juliet code samples to reduce label leakage.
"""

import re
import pandas as pd

INPUT_FILES = {
    "train": "data/splits/train.parquet",
    "val": "data/splits/val.parquet",
    "test": "data/splits/test.parquet"
}

OUTPUT_FILES = {
    "train": "data/splits/train_sanitized.parquet",
    "val": "data/splits/val_sanitized.parquet",
    "test": "data/splits/test_sanitized.parquet"
}


def sanitize_code(code: str) -> str:

    # -----------------------------------------
    # Remove CWE identifiers
    # -----------------------------------------

    code = re.sub(r'CWE\d+_[A-Za-z0-9_]*', 'FUNC', code)

    # -----------------------------------------
    # Remove obvious label words
    # -----------------------------------------

    code = re.sub(r'\bgoodG2B\b', 'FUNC', code)
    code = re.sub(r'\bgoodB2G\b', 'FUNC', code)

    code = re.sub(r'\bgood\b', 'FUNC', code)
    code = re.sub(r'\bbad\b', 'FUNC', code)

    # -----------------------------------------
    # Remove helper variants
    # -----------------------------------------

    code = re.sub(r'\b[A-Za-z0-9_]*(Good|Bad|Sink|Source)\b', 'FUNC', code)

    return code


def process_split(name):

    print(f"\nProcessing {name}...")

    df = pd.read_parquet(INPUT_FILES[name])

    df["code"] = df["code"].apply(sanitize_code)

    df.to_parquet(OUTPUT_FILES[name], index=False)

    print(f"Saved: {OUTPUT_FILES[name]}")


def main():

    for split in ["train", "val", "test"]:
        process_split(split)

    print("\nDONE")


if __name__ == "__main__":
    main()

