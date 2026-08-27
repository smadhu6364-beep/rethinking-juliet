"""
Create deduplicated train/validation/test splits
for vulnerability detection experiments.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RANDOM_SEED = 42

INPUT_PATH = "data/processed/juliet_functions_full.parquet"

TRAIN_OUT = "data/splits/train.parquet"
VAL_OUT = "data/splits/val.parquet"
TEST_OUT = "data/splits/test.parquet"


def main():

    print("Loading dataset...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Original samples: {len(df)}")

    # -----------------------------------------
    # Remove exact duplicate code samples
    # -----------------------------------------

    df = df.drop_duplicates(subset=["code"])

    print(f"After deduplication: {len(df)}")

    # -----------------------------------------
    # Train/temp split
    # 70% train
    # 30% temp
    # -----------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )

    # -----------------------------------------
    # Split temp into validation + test
    # 15% validation
    # 15% test
    # -----------------------------------------

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED
    )

    # -----------------------------------------
    # Create output directory
    # -----------------------------------------

    Path("data/splits").mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # Save splits
    # -----------------------------------------

    train_df.to_parquet(TRAIN_OUT, index=False)
    val_df.to_parquet(VAL_OUT, index=False)
    test_df.to_parquet(TEST_OUT, index=False)

    # -----------------------------------------
    # Final stats
    # -----------------------------------------

    print("\nDONE\n")

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    print("\nLabel distribution:")

    print("\nTRAIN")
    print(train_df["label"].value_counts(normalize=True))

    print("\nVALIDATION")
    print(val_df["label"].value_counts(normalize=True))

    print("\nTEST")
    print(test_df["label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()

