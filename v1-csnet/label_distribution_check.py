import pandas as pd

train_df = pd.read_parquet(
    "data/splits/train_clean.parquet"
)

print("\nTrain Label Distribution:\n")

print(
    train_df["label"].value_counts(normalize=True)
)