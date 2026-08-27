import pandas as pd

print("Loading datasets...")

train = pd.read_parquet(
    "data/splits/train_clean.parquet"
)

val = pd.read_parquet(
    "data/splits/val_clean.parquet"
)

print("Checking overlap...")

overlap = set(train["code"]) & set(val["code"])

print(f"Train samples: {len(train)}")
print(f"Validation samples: {len(val)}")
print(f"Overlap count: {len(overlap)}")

if len(overlap) > 0:

    print("\nWARNING: DATA LEAKAGE DETECTED")

else:

    print("\nGOOD: NO DIRECT TRAIN/VAL OVERLAP")