import pandas as pd

train = pd.read_parquet(
    "data/splits/train_sanitized.parquet"
)

val = pd.read_parquet(
    "data/splits/val_sanitized.parquet"
)

test = pd.read_parquet(
    "data/splits/test_sanitized.parquet"
)

train_codes = set(train["code"])
val_codes = set(val["code"])
test_codes = set(test["code"])

print(f"Train: {len(train)} rows, {len(train_codes)} unique codes")
print(f"Val:   {len(val)} rows, {len(val_codes)} unique codes")
print(f"Test:  {len(test)} rows, {len(test_codes)} unique codes")

print()

print(f"Train ∩ Val:  {len(train_codes & val_codes)} overlapping codes")
print(f"Train ∩ Test: {len(train_codes & test_codes)} overlapping codes")
print(f"Val ∩ Test:   {len(val_codes & test_codes)} overlapping codes")