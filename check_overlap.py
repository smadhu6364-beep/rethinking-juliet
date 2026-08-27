
import pandas as pd

train = pd.read_parquet("data/splits/train_clean.parquet")
test = pd.read_parquet("data/splits/test_clean.parquet")

train_codes = set(train["code"])
test_codes = set(test["code"])

overlap = train_codes & test_codes

print(f"Train rows: {len(train)}")
print(f"Test rows: {len(test)}")

print(f"Unique train codes: {len(train_codes)}")
print(f"Unique test codes: {len(test_codes)}")

print(f"\nOVERLAP: {len(overlap)} samples")

