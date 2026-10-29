import pandas as pd

train = pd.read_parquet(
    "data/splits/train_sanitized.parquet"
)

# --------------------------------------------------
# VULNERABLE SAMPLES
# --------------------------------------------------

print("=" * 70)
print("VULNERABLE SAMPLES")
print("=" * 70)

for i, row in train[train["label"] == 1].sample(
    3,
    random_state=42
).iterrows():

    print(
        f"\n--- CWE: {row.get('cwe_id', 'unknown')} "
        f"| fn: {row.get('function_name', 'unknown')}"
    )

    print(row["code"][:500])

# --------------------------------------------------
# SAFE SAMPLES
# --------------------------------------------------

print("\n" + "=" * 70)
print("SAFE SAMPLES")
print("=" * 70)

for i, row in train[train["label"] == 0].sample(
    3,
    random_state=42
).iterrows():

    print(
        f"\n--- CWE: {row.get('cwe_id', 'unknown')} "
        f"| fn: {row.get('function_name', 'unknown')}"
    )

    print(row["code"][:500])