"""
Rebuild train/val/test splits with proper deduplication.

This addresses the data leakage discovered in the external stress test:
- 1,794 train/val code overlaps (31% of unique val codes)
- 1,822 train/test code overlaps (32% of unique test codes)

Strategy:
1. Load the full extracted dataset (juliet_functions_full.parquet, 116k functions)
2. Deduplicate on code string (drop duplicates, keep first)
3. Stratified split 80/10/10 by (cwe_id, label) joint distribution
4. Verify zero overlap between splits
5. Save as *_dedup.parquet so we keep the originals intact
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ----- Load -----
print("Loading full dataset...")
df = pd.read_parquet("data/processed/juliet_functions_full.parquet")
print(f"  Loaded {len(df):,} functions")
print(f"  Label distribution: {df['label'].value_counts(normalize=True).round(3).to_dict()}")

# ----- Deduplicate -----
print("\nDeduplicating on code string...")
before = len(df)
df = df.drop_duplicates(subset='code', keep='first').reset_index(drop=True)
after = len(df)
print(f"  Dropped {before - after:,} duplicates ({100*(before-after)/before:.1f}%)")
print(f"  Remaining: {after:,} unique functions")
print(f"  Label distribution after dedup: {df['label'].value_counts(normalize=True).round(3).to_dict()}")

# ----- Build stratification key -----
df['stratify_key'] = df['cwe_id'].astype(str) + "_" + df['label'].astype(str)

# Drop tiny strata (can't split into 3 parts with <3 samples)
group_sizes = df['stratify_key'].value_counts()
small_groups = group_sizes[group_sizes < 10].index
if len(small_groups) > 0:
    n_dropped = df['stratify_key'].isin(small_groups).sum()
    print(f"\nDropping {len(small_groups)} small strata (<3 samples each), {n_dropped} rows total")
    df = df[~df['stratify_key'].isin(small_groups)].reset_index(drop=True)

# ----- Split 80/10/10 stratified -----
print("\nStratified split 80/10/10...")
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df['stratify_key'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['stratify_key'], random_state=42
)

train_df = train_df.drop(columns=['stratify_key']).reset_index(drop=True)
val_df = val_df.drop(columns=['stratify_key']).reset_index(drop=True)
test_df = test_df.drop(columns=['stratify_key']).reset_index(drop=True)

print(f"  Train: {len(train_df):,}  balance: {train_df['label'].value_counts(normalize=True).round(3).to_dict()}")
print(f"  Val:   {len(val_df):,}  balance: {val_df['label'].value_counts(normalize=True).round(3).to_dict()}")
print(f"  Test:  {len(test_df):,}  balance: {test_df['label'].value_counts(normalize=True).round(3).to_dict()}")

# ----- Verify zero overlap -----
print("\nVerifying zero overlap between splits...")
train_codes = set(train_df['code'])
val_codes = set(val_df['code'])
test_codes = set(test_df['code'])

train_val = len(train_codes & val_codes)
train_test = len(train_codes & test_codes)
val_test = len(val_codes & test_codes)

print(f"  Train ∩ Val:  {train_val}")
print(f"  Train ∩ Test: {train_test}")
print(f"  Val ∩ Test:   {val_test}")

assert train_val == 0, f"Train/val overlap: {train_val} — split is broken"
assert train_test == 0, f"Train/test overlap: {train_test} — split is broken"
assert val_test == 0, f"Val/test overlap: {val_test} — split is broken"
print("  All zero. Splits are clean.")

# ----- Save -----
print("\nSaving clean splits...")
train_df.to_parquet("data/splits/train_dedup.parquet", index=False)
val_df.to_parquet("data/splits/val_dedup.parquet", index=False)
test_df.to_parquet("data/splits/test_dedup.parquet", index=False)
print("Done.")
print("\nFiles saved:")
print("  data/splits/train_dedup.parquet")
print("  data/splits/val_dedup.parquet")
print("  data/splits/test_dedup.parquet")