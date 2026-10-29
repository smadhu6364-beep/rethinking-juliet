"""
Find the working parquet and identify where class balance got broken.
"""
import os
import pandas as pd

print("=" * 70)
print("PARQUET INVENTORY")
print("=" * 70)

for root, dirs, files in os.walk("data"):
    for f in sorted(files):
        if f.endswith('.parquet'):
            path = os.path.join(root, f)
            try:
                df = pd.read_parquet(path)
                n = len(df)
                if 'label' in df.columns:
                    n_vuln = int(df['label'].sum())
                    n_safe = n - n_vuln
                    pct_vuln = 100 * n_vuln / n if n > 0 else 0
                    print(f"\n{path}")
                    print(f"  Rows: {n:,}")
                    print(f"  Vulnerable: {n_vuln:,} ({pct_vuln:.2f}%)")
                    print(f"  Safe:       {n_safe:,} ({100-pct_vuln:.2f}%)")
                else:
                    print(f"\n{path}")
                    print(f"  Rows: {n:,}")
                    print(f"  No 'label' column")
            except Exception as e:
                print(f"\n{path}: ERROR — {e}")

print("\n" + "=" * 70)