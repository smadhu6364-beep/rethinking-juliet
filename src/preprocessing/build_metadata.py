import pandas as pd
from pathlib import Path
import re


ROOT = Path("data/raw/juliet-cpp")


def extract_cwe(filepath: str):

    match = re.search(r'CWE(\d+)', filepath)

    if match:
        return f"CWE-{match.group(1)}"

    return None


rows = []

for file in ROOT.rglob("*.*"):

    if file.suffix not in [".c", ".cpp"]:
        continue

    cwe = extract_cwe(str(file))

    if cwe is None:
        continue

    rows.append({
        "file_path": str(file),
        "cwe_id": cwe
    })


df = pd.DataFrame(rows)

print(f"Files found: {len(df)}")


print(df.head())

output_path = "data/juliet_metadata_fixed.csv"

df.to_csv(output_path, index=False)

print(f"\nMetadata saved to: {output_path}")