"""
Extract individual C/C++ functions from Juliet source files
and label them based on Juliet's naming convention.

Usage:
    python src/preprocessing/extract_functions.py \
        --metadata data/juliet_metadata_fixed.csv \
        --output data/processed/juliet_functions_sample.parquet \
        --sample 1000

Output columns:
    code
    label
    cwe_id
    function_name
    source_file
"""

import pandas as pd
import re
import json
import argparse
from pathlib import Path


def label_from_name(fn_name: str) -> int | None:
    """
    Juliet naming convention:

    *_bad      -> vulnerable
    *_good*    -> safe (includes goodG2B/goodB2G variants)
    """

    # Skip helper functions first
    if re.search(r'(Source|Sink|Helper)$', fn_name):
        return None

    if re.search(r'\d+(Source|Sink)$', fn_name):
        return None

    # Main Juliet rules -- check "bad" before "good" so goodG2B/goodB2G
    # (NIST-documented safe variants: a fixed/good half paired with a
    # bad-looking half, to test whether a detector is fooled by
    # superficial resemblance to bad()) are correctly labeled safe,
    # not vulnerable. Matches the convention already correct in
    # src/preprocessing/multilang/labeling.py.
    if fn_name.endswith('_bad'):
        return 1

    if '_good' in fn_name or 'goodG2B' in fn_name or 'goodB2G' in fn_name:
        return 0

    return None


def extract_functions_from_file(filepath: str) -> list[dict]:
    """
    Extract C/C++ functions from a source file.
    """

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

    except Exception:
        return []

    pattern = re.compile(
        r'(?:static\s+)?[\w\s\*]+?\s+'
        r'(\w+)\s*\([^)]*\)\s*'
        r'\{((?:[^{}]|\{[^{}]*\})*)\}',
        re.MULTILINE | re.DOTALL
    )

    functions = []

    for match in pattern.finditer(content):

        functions.append({
            'function_name': match.group(1),
            'code': match.group(0)
        })

    return functions


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sample', type=int, default=None)

    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)

    if args.sample:
        metadata = metadata.head(args.sample)

    print(f"Processing {len(metadata)} files...")

    rows = []

    skipped = 0

    for i, row in metadata.iterrows():

        functions = extract_functions_from_file(row['file_path'])

        for fn in functions:

            label = label_from_name(fn['function_name'])

            if label is None:
                skipped += 1
                continue

            rows.append({
                'code': fn['code'],
                'label': label,
                'cwe_id': row['cwe_id'],
                'function_name': fn['function_name'],
                'source_file': row['file_path']
            })

        if (i + 1) % 100 == 0:
            print(f"{i + 1} files processed")

    df = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(args.output, index=False)

    print("\nDONE")
    print(f"Functions extracted: {len(df)}")
    print(f"Skipped helpers: {skipped}")

    print(df.head())

    # -----------------------------------------
    # Save extraction stats to a separate file
    # -----------------------------------------

    stats = {
        "metadata": args.metadata,
        "output": args.output,
        "files_processed": len(metadata),
        "functions_extracted": len(df),
        "skipped_helpers": skipped,
        "label_counts": {int(k): int(v) for k, v in df["label"].value_counts().items()} if len(df) else {},
    }

    stats_path = Path(args.output).with_suffix(".stats.json")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nExtraction stats saved to {stats_path}")

if __name__ == '__main__':
       main()