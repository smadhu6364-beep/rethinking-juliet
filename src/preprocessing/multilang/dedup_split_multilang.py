"""
Deduplication and splitting, matching the methodology the paper states for
C/C++ (Section III-C): exact-string deduplication before splitting,
stratified sampling over (CWE, label) pairs to preserve class balance
across partitions, 80/10/10 train/val/test, overlap verified
programmatically afterward -- not assumed.

Using the identical procedure here is the whole point: if this project
later reports "Java shows the same artefacts as C/C++", that claim is only
credible if the split methodology that produced the C/C++ result wasn't
also different in some way that could explain the similarity.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Exact-string dedup on source_code, same rule as C/C++. Keeps the
    first occurrence, logs how many rows were dropped."""
    before = len(df)
    deduped = df.drop_duplicates(subset="source_code", keep="first").reset_index(drop=True)
    after = len(deduped)
    logger.info("Deduplication: %d -> %d rows (%d exact duplicates removed)",
                before, after, before - after)
    return deduped


def stratified_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    80/10/10 stratified split over (CWE, label) pairs -- identical target
    ratios to the C/C++ pipeline. Falls back to stratifying on label alone
    for any (CWE, label) group too small to split three ways, and logs
    every such fallback so it's visible, not silent.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9

    df = df.copy()
    df["_strata"] = df["cwe_id"].fillna("NOCWE").astype(str) + "_" + df["label"].astype(str)

    strata_counts = df["_strata"].value_counts()
    tiny_strata = strata_counts[strata_counts < 3].index.tolist()
    if tiny_strata:
        logger.warning(
            "%d (CWE, label) strata have fewer than 3 samples and cannot "
            "be stratified three ways; falling back to label-only "
            "stratification for those rows: %s",
            len(tiny_strata), tiny_strata[:10],
        )
        df.loc[df["_strata"].isin(tiny_strata), "_strata"] = (
            "SMALL_" + df.loc[df["_strata"].isin(tiny_strata), "label"].astype(str)
        )

    train_df, temp_df = train_test_split(
        df, train_size=train_frac, stratify=df["_strata"], random_state=random_state
    )

    temp_df = temp_df.copy()
    temp_strata_counts = temp_df["_strata"].value_counts()
    temp_tiny_strata = temp_strata_counts[temp_strata_counts < 2].index.tolist()
    if temp_tiny_strata:
        logger.warning(
            "%d strata have fewer than 2 samples in the val/test partition "
            "(post first split) and cannot be stratified two ways; falling "
            "back to label-only stratification for those rows: %s",
            len(temp_tiny_strata), temp_tiny_strata[:10],
        )
        temp_df.loc[temp_df["_strata"].isin(temp_tiny_strata), "_strata"] = (
            "SMALL_" + temp_df.loc[temp_df["_strata"].isin(temp_tiny_strata), "label"].astype(str)
        )

    relative_val_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, train_size=relative_val_frac, stratify=temp_df["_strata"], random_state=random_state
    )

    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        logger.info(
            "%s: %d rows (%.2f%% vulnerable)",
            name, len(split_df), 100 * (split_df["label"] == 1).mean(),
        )

    return {
        "train": train_df.drop(columns="_strata").reset_index(drop=True),
        "val": val_df.drop(columns="_strata").reset_index(drop=True),
        "test": test_df.drop(columns="_strata").reset_index(drop=True),
    }


def verify_zero_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """
    Programmatic overlap check -- do not just assume the split is clean.
    Raises if any exact-string overlap is found between any two splits,
    since a silent failure here would reproduce exactly the kind of
    contamination Hamilton is questioning for the C/C++ results.
    """
    train_set = set(splits["train"]["source_code"])
    val_set = set(splits["val"]["source_code"])
    test_set = set(splits["test"]["source_code"])

    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    if train_val or train_test or val_test:
        raise RuntimeError(
            f"Split overlap detected -- train/val: {len(train_val)}, "
            f"train/test: {len(train_test)}, val/test: {len(val_test)}. "
            f"Splits are NOT clean. Do not proceed to training on this data."
        )
    logger.info("Overlap check passed: zero exact-string overlap between all three splits.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedup and split extracted function dataset")
    parser.add_argument("--input", required=True, help="Path to extracted parquet (pre-dedup)")
    parser.add_argument("--output-dir", required=True, help="Directory to write train/val/test parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = deduplicate(df)
    splits = stratified_split(df)
    verify_zero_overlap(splits)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        out_path = output_dir / f"{name}.parquet"
        split_df.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(split_df))


if __name__ == "__main__":
    main()
