"""
Sanitize, deduplicate and split the re-extracted Juliet functions, and write
the label-audit tables reviewers asked for.

Order (one pipeline, used for every result in the revision):
  1. extract + label (extract_v2.py)
  2. sanitize: strip comments, replace bad/good/G2B/B2G identifiers and CWE
     names with FUNC (canonical sanitizer from the original repo)
  3. normalise whitespace; exact-dedup on the sanitized text, i.e. the text
     the models see; drop texts that occur with both labels
  4. split 80/10/10, two ways:
       random  - stratified by (CWE, label), the protocol of the original paper
       family  - whole Juliet functional variants (all flow variants and
                 files of one template) assigned to one partition, per CWE

Writes out/<lang>_{random,family}_{train,val,test}.parquet and audit CSVs.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sanitize import sanitize_source_code  # copy of src/preprocessing/sanitize.py

OUT = Path(__file__).resolve().parent / "out"
SEED = 42


def normalise(code: str) -> str:
    code = sanitize_source_code(code)
    return re.sub(r"\s+", " ", code).strip()


def split_random(df: pd.DataFrame, rng) -> pd.Series:
    part = pd.Series("train", index=df.index)
    for _, g in df.groupby(["cwe", "label"]):
        idx = rng.permutation(g.index.to_numpy())
        n = len(idx)
        n_te = int(round(n * 0.1))
        n_va = int(round(n * 0.1))
        part[idx[:n_te]] = "test"
        part[idx[n_te:n_te + n_va]] = "val"
    return part


def split_family(df: pd.DataFrame, rng) -> pd.Series:
    """Assign whole families to partitions, CWE by CWE, filling test then
    val to ~10% of that CWE's functions each."""
    part = pd.Series("train", index=df.index)
    for _, g in df.groupby("cwe"):
        fams = g.groupby("family").size()
        order = rng.permutation(fams.index.to_numpy())
        total = len(g)
        acc, target = 0, {"test": 0.1 * total, "val": 0.1 * total}
        assign = {}
        for f in order:
            if acc < target["test"]:
                assign[f] = "test"
            elif acc < target["test"] + target["val"]:
                assign[f] = "val"
            else:
                assign[f] = "train"
            acc += fams[f]
        # CWEs with a single family stay in train (cannot be held out)
        if len(fams) < 3:
            assign = {f: "train" for f in fams.index}
        part[g.index] = g["family"].map(assign).to_numpy()
    return part


def build(lang: str) -> None:
    df = pd.read_parquet(OUT / f"functions_{lang}.parquet")

    # ---- audit: every scanned function by CWE / role / outcome ----
    df["outcome"] = np.where(df.label.isna(), "excluded: " + df.reason,
                             np.where(df.label.fillna(-1) == 1, "vulnerable", "safe"))
    audit = (df.groupby(["cwe", "role", "outcome"]).size()
               .rename("functions").reset_index())
    audit.to_csv(OUT / f"audit_{lang}_by_cwe_role.csv", index=False)

    lab = df[df.label.notna()].copy()
    lab["label"] = lab["label"].astype(int)
    lab["code"] = lab.code_raw.map(normalise)
    lab["len_raw"] = lab.code_raw.str.len()
    lab["len"] = lab.code.str.len()

    # ---- exact dedup on sanitized text ----
    n0 = len(lab)
    labels_per_text = lab.groupby("code")["label"].nunique()
    conflicted = labels_per_text[labels_per_text > 1].index
    n_conf = lab.code.isin(conflicted).sum()
    lab = lab[~lab.code.isin(conflicted)]
    lab = lab.drop_duplicates("code", keep="first")
    stages = dict(lang=lang, scanned=len(df), labelled=n0,
                  removed_label_conflict=int(n_conf),
                  unique_after_dedup=len(lab),
                  vulnerable=int(lab.label.sum()),
                  safe=int((lab.label == 0).sum()))

    rng = np.random.default_rng(SEED)
    lab["split_random"] = split_random(lab, rng)
    lab["split_family"] = split_family(lab, np.random.default_rng(SEED))

    keep = ["code", "code_raw", "label", "cwe", "family", "file",
            "function_name", "role", "len", "len_raw"]
    for scheme in ("random", "family"):
        for p in ("train", "val", "test"):
            sub = lab[lab[f"split_{scheme}"] == p][keep]
            sub.reset_index(drop=True).to_parquet(
                OUT / f"{lang}_{scheme}_{p}.parquet", index=False)
            stages[f"{scheme}_{p}_n"] = len(sub)
            stages[f"{scheme}_{p}_vuln_pct"] = round(100 * sub.label.mean(), 2)
        # overlap checks: exact text and family
        tr = set(lab[lab[f"split_{scheme}"] == "train"].code)
        te = lab[lab[f"split_{scheme}"] == "test"]
        stages[f"{scheme}_text_overlap"] = int(te.code.isin(tr).sum())
        trf = set(lab[lab[f"split_{scheme}"] == "train"].family)
        stages[f"{scheme}_test_in_train_family_pct"] = round(
            100 * te.family.isin(trf).mean(), 2)

    # ---- per-CWE coverage in final splits ----
    cov = (lab.groupby(["cwe", "split_random", "label"]).size()
              .unstack(["split_random", "label"], fill_value=0))
    cov.columns = [f"{s}_{'vuln' if l else 'safe'}" for s, l in cov.columns]
    cov = cov.reindex(sorted(cov.columns), axis=1)
    cov.to_csv(OUT / f"coverage_{lang}_per_cwe.csv")
    all_cwes = sorted(df.cwe.unique())
    stages["cwes_in_suite"] = len(all_cwes)
    stages["cwes_with_vuln_after_dedup"] = int((lab[lab.label == 1].cwe.nunique()))
    stages["cwes_with_safe_after_dedup"] = int((lab[lab.label == 0].cwe.nunique()))
    stages["cwes_both_classes"] = int(len(set(lab[lab.label == 1].cwe) &
                                          set(lab[lab.label == 0].cwe)))
    te = lab[lab.split_random == "test"]
    stages["cwes_with_vuln_in_random_test"] = int(te[te.label == 1].cwe.nunique())
    pd.Series(stages).to_json(OUT / f"stages_{lang}.json", indent=1)
    print(pd.Series(stages).to_string())


if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["cpp", "java", "csharp"]):
        build(lang)
