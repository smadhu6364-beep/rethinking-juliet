"""
Master verification script.
Checks all project components and prints a full status report.
Run from project root: python verify_all.py
"""

import os
import json
import subprocess
import pandas as pd

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
results = []


def check(name, condition, pass_msg, fail_msg):
    status = PASS if condition else FAIL
    msg = pass_msg if condition else fail_msg
    results.append((status, name, msg))
    print(f"{status} {name}: {msg}")


def check_file(name, path, min_size_kb=0):
    exists = os.path.exists(path)
    size_ok = (os.path.getsize(path) // 1024) >= min_size_kb if exists else False
    check(
        name,
        exists and size_ok,
        f"exists ({os.path.getsize(path)//1024} KB)" if exists else "exists",
        f"MISSING at {path}",
    )


print("=" * 65)
print("PROJECT VERIFICATION REPORT")
print("=" * 65)

# --- DATA CHECKS ---
print("\n--- DATA ---")

# Full parquet
try:
    df = pd.read_parquet("data/processed/juliet_functions_full.parquet")
    check(
        "Full Juliet parquet",
        len(df) == 116263,
        f"{len(df):,} rows, balance {df['label'].value_counts(normalize=True).round(3).to_dict()}",
        f"Wrong row count: {len(df):,}, expected 116263",
    )
except Exception as e:
    check("Full Juliet parquet", False, "", str(e))

# Dedup splits
for split in ["train", "val", "test"]:
    path = f"data/splits/{split}_dedup.parquet"
    try:
        df = pd.read_parquet(path)
        expected = {"train": 73295, "val": 9162, "test": 9162}[split]
        vuln_pct = df["label"].mean()
        check(
            f"{split}_dedup.parquet",
            abs(len(df) - expected) < 100 and 0.4 < vuln_pct < 0.7,
            f"{len(df):,} rows, {vuln_pct:.2%} vulnerable",
            f"Wrong: {len(df):,} rows or bad balance {vuln_pct:.2%}",
        )
    except Exception as e:
        check(f"{split}_dedup.parquet", False, "", str(e))

# Zero overlap
try:
    train_codes = set(pd.read_parquet("data/splits/train_dedup.parquet")["code"])
    val_codes = set(pd.read_parquet("data/splits/val_dedup.parquet")["code"])
    test_codes = set(pd.read_parquet("data/splits/test_dedup.parquet")["code"])
    tv = len(train_codes & val_codes)
    tt = len(train_codes & test_codes)
    check("Zero train/val overlap", tv == 0, "0 overlapping codes", f"{tv} codes overlap (LEAKAGE)")
    check("Zero train/test overlap", tt == 0, "0 overlapping codes", f"{tt} codes overlap (LEAKAGE)")
except Exception as e:
    check("Overlap check", False, "", str(e))

# Broken files deleted
broken = [
    "data/splits/train_clean.parquet",
    "data/splits/val_clean.parquet",
    "data/splits/test_clean.parquet",
]
for f in broken:
    check(
        f"Broken file deleted: {os.path.basename(f)}",
        not os.path.exists(f),
        "correctly deleted",
        "STILL EXISTS — delete it",
    )

# Leakage diagnostic
try:
    train = pd.read_parquet("data/splits/train_dedup.parquet")
    train["code_len"] = train["code"].str.len()
    vuln_mean = train[train["label"] == 1]["code_len"].mean()
    safe_mean = train[train["label"] == 0]["code_len"].mean()
    ratio = vuln_mean / safe_mean
    check(
        "Code length gap (4x)",
        ratio > 3.0,
        f"vulnerable={vuln_mean:.0f} chars, safe={safe_mean:.0f} chars, ratio={ratio:.1f}x",
        f"Ratio only {ratio:.1f}x, expected >3.0x",
    )
    vuln_names = set(train[train["label"] == 1]["function_name"])
    safe_names = set(train[train["label"] == 0]["function_name"])
    overlap = len(vuln_names & safe_names)
    check(
        "Zero function name overlap",
        overlap == 0,
        f"0 shared names across {len(vuln_names | safe_names):,} unique names",
        f"{overlap} names appear in both classes",
    )
except Exception as e:
    check("Leakage diagnostic", False, "", str(e))

# --- MODEL CHECKS ---
print("\n--- MODELS ---")

# CodeBERT checkpoint
ckpt = "experiments/codebert-test/checkpoint-1000"
check_file("CodeBERT checkpoint dir", ckpt)

# CodeBERT evaluation results
try:
    r = json.load(open("results/codebert_test_evaluation.json"))
    check(
        "CodeBERT F1 on dedup test",
        r["f1"] >= 0.97,
        f"F1={r['f1']}, Acc={r['accuracy']}, P={r['precision']}, R={r['recall']}",
        f"F1 only {r['f1']}, expected >=0.97",
    )
except Exception as e:
    check("CodeBERT evaluation results", False, "", str(e))

# GraphCodeBERT results
gcb = "results/graphcodebert_evaluation.json"
if os.path.exists(gcb):
    r = json.load(open(gcb))
    check("GraphCodeBERT evaluation", r["f1"] >= 0.90, f"F1={r['f1']}", f"F1 only {r['f1']}")
else:
    results.append((WARN, "GraphCodeBERT evaluation", "Not yet run — train on Renku GPU first"))
    print(f"{WARN} GraphCodeBERT evaluation: Not yet run")

# RF results
rf = "results/rf_baseline_evaluation.json"
if os.path.exists(rf):
    r = json.load(open(rf))
    check("Random Forest evaluation", r["f1"] >= 0.80, f"F1={r['f1']}", f"F1 only {r['f1']}")
else:
    results.append((WARN, "Random Forest evaluation", "Not yet run — run python evaluate_rf.py"))
    print(f"{WARN} Random Forest evaluation: Not yet run")

# --- RESULTS CHECKS ---
print("\n--- RESULTS ---")

result_files = {
    "results/external_eval_results.txt": 1,
    "results/leakage_diagnostics.txt": 1,
    "results/rq3_results.json": 5,
    "results/faithfulness_results.json": 2,
    "results/codebert_test_evaluation.json": 0,
}
for path, min_kb in result_files.items():
    check_file(os.path.basename(path), path, min_kb)

# RQ3 accuracy
try:
    data = json.load(open("results/rq3_results.json"))
    correct = sum(1 for r in data if r["inference"]["prediction"] == r["expected"])
    check(
        "RQ3 pipeline accuracy",
        len(data) == 20,
        f"{correct}/{len(data)} correct ({100*correct/len(data):.1f}%)",
        f"Wrong sample count: {len(data)}, expected 20",
    )
except Exception as e:
    check("RQ3 results", False, "", str(e))

# Faithfulness
try:
    data = json.load(open("results/faithfulness_results.json"))
    deltas = [abs(r.get("confidence_delta", 0)) for r in data if r.get("confidence_delta") is not None]
    avg = sum(deltas) / len(deltas) if deltas else 99
    check(
        "Faithfulness delta near zero",
        avg < 0.01,
        f"Average delta={avg:.6f} across {len(data)} samples",
        f"Average delta={avg:.4f}, expected <0.01",
    )
except Exception as e:
    check("Faithfulness results", False, "", str(e))

# --- SCRIPTS CHECKS ---
print("\n--- SCRIPTS ---")

scripts = [
    "external_eval.py",
    "diagnose_leakage.py",
    "evaluate_codebert.py",
    "evaluate_rf.py",
    "src/preprocessing/rebuild_splits_clean.py",
    "src/models/train_codebert.py",
    "src/models/train_graphcodebert.py",
    "docs/rq3-design.md",
]
for s in scripts:
    check_file(os.path.basename(s), s)

# --- GIT CHECKS ---
print("\n--- GIT ---")

try:
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    check(
        "Correct git branch",
        branch == "indu/rq3-design",
        f"On branch: {branch}",
        f"Wrong branch: {branch}",
    )
    status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    check(
        "Clean working tree",
        status == "",
        "Nothing to commit",
        f"Uncommitted changes: {status[:100]}",
    )
except Exception as e:
    check("Git checks", False, "", str(e))

# --- SUMMARY ---
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

passes = sum(1 for r in results if r[0] == PASS)
fails = sum(1 for r in results if r[0] == FAIL)
warns = sum(1 for r in results if r[0] == WARN)
total = len(results)

print(f"PASS:    {passes}/{total}")
print(f"FAIL:    {fails}/{total}")
print(f"PENDING: {warns}/{total}")

if fails == 0:
    print("\nAll critical checks PASSED. Project is ready for final push.")
else:
    print(f"\n{fails} checks FAILED. Fix before committing to main branch.")
    print("\nFailed checks:")
    for status, name, msg in results:
        if status == FAIL:
            print(f"  {name}: {msg}")

# Save report
report = {
    "total": total,
    "passes": passes,
    "fails": fails,
    "warnings": warns,
    "details": [{"status": s, "check": n, "message": m} for s, n, m in results],
}
with open("results/verification_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nFull report saved to results/verification_report.json")