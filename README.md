# Rethinking Juliet: label validity, structural shortcuts and transfer

Code and data for the revised IEEE Access manuscript *Rethinking Juliet: Label Validity, Structural Shortcuts, and Transfer in Learned Vulnerability Detection* (K. Jadhav and M. Sudhan). A preliminary C/C++-only version appeared at CSNet 2026.

## What is here

| Path | Contents |
|---|---|
| `code/extract_v2.py` | Brace-matching function scanner and labelling against Juliet's `manifest.xml` flaw lines |
| `code/build_splits.py`, `code/sanitize.py` | Sanitisation, deduplication (label-conflict and exact), random and family-disjoint 80/10/10 splits |
| `code/experiments_cpu.py`, `code/rf_external.py` | Length-only and TF-IDF Random Forest baselines, shortcut controls, near-duplicate measurement, Random Forest on Devign |
| `code/finetune.py`, `code/hybrid.py`, `code/rq3.py`, `code/kaggle_run.py` | CodeBERT/GraphCodeBERT fine-tuning, frozen-encoder hybrid, attention-masking test (run on Kaggle 2×T4) |
| `code/truncation_and_dedup.py` | Flaw-line position vs the 256-token limit; deduplication accounting |
| `code/make_tables.py` | Builds every number and table in the paper from `results/` |
| `data/audit_*_by_cwe_role.csv` | Every scanned function by CWE, role and outcome (vulnerable, safe, or excluded with reason) |
| `data/coverage_*_per_cwe.csv` | Vulnerable and safe counts per CWE in each split |
| `data/split_ids_*.csv` | Split membership of every function: file, function name, line span, label, CWE, family |
| `data/ood35.json` | The 35-function diagnostic suite |
| `results/*.json` | Raw results for every run (seeds, confusion matrices, Devign, diagnostic suite, masking) |

## Reproducing

1. Download Juliet C/C++ v1.3 and Java v1.3 from the NIST SARD and point `RAW` in `code/extract_v2.py` at them.
2. `python extract_v2.py cpp java` then `python build_splits.py cpp java`.
3. `python experiments_cpu.py cpp java` and `python rf_external.py` (CPU).
4. Fine-tuning: `python finetune.py codebert cpp family 0 --epochs 3` etc., or run `kaggle_run.py` on a 2-GPU Kaggle notebook (about 5 hours).
5. `python make_tables.py` regenerates the paper's numbers.

Devign test data: CodeXGLUE defect-detection test split (`google/code_x_glue_cc_defect_detection` on Hugging Face).

Juliet C# is not included: NIST's C# release has no flaw manifest, so its labels could not be audited to the same standard.

## Earlier version

`v1-csnet/` holds the code of the preliminary CSNet 2026 study (regex extraction, random split). It is kept for reference only; the revised study does not use it.
