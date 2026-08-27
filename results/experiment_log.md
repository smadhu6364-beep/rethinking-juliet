```md id="h8qv2m"
# Experiment Log

## Project
Hybrid LLM-Based Software Vulnerability Detection and Explanation Framework

---

# Dataset Preprocessing

## Initial Extraction

- Source dataset: Juliet Test Suite C/C++
- Extracted functions: 116,263
- CWE categories: 113
- Initial label distribution:
  - Vulnerable: 77,563
  - Safe: 38,700

---

## Duplicate Analysis

- Exact duplicate rate:
  - 24,586 duplicated samples
  - ~21.15% duplicate ratio

Observation:
Juliet contains substantial structural repetition across generated test cases.

---

## Leakage Investigation

### Initial Pipeline

Pipeline:
extract → deduplicate → split → sanitize

Problem:
Sanitization introduced new duplicate samples across train and test sets.

Verification result:
- Overlap across train/test:
  - 1,822 samples

Conclusion:
The initial preprocessing pipeline caused post-sanitization leakage.

---

## Sanitization-Aware Deduplication

Pipeline tested:
extract → sanitize → deduplicate → split

Results:
- Remaining samples:
  - ~29k
- Severe class collapse observed
- Safe-class samples almost disappeared

Observation:
Aggressive normalization collapsed many semantically distinct safe implementations into identical forms.

Conclusion:
Sanitization-aware deduplication reduced leakage but destroyed semantic diversity and label balance.

---

## Final Preprocessing Decision

Final adopted pipeline:
- Function extraction
- Helper exclusion
- Raw-code deduplication
- Stratified splitting
- Optional sanitization during modeling

Rationale:
This preserved dataset size and class diversity while acknowledging residual synthetic structural regularities.

---

# Random Forest Baseline

## Features
- TF-IDF
- 1-2 gram features

## Observation

Initial near-perfect scores were likely inflated by dataset regularities and synthetic structure.

---

# CodeBERT Experiment

## Environment
- HuggingFace Transformers
- PyTorch
- Google Colab T4 GPU

## Training Setup
- Model:
  - microsoft/codebert-base
- Epochs:
  - 1
- Batch size:
  - 8
- Max length:
  - 256

## Result

Validation/Test metrics:
- Accuracy: 1.0
- Precision: 1.0
- Recall: 1.0
- F1-score: 1.0

Observation:
CodeBERT learned Juliet structural patterns extremely effectively.

---

# Attention Analysis

## Token-Level Attention

Observation:
Raw attention heavily focused on syntax and formatting tokens:
- braces
- punctuation
- structural tokens

rather than vulnerability APIs.

---

## Line-Level Attention

Observation:
Line-level aggregation still demonstrated strong structural bias.

Conclusion:
Raw transformer attention alone is insufficient for reliable human-readable vulnerability explanations.

---

# Explanation Pipeline

Prototype pipeline implemented:

Code
→ attention extraction
→ suspicious lines
→ explanation generation

## Observation

Natural-language explanation generation produced substantially more interpretable outputs than raw attention visualization alone.

---

# Key Research Insight

The project revealed a trade-off between:
- leakage reduction
- semantic preservation
- explanation interpretability

within synthetic vulnerability benchmarks such as Juliet.

---

# Next Steps

- GraphCodeBERT fine-tuning
- Hybrid model implementation
- Attention filtering refinement
- LLM API integration
- Human evaluation setup
```
