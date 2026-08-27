
# Model Comparison Results

Labels were corrected on 2026-07-07 (a bug labeled Juliet's `goodG2B`/`goodB2G` safe
naming variants as vulnerable). Random Forest and CodeBERT below are re-evaluated
against the corrected `test_dedup.parquet`. GraphCodeBERT and Hybrid have no trained
checkpoint available to re-run and still show their pre-fix numbers — treat those two
rows as provisional until retrained and re-evaluated.

| Model | Accuracy | Precision | Recall | F1 Score | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | Verified post-fix | TF-IDF baseline |
| CodeBERT | 0.550 | 0.257 | 1.000 | 0.409 | Verified post-fix* | Semantic transformer |
| GraphCodeBERT | 1.0 | 1.0 | 1.0 | 1.0 | **Provisional (pre-fix)** | Graph-aware transformer |
| Hybrid Model | 1.0 | 1.0 | 1.0 | 1.0 | **Provisional (pre-fix)** | Semantic + structural fusion |

\* The CodeBERT checkpoint itself was trained before the fix, on the old
(pre-fix) labels, and then evaluated against the corrected test set — so this
number reflects a train/test label distribution shift, not a fully controlled
retrain. A CodeBERT model retrained on the corrected splits has not been produced
yet (see `results/codebert_test_evaluation.BEFORE_label_fix.json` for the stale
pre-fix number that is still cited in older paper drafts).

---

# Preliminary Analysis

## Random Forest
The Random Forest baseline achieved extremely high scores on the Juliet dataset both
before and after the label fix. This is consistent with the paper's central claim
that surface statistics (code length, TF-IDF token frequencies) alone explain
benchmark-level performance on Juliet, independent of labeling correctness.

## CodeBERT
Pre-fix, CodeBERT appeared to match the Random Forest ceiling (F1 = 0.979). Re-evaluated
against corrected labels, its (pre-fix-trained) checkpoint collapses to F1 = 0.409 —
below even the length-only classifier. This undercuts the idea that CodeBERT learned
genuine semantic understanding transferable to correctly labeled data; a controlled
retrain on the corrected splits is needed to know whether a freshly trained CodeBERT
would do better.

## GraphCodeBERT
GraphCodeBERT achieved perfect evaluation scores on the pre-fix experimental split, but
has not yet been retrained or re-evaluated against the corrected labels. Whether it
would hold up the way Random Forest did, or collapse the way CodeBERT did, is unknown
and is the most important open item before RQ2 can be answered on verified data.

## Hybrid Model
The hybrid architecture combines semantic representations from CodeBERT with structural
representations from GraphCodeBERT through CLS embedding concatenation and an MLP
classifier head. Like GraphCodeBERT, its reported F1 = 1.000 predates the label fix and
has not been re-verified.

---

# Key Dissertation Argument

Higher benchmark performance does not necessarily imply stronger real-world vulnerability detection capability. Synthetic datasets may contain hidden regularities that inflate evaluation metrics.

The dissertation therefore evaluates not only predictive performance, but also explainability, robustness, and architectural design considerations.
