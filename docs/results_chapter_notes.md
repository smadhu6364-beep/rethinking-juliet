
# Results Chapter Notes

# Introduction

This chapter presents the experimental evaluation results for the proposed software vulnerability detection approaches, including classical machine learning baselines, transformer-based models, hybrid architectures, and explainability techniques.

The evaluation focuses on:
- predictive performance
- architectural comparison
- explainability generation
- methodological limitations

---

# Random Forest Baseline

The Random Forest baseline achieved extremely high classification scores on the Juliet dataset.

Although this initially suggests strong predictive performance, the results must be interpreted critically. The Juliet dataset contains synthetic structural regularities that may unintentionally leak label information into machine learning models.

As a result, near-perfect scores may not accurately reflect real-world vulnerability detection capability.

The Random Forest baseline therefore serves primarily as a comparative reference point rather than evidence of robust generalization.

---

# CodeBERT Results

CodeBERT demonstrated strong vulnerability detection performance using transformer-based semantic representations of source code.

The model successfully captured contextual information from vulnerable functions and produced highly accurate predictions on the evaluation split.

The results suggest that pretrained transformer representations can effectively model software security patterns within source code sequences.

---

# GraphCodeBERT Results

GraphCodeBERT achieved near-perfect evaluation metrics on the sanitized Juliet dataset.

The incorporation of graph-aware structural representations may improve sensitivity to vulnerability-relevant code relationships.

However, the perfect evaluation scores must be interpreted cautiously due to the known limitations of synthetic benchmark datasets.

These results therefore demonstrate strong benchmark performance, but do not necessarily imply equivalent real-world robustness.

---

# Hybrid Architecture Results

The hybrid architecture combined semantic embeddings from CodeBERT with structural embeddings from GraphCodeBERT through CLS token concatenation and a lightweight MLP classifier.

The architecture was designed to evaluate whether complementary transformer representations could improve vulnerability-sensitive feature learning.

Initial experiments confirmed that the architecture was operational and trainable within GPU-constrained environments.

Further evaluation is required to determine whether the hybrid approach consistently improves predictive performance over individual transformer baselines.

---

# Explainability Results

The explainability pipeline successfully generated:
- token importance rankings
- line-level suspiciousness scores
- natural language explanations

Attention-based evidence extraction enabled localized explanations for vulnerability predictions.

However, attention-based explanations alone do not guarantee true model faithfulness. Additional evaluation using perturbation-based analysis and human reviewer assessment is therefore required.

---

# Overall Findings

The experiments demonstrate that transformer-based models provide highly effective vulnerability detection capability on the Juliet benchmark.

However, the consistently near-perfect performance across multiple architectures also highlights the importance of critical dataset analysis and realistic evaluation methodology.

The dissertation therefore emphasizes not only predictive accuracy, but also explainability, methodological transparency, and limitations of synthetic evaluation environments.
