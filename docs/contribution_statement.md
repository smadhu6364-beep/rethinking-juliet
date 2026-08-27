
# Dissertation Contribution Statement

# Core Research Contribution

This dissertation proposes a hybrid transformer-based software vulnerability detection framework that combines semantic and structural source code representations with explainability mechanisms.

The work integrates:
- CodeBERT semantic embeddings
- GraphCodeBERT structural embeddings
- hybrid representation fusion
- attention-based explainability techniques

within a unified experimental framework.

---

# Technical Contributions

## 1. Hybrid Transformer Architecture

A dual-transformer architecture was implemented that combines:
- CodeBERT
- GraphCodeBERT

through CLS embedding concatenation and an MLP classifier head.

The architecture was designed to evaluate whether complementary semantic and structural representations improve vulnerability-sensitive feature learning.

---

## 2. Explainability Pipeline

An explainability framework was developed that supports:
- attention extraction
- token importance ranking
- line-level suspiciousness scoring
- natural language explanation generation

This enables localized interpretation of vulnerability predictions.

---

## 3. Dataset Sanitization Pipeline

A preprocessing pipeline was implemented to:
- extract functions
- sanitize vulnerable identifiers
- reduce direct label leakage
- create reproducible dataset splits

This supports more controlled experimental evaluation.

---

## 4. Critical Evaluation of Benchmark Performance

The dissertation critically examines the limitations of synthetic vulnerability datasets such as Juliet.

The work argues that near-perfect benchmark performance may not imply strong real-world generalization due to structural regularities and hidden leakage effects.

This contributes methodological transparency to vulnerability detection evaluation.

---

# Academic Contribution

The dissertation contributes an experimentally grounded study of:
- transformer-based vulnerability detection
- hybrid representation learning
- explainable AI for software security
- benchmark evaluation limitations

within the context of secure software engineering research.

---

# Practical Contribution

The implemented framework demonstrates how modern transformer architectures and explainability techniques can be integrated into automated vulnerability analysis pipelines.

The modular architecture also supports future extension for:
- real-world datasets
- additional code models
- graph neural integration
- advanced explainability methods

