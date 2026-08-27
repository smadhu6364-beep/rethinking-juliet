
# Threats to Validity

# Dataset Validity

The experiments primarily relied on the Juliet Test Suite, which is a synthetic vulnerability benchmark.

Although Juliet is widely used in academic research, synthetic datasets may contain structural regularities and hidden artifacts that simplify classification tasks beyond realistic software security conditions.

As a result, near-perfect benchmark performance may not accurately represent real-world vulnerability detection capability.

---

# Leakage and Benchmark Bias

Despite preprocessing and sanitization efforts, some structural leakage may still remain within the dataset.

Machine learning models may learn unintended correlations associated with vulnerable and benign function generation patterns.

This threatens external validity and limits confidence in real-world generalization.

---

# Limited Real-World Evaluation

The dissertation does not include large-scale evaluation on industrial or open-source vulnerability datasets.

Therefore, conclusions are limited primarily to controlled benchmark experimentation.

Further real-world validation would be required to assess deployment suitability.

---

# Computational Constraints

Due to limited GPU resources and time constraints, transformer backbones were initially frozen during hybrid model experimentation.

Although this improved training stability and reduced computational cost, full end-to-end fine-tuning may produce different performance characteristics.

---

# Explainability Limitations

Attention-based explanations do not necessarily guarantee faithful model reasoning.

Attention scores may correlate with important features without fully representing internal decision processes.

The dissertation therefore proposes additional perturbation-based evaluation and human assessment methods to partially address this limitation.

---

# Reviewer Bias

Human utility evaluation may be influenced by reviewer subjectivity and varying levels of software security expertise.

Additionally, one reviewer may be associated with the project itself, introducing potential bias despite structured evaluation criteria.

---

# Reproducibility Constraints

Although experimental configurations and preprocessing steps were documented, results may still vary across:
- hardware environments
- random initialization
- library versions
- dataset preprocessing variations

Efforts were made to improve reproducibility through fixed seeds and explicit configuration reporting.

---

# Scope Limitations

The dissertation focuses primarily on:
- static vulnerability detection
- transformer-based architectures
- attention-based explainability

The work does not include:
- dynamic analysis
- symbolic execution
- graph neural networks
- runtime vulnerability detection

These remain important future research directions.
