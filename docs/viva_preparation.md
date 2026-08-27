
# Viva Preparation Notes

# Likely Examiner Questions

---

# Why did multiple models achieve near-perfect scores?

The Juliet dataset contains synthetic structural regularities and potential leakage artifacts that may simplify classification beyond realistic real-world conditions.

Although sanitization reduced direct leakage, benchmark inflation likely remains present.

The dissertation therefore critically interprets these results rather than claiming real-world robustness.

---

# Why use both CodeBERT and GraphCodeBERT?

CodeBERT primarily captures semantic and contextual information from source code sequences.

GraphCodeBERT incorporates graph-aware structural representations.

The hybrid architecture was designed to evaluate whether combining semantic and structural representations improves vulnerability-sensitive feature learning.

---

# Why freeze the transformer backbones?

Freezing the backbones:
- reduced GPU memory usage
- stabilized experimentation
- enabled faster iteration
- preserved pretrained representations

This design decision was appropriate for MSc-scale experimentation within limited computational resources.

---

# Why use the Juliet dataset despite limitations?

Juliet remains one of the most widely used vulnerability detection benchmarks in academic literature.

The dissertation explicitly acknowledges its limitations and critically discusses benchmark leakage and generalization concerns.

The goal was controlled comparative experimentation rather than claiming production-ready deployment.

---

# Why use attention-based explainability?

Attention extraction provides localized evidence signals that can identify suspicious tokens and code regions contributing to predictions.

However, the dissertation also acknowledges that attention alone does not guarantee explanation faithfulness.

Additional perturbation-based evaluation and human assessment were therefore proposed.

---

# What is the primary contribution of this dissertation?

The dissertation contributes:
- a hybrid transformer-based vulnerability detection architecture
- an explainability framework
- critical evaluation of benchmark limitations
- experimentally grounded analysis of semantic and structural representation learning

within secure software engineering research.

---

# What are the main limitations?

Major limitations include:
- synthetic dataset bias
- limited real-world validation
- frozen transformer backbones
- computational constraints
- small-scale explainability evaluation

These limitations are explicitly acknowledged in the dissertation.

---

# Why is explainability important in vulnerability detection?

Security analysts require interpretable evidence to trust automated predictions.

Explainability improves:
- analyst confidence
- debugging capability
- transparency
- security workflow integration

particularly in high-stakes software security environments.

---

# Future Work

Potential future extensions include:
- evaluation on real-world datasets
- full end-to-end fine-tuning
- graph neural network integration
- dynamic analysis fusion
- stronger faithfulness evaluation methods
- larger-scale human reviewer studies

