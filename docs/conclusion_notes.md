
# Conclusion Chapter Notes

# Dissertation Summary

This dissertation investigated transformer-based software vulnerability detection and explainable AI techniques within secure software engineering research.

The work explored:
- classical machine learning baselines
- semantic transformer architectures
- structural transformer architectures
- hybrid representation learning
- explainability mechanisms

for automated vulnerability analysis.

---

# Key Findings

Transformer-based models demonstrated extremely strong performance on the sanitized Juliet dataset.

CodeBERT successfully captured semantic source code patterns, while GraphCodeBERT incorporated structural relationships through graph-aware representations.

A hybrid architecture combining semantic and structural transformer embeddings was implemented and experimentally validated.

The explainability pipeline further demonstrated the ability to generate:
- token importance rankings
- suspicious code line identification
- natural language explanation outputs

for vulnerability predictions.

---

# Critical Evaluation

Although experimental results achieved near-perfect benchmark metrics, the dissertation critically examined the limitations of synthetic datasets such as Juliet.

The work argues that benchmark performance alone does not necessarily imply robust real-world vulnerability detection capability.

Dataset leakage, structural regularities, and limited real-world validation remain important concerns within vulnerability detection research.

---

# Contributions

The dissertation contributed:
- a hybrid transformer-based vulnerability detection architecture
- an explainability framework for vulnerability prediction interpretation
- reproducible preprocessing and evaluation pipelines
- critical analysis of benchmark evaluation limitations

within the context of secure software engineering.

---

# Limitations

Several limitations were identified, including:
- reliance on synthetic benchmark data
- frozen transformer backbones
- computational resource constraints
- limited explainability evaluation scale
- absence of real-world deployment testing

These limitations were explicitly acknowledged throughout the evaluation process.

---

# Future Work

Future research directions include:
- evaluation on real-world vulnerability datasets
- end-to-end hybrid fine-tuning
- graph neural network integration
- dynamic analysis incorporation
- stronger explainability faithfulness evaluation
- larger-scale human reviewer studies

Further work is required to determine whether transformer-based vulnerability detection approaches generalize reliably beyond controlled benchmark environments.

---

# Final Statement

The dissertation demonstrates that transformer-based architectures and explainability techniques provide promising directions for automated software vulnerability detection research.

At the same time, the work highlights the importance of methodological transparency, critical evaluation, and realistic benchmarking when developing AI-based software security systems.
