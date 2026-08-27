
# Abstract Draft Notes

Software vulnerabilities continue to pose significant security risks across modern software systems, motivating the need for accurate and explainable automated vulnerability detection approaches.

This dissertation investigates transformer-based software vulnerability detection using semantic, structural, and hybrid source code representations. The work evaluates classical machine learning baselines, pretrained transformer architectures, and explainability techniques within a unified experimental framework.

A hybrid architecture combining CodeBERT and GraphCodeBERT representations was implemented to explore complementary semantic and structural vulnerability-sensitive feature learning. The dissertation also proposes an attention-based explainability pipeline capable of generating token importance rankings, line-level suspiciousness scoring, and natural language explanations for model predictions.

Experiments were conducted using the Juliet Test Suite with preprocessing, sanitization, and controlled dataset splitting procedures. Results demonstrated extremely strong benchmark performance across transformer-based architectures, including near-perfect classification metrics on the sanitized dataset.

However, the dissertation critically examines the limitations of synthetic benchmark datasets and discusses the potential influence of structural regularities and dataset leakage on evaluation outcomes.

The work contributes:
- a hybrid transformer-based vulnerability detection framework
- an explainability pipeline for vulnerability prediction interpretation
- critical analysis of benchmark evaluation limitations
- experimentally grounded comparison of semantic and structural code representations

within the context of secure software engineering research.

Future work includes evaluation on real-world datasets, end-to-end hybrid fine-tuning, and stronger explainability faithfulness assessment methods.
