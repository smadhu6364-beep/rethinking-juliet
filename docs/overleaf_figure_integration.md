
# Overleaf Figure Integration Notes

# Figure 1 — Hybrid Architecture

## File
hybrid_architecture.png

## Suggested Chapter
Chapter 5 — Transformer and Hybrid Architectures

## Suggested Caption
Overview of the proposed hybrid vulnerability detection architecture combining semantic representations from CodeBERT and structural representations from GraphCodeBERT through CLS embedding concatenation and MLP classification.

## Suggested Label
fig:hybrid_architecture

---

# Figure 2 — Explainability Pipeline

## File
explainability_pipeline.png

## Suggested Chapter
Chapter 6 — Explainability Framework

## Suggested Caption
Attention-based explainability workflow used for token importance extraction, line-level suspiciousness scoring, and natural language explanation generation.

## Suggested Label
fig:explainability_pipeline

---

# Figure 3 — Model Comparison

## File
model_comparison.png

## Suggested Chapter
Chapter 7 — Experimental Results and Discussion

## Suggested Caption
Comparative performance visualization for Random Forest, CodeBERT, GraphCodeBERT, and Hybrid transformer models on the sanitized Juliet dataset.

## Suggested Label
fig:model_comparison

---

# Figure 4 — Confusion Matrix

## File
confusion_matrix.png

## Suggested Chapter
Chapter 7 — Experimental Results and Discussion

## Suggested Caption
Confusion matrix illustrating vulnerable and benign classification outcomes for the hybrid transformer architecture.

## Suggested Label
fig:confusion_matrix

---

# Figure 5 — ROC Curve

## File
roc_curve.png

## Suggested Chapter
Chapter 7 — Experimental Results and Discussion

## Suggested Caption
Receiver Operating Characteristic (ROC) curve demonstrating classification discrimination capability of the vulnerability detection framework.

## Suggested Label
fig:roc_curve

---

# Suggested LaTeX Template

\begin{figure}[h]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/FILENAME}
    \caption{CAPTION}
    \label{fig:LABEL}
\end{figure}
