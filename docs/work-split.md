# Hybrid Transformer-Based Software Vulnerability Detection with Explainable AI

## Complete Project Progress Report

---

# 1. PROJECT OVERVIEW

## Dissertation Topic

```text id="m7q2vx"
Hybrid Transformer-Based Software Vulnerability Detection with Explainable AI
```

## Core Objective

Build and evaluate:

* transformer-based vulnerability detection models
* hybrid semantic + structural architectures
* explainability mechanisms for vulnerability prediction interpretation

using:

* CodeBERT
* GraphCodeBERT
* attention-based explainability

---

# 2. DATASET PIPELINE

## Dataset Used

```text id="x4m8wr"
Juliet Test Suite (C/C++)
```

## Labels

| Label | Meaning    |
| ----- | ---------- |
| 0     | Benign     |
| 1     | Vulnerable |

---

## Preprocessing Completed

### Function Extraction

Implemented:

```text id="r8m2vn"
extract_functions.py
```

Purpose:

* extract vulnerable/benign functions
* normalize code samples

---

### Sanitization Pipeline

Implemented:

```text id="j2m8vn"
sanitize_code.py
```

Purpose:

* reduce direct leakage
* remove obvious vulnerable naming artifacts

---

### Dataset Splitting

Implemented:

* `create_splits.py`
* `rebuild_splits.py`
* `rebuild_splits_light.py`

Generated:

* train_sanitized.parquet
* val_sanitized.parquet
* test_sanitized.parquet

---

# 3. BASELINE MODEL

## Random Forest Baseline

Implemented:

```text id="p8m3vq"
train_random_forest.py
```

## Pipeline

* TF-IDF features
* Random Forest classifier
* sanitized Juliet dataset

---

## Observed Results

| Metric    | Approximate Score |
| --------- | ----------------- |
| Accuracy  | ~0.99             |
| Precision | ~0.99             |
| Recall    | ~0.99             |
| F1        | ~0.99             |

---

## Critical Interpretation

Important finding:

```text id="k5n8wr"
near-perfect scores likely inflated by dataset leakage and synthetic structural regularities
```

This became a major dissertation discussion point.

---

# 4. CODEBERT MODEL

## Model

```text id="m4q8vk"
microsoft/codebert-base
```

## Training Script

```text id="x7m2wr"
train_codebert.py
```

---

## Configuration

| Parameter     | Value |
| ------------- | ----- |
| Max Length    | 256   |
| Epochs        | 1     |
| Learning Rate | 2e-5  |
| Batch Size    | 4     |
| Weight Decay  | 0.01  |

---

## Dataset

* train_sanitized.parquet
* val_sanitized.parquet

---

## Hardware

```text id="pchey9"
Google Colab GPU
```

---

## Results

Observed:

```text id="3t24fm"
very high transformer performance
```

Approximate:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~0.99 |
| Precision | ~0.99 |
| Recall    | ~0.99 |
| F1        | ~0.99 |

---

# 5. GRAPHCODEBERT MODEL

## Model

```text id="cijidr"
microsoft/graphcodebert-base
```

## Training Script

```text id="4s9k9x"
train_graphcodebert.py
```

---

## Configuration

| Parameter     | Value |
| ------------- | ----- |
| Max Length    | 256   |
| Epochs        | 1     |
| Learning Rate | 2e-5  |
| Batch Size    | 4     |
| Weight Decay  | 0.01  |

---

## Hardware

```text id="otpmkd"
Google Colab GPU
```

---

# OFFICIAL OBSERVED RESULTS

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 1.0   |
| Precision | 1.0   |
| Recall    | 1.0   |
| F1        | 1.0   |

---

## Critical Interpretation

Dissertation argument:

```text id="m7q2vx"
perfect benchmark performance does NOT imply real-world robustness
```

Reasons:

* synthetic dataset artifacts
* structural regularities
* residual leakage
* benchmark inflation

This became:

```text id="x4m8wr"
one of the strongest critical analysis sections
```

---

# 6. HYBRID MODEL

## Architecture

Implemented:

```text id="r8m2vn"
train_hybrid.py
```

---

# Hybrid Design

## Branch 1

```text id="j2m8vn"
CodeBERT semantic embeddings
```

## Branch 2

```text id="p8m3vq"
GraphCodeBERT structural embeddings
```

---

## Fusion Strategy

Implemented:

```text id="k5n8wr"
CLS embedding concatenation
```

Then passed into:

* MLP classifier
* ReLU activation
* Dropout
* binary classification output

---

## Freezing Strategy

Both transformer backbones:

```text id="m4q8vk"
frozen initially
```

Reason:

* reduced GPU memory
* stabilized training
* faster experimentation

---

## Architecture Validation

Successfully validated:

```text id="x7m2wr"
forward pass operational
```

Observed output:

```text id="pchey9"
torch.Size([2, 2])
```

This confirmed:

* dual-transformer integration works
* tensor dimensions correct
* hybrid pipeline operational

---

# 7. EXPLAINABILITY PIPELINE

## Files Implemented

| File                    | Purpose                       |
| ----------------------- | ----------------------------- |
| extract_attention.py    | attention extraction          |
| line_attention.py       | line suspiciousness scoring   |
| generate_explanation.py | natural language explanations |

---

# Attention Extraction

Successfully extracted:

* transformer attention tensors
* token importance rankings
* suspicious token signals

Observed:

```text id="3t24fm"
12 attention layers
```

Tensor shape:

```text id="cijidr"
torch.Size([1, 12, 194, 194])
```

---

# Line-Level Suspiciousness

Generated outputs like:

| Suspicious Line   | Score    |
| ----------------- | -------- |
| data = NULL;      | 0.111111 |
| printWLine(data); | 0.090909 |
| FUNC(dataList);   | 0.090909 |

---

# Generated Explanations

System successfully produced:

* vulnerability descriptions
* suspicious operation analysis
* human-readable explanations

Example:

```text id="4s9k9x"
Potential buffer overflow vulnerability detected.
```

---

# 8. VISUALIZATION SYSTEM

## Generated Figures

| Figure                          | Status |
| ------------------------------- | ------ |
| Model comparison chart          | ✅      |
| Hybrid architecture diagram     | ✅      |
| Explainability pipeline diagram | ✅      |
| Confusion matrix                | ✅      |
| ROC curve                       | ✅      |

---

# Architecture Diagram

Successfully visualized:

* CodeBERT branch
* GraphCodeBERT branch
* concatenation
* MLP classifier

---

# Explainability Diagram

Successfully visualized:

```text id="otpmkd"
Code → Attention Extraction → Line Scoring → Explanation
```

---

# 9. LIVE DEMO SYSTEM

Implemented:

```text id="m7q2vx"
demo_system.py
```

---

# Demo Capability

Input:

```c id="x4m8wr"
strcpy(buffer, user_input);
```

Output:

* prediction
* suspicious lines
* natural language explanation

---

# Example Demo Output

Prediction:

```text id="r8m2vn"
VULNERABLE
```

Confidence:

```text id="j2m8vn"
0.98
```

Explanation:

```text id="p8m3vq"
Potential buffer overflow vulnerability detected.
```

---

# 10. DISSERTATION STATUS

## Completed Academic Assets

| Asset                     | Status |
| ------------------------- | ------ |
| methodology notes         | ✅      |
| results notes             | ✅      |
| conclusion notes          | ✅      |
| threats to validity       | ✅      |
| viva preparation          | ✅      |
| contribution statement    | ✅      |
| experiment configurations | ✅      |
| figure integration notes  | ✅      |

---

# 11. KEY RESEARCH CONTRIBUTIONS

## Contribution 1

```text id="k5n8wr"
Hybrid semantic + structural transformer architecture
```

---

## Contribution 2

```text id="m4q8vk"
Explainability framework for vulnerability prediction interpretation
```

---

## Contribution 3

```text id="x7m2wr"
Critical analysis of synthetic benchmark inflation
```

---

## Contribution 4

```text id="pchey9"
Reproducible preprocessing + evaluation pipeline
```

---

# 12. CURRENT PROJECT STATUS

| Area                     | Completion |
| ------------------------ | ---------- |
| Technical Implementation | ~80%       |
| Experimental Framework   | ~75%       |
| Dissertation Writing     | ~55-60%    |
| Overall Dissertation     | ~70-75%    |

---

# 13. MOST IMPORTANT ACHIEVEMENT

You successfully moved from:

```text id="3t24fm"
isolated training scripts
```

to:

```text id="cijidr"
a complete explainable hybrid vulnerability detection research system
```

That includes:

* ML baselines
* transformers
* hybrid architectures
* explainability
* evaluation
* visualization
* dissertation narrative
* viva preparation

That is a very substantial MSc-level project.
