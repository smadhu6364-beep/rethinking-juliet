
# Experiment Tracking Log

## Random Forest Baseline

### Configuration
- TF-IDF features
- Random Forest classifier
- Juliet dataset
- Sanitized preprocessing

### Results
| Metric | Score |
|---|---|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 | TBD |

### Observations
- Near-perfect scores likely caused by dataset leakage and structural regularities in Juliet.
- Baseline does not generalize reliably to real-world vulnerability detection.

---

# CodeBERT

### Configuration
- Model: microsoft/codebert-base
- Epochs: 1
- Batch size: 4
- Max length: 256

### Results
| Metric | Score |
|---|---|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 | TBD |

### Observations
- Transformer captures semantic information from source code.
- Strong performance observed on sanitized dataset.

---

# GraphCodeBERT

### Configuration
- Model: microsoft/graphcodebert-base
- Epochs: 1
- Batch size: 4
- Max length: 256

### Results
| Metric | Score |
|---|---|
| Accuracy | 1.0 |
| Precision | 1.0 |
| Recall | 1.0 |
| F1 | 1.0 |

### Observations
- Graph-aware transformer successfully trained.
- Perfect scores likely influenced by benchmark regularities.
- Structural representation may improve vulnerability-sensitive reasoning.

---

# Hybrid Model

### Configuration
- CodeBERT + GraphCodeBERT
- Frozen transformer backbones
- MLP classifier head
- CLS embedding concatenation

### Results
| Metric | Score |
|---|---|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 | TBD |

### Observations
- Hybrid architecture combines semantic and structural embeddings.
- Intended to evaluate complementary representation learning.

---

# Explainability Pipeline

### Components
- Attention extraction
- Token importance ranking
- Line-level suspiciousness scoring
- Explanation generation

### Observations
- Attention-based explanations operational.
- Requires human evaluation for usefulness and faithfulness assessment.

---

# General Dissertation Notes

- Juliet dataset may inflate performance due to synthetic regularities.
- Sanitization reduces direct leakage but cannot eliminate all structural artifacts.
- Real-world generalization remains an acknowledged limitation.

