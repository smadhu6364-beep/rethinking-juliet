
# Experiment Configurations

## Dataset

- Dataset: Juliet Test Suite
- Language: C/C++
- Labels:
  - 0 = benign
  - 1 = vulnerable
- Preprocessing:
  - function extraction
  - sanitization
  - dataset splitting

---

# Random Forest Baseline

## Configuration
- Feature extraction: TF-IDF
- Classifier: Random Forest
- Random state: 42

## Notes
- Used as classical ML baseline.
- Extremely high scores likely inflated by synthetic dataset regularities.

---

# CodeBERT

## Configuration
- Model: microsoft/codebert-base
- Max sequence length: 256
- Epochs: 1
- Learning rate: 2e-5
- Batch size: 4
- Weight decay: 0.01

## Hardware
- Google Colab GPU

---

# GraphCodeBERT

## Configuration
- Model: microsoft/graphcodebert-base
- Max sequence length: 256
- Epochs: 1
- Learning rate: 2e-5
- Batch size: 4
- Weight decay: 0.01

## Hardware
- Google Colab GPU

---

# Hybrid Model

## Configuration
- Architecture:
  - CodeBERT branch
  - GraphCodeBERT branch
  - CLS embedding concatenation
  - MLP classifier head

## Training Strategy
- Transformer backbones frozen
- Only classifier head trained initially

## Hyperparameters
- Learning rate: 2e-5
- Batch size: 2
- Epochs: 1

## Hardware
- Google Colab GPU

---

# Explainability Pipeline

## Components
- Attention extraction
- Token importance ranking
- Line-level suspiciousness scoring
- Natural language explanation generation

## Evaluation Plan
- Faithfulness assessment
- Human utility evaluation
- Reviewer agreement analysis

---

# Reproducibility Notes

- Random seed fixed where applicable.
- HuggingFace Transformers library used for transformer experiments.
- PyTorch backend used for training and inference.
- Experiments executed using Google Colab GPU environment.
