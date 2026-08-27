"""
Formal evaluation of the codebert-test checkpoint.
This is the model trained on leaky splits (train_sanitized.parquet).
Results go in the dissertation as the baseline / leaky-data model.
"""

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)
from tqdm import tqdm

# ---- Config ----
CHECKPOINT = "experiments/codebert-test/checkpoint-1000"
TEST_FILE  = "data/splits/test_dedup.parquet"
SAMPLE_N   = 1000  # use 1000 samples from test set for speed
LABEL_MAP  = {0: "SAFE", 1: "VULNERABLE"}

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Running on: {device}")

print(f"\nLoading test data from {TEST_FILE}...")
test_df = pd.read_parquet(TEST_FILE).sample(
    min(SAMPLE_N, len(pd.read_parquet(TEST_FILE))),
    random_state=42
)
print(f"Test samples: {len(test_df)}")
print(f"Label balance: {test_df['label'].value_counts(normalize=True).round(3).to_dict()}")

# ---- Run inference ----
def predict_batch(codes, batch_size=16):
    all_preds = []
    all_probs = []
    for i in tqdm(range(0, len(codes), batch_size)):
        batch = codes[i:i+batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())
    return all_preds, all_probs

print("\nRunning inference...")
codes = test_df['code'].tolist()
preds, probs = predict_batch(codes)
true_labels = test_df['label'].tolist()

# ---- Metrics ----
print("\n" + "="*60)
print("EVALUATION RESULTS — codebert-test checkpoint")
print("="*60)

acc  = accuracy_score(true_labels, preds)
prec = precision_score(true_labels, preds, zero_division=0)
rec  = recall_score(true_labels, preds, zero_division=0)
f1   = f1_score(true_labels, preds, zero_division=0)

print(f"\nAccuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, preds)
print(f"              Predicted SAFE  Predicted VULN")
print(f"Actual SAFE        {cm[0][0]:5d}           {cm[0][1]:5d}")
print(f"Actual VULN        {cm[1][0]:5d}           {cm[1][1]:5d}")

print("\nClassification Report:")
print(classification_report(
    true_labels, preds,
    target_names=["SAFE", "VULNERABLE"]
))

# Prediction distribution
n_safe = preds.count(0)
n_vuln = preds.count(1)
print(f"Predicted SAFE:       {n_safe}/{len(preds)} ({100*n_safe/len(preds):.1f}%)")
print(f"Predicted VULNERABLE: {n_vuln}/{len(preds)} ({100*n_vuln/len(preds):.1f}%)")
print(f"Actual SAFE:          {true_labels.count(0)}/{len(true_labels)}")
print(f"Actual VULNERABLE:    {true_labels.count(1)}/{len(true_labels)}")

# ---- Save ----
import json
results = {
    "model": "codebert-test",
    "data": "test_dedup.parquet",
    "n_samples": len(test_df),
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4),
    "confusion_matrix": cm.tolist(),
    "predicted_vulnerable": n_vuln,
    "predicted_safe": n_safe,
}

with open("results/codebert_test_evaluation.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/codebert_test_evaluation.json")