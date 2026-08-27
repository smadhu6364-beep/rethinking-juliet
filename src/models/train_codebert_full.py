"""
Full-scale CodeBERT fine-tuning on the corrected Juliet C/C++ dedup splits.

Mirrors train_graphcodebert_renku.py, but for CodeBERT. Unlike
train_codebert.py (a 4000/1000/1000-sample debug smoke test), this trains
on the full train_dedup.parquet / val_dedup.parquet / test_dedup.parquet
splits for 3 epochs -- the controlled, end-to-end retrain on corrected
labels needed for a trustworthy comparison against the pre-fix number in
results/codebert_test_evaluation.json.

Requires a GPU to finish in reasonable time. Intended for Colab/cluster,
not a CPU-only laptop.

Run: python src/models/train_codebert_full.py
"""

import pandas as pd
import numpy as np
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    f1_score, accuracy_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)

# ---- Config ---
MODEL_NAME = "microsoft/codebert-base"
TRAIN_FILE = "data/splits/train_dedup.parquet"
VAL_FILE = "data/splits/val_dedup.parquet"
TEST_FILE = "data/splits/test_dedup.parquet"
OUTPUT_DIR = "experiments/codebert-full"
RESULTS_FILE = "results/codebert_evaluation.AFTER_label_fix.json"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

print("=" * 60)
print("CodeBERT Training — full corrected C/C++ retrain")
print("=" * 60)
print(f"Model:   {MODEL_NAME}")
print(f"Epochs:  {EPOCHS}")
print(f"Batch:   {BATCH_SIZE}")
print(f"Device:  {'GPU' if torch.cuda.is_available() else 'CPU'}")
print()

# ---- Load data ---
print("Loading data...")
train_df = pd.read_parquet(TRAIN_FILE)
val_df = pd.read_parquet(VAL_FILE)
test_df = pd.read_parquet(TEST_FILE)
print(f"Train: {len(train_df):,} rows | "
      f"balance: {train_df['label'].value_counts(normalize=True).round(3).to_dict()}")
print(f"Val:   {len(val_df):,} rows")
print(f"Test:  {len(test_df):,} rows")

# ---- Tokenise ---
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["code"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )


print("Tokenising datasets (this takes a few minutes)...")
train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds = val_ds.rename_column("label", "labels")
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds = test_ds.rename_column("label", "labels")
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# ---- Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1": round(f1_score(labels, preds, average="weighted"), 4),
        "precision": round(precision_score(labels, preds, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(labels, preds, average="weighted", zero_division=0), 4),
    }


# ---- Model ---
print("\nLoading CodeBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# ---- Training args ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# ---- Train ---
print("\nStarting training...")
print(f"Total steps: ~{len(train_ds) * EPOCHS // BATCH_SIZE}")
trainer.train()
print("Training complete.")

# ---- Test evaluation ---
print("\nEvaluating on test set...")
results = trainer.predict(test_ds)
preds = results.predictions.argmax(axis=1)
labels = test_df["label"].values

acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds, zero_division=0)
rec = recall_score(labels, preds, zero_division=0)
f1 = f1_score(labels, preds, zero_division=0)
cm = confusion_matrix(labels, preds)

print("\n" + "=" * 60)
print("CODEBERT TEST RESULTS — full corrected C/C++ retrain")
print("=" * 60)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(f"  Actual SAFE  predicted SAFE:  {cm[0][0]}")
print(f"  Actual SAFE  predicted VULN:  {cm[0][1]}")
print(f"  Actual VULN  predicted SAFE:  {cm[1][0]}")
print(f"  Actual VULN  predicted VULN:  {cm[1][1]}")
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["SAFE", "VULNERABLE"]))

# ---- Save ---
output = {
    "model": MODEL_NAME,
    "checkpoint": OUTPUT_DIR,
    "data": "train_dedup (full) + val_dedup (full) + test_dedup (full), post label-fix",
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "test_samples": len(test_df),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4),
    "confusion_matrix": cm.tolist(),
}

os.makedirs("results", exist_ok=True)
with open(RESULTS_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {RESULTS_FILE}")
