
"""
Small CodeBERT sanity fine-tuning experiment.
"""

import json
from pathlib import Path

import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# -----------------------------------------
# LOAD DATASETS
# -----------------------------------------

print("Loading datasets...")

train_df = pd.read_parquet("data/splits/train_dedup.parquet")
val_df = pd.read_parquet("data/splits/val_dedup.parquet")
test_df = pd.read_parquet("data/splits/test_dedup.parquet")

# -----------------------------------------
# SMALL SUBSET
# -----------------------------------------

print("Creating small subsets...")

train_df = train_df.sample(
    4000,
    random_state=42
)

val_df = val_df.sample(
    1000,
    random_state=42
)
test_df = test_df.sample(
    1000,
    random_state=42
)

# -----------------------------------------
# CONVERT TO HF DATASETS
# -----------------------------------------

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
val_dataset = Dataset.from_pandas(val_df)

# -----------------------------------------
# TOKENIZER
# -----------------------------------------

MODEL_NAME = "microsoft/codebert-base"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# -----------------------------------------
# TOKENIZATION
# -----------------------------------------

def tokenize(batch):

    return tokenizer(
        batch["code"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

print("Tokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
  tokenize,
  batched=True)

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

# -----------------------------------------
# FORMAT
# -----------------------------------------

train_dataset = train_dataset.rename_column(
    "label",
    "labels"
)

val_dataset = val_dataset.rename_column(
    "label",
    "labels"
)

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

val_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

test_dataset = test_dataset.rename_column(
    "label",
    "labels"
)

test_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

# -----------------------------------------
# MODEL
# -----------------------------------------

print("Loading CodeBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# -----------------------------------------
# METRICS
# -----------------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = logits.argmax(axis=-1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary"
        )
    )

    acc = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# -----------------------------------------
# TRAINING CONFIG
# -----------------------------------------

training_args = TrainingArguments(
    output_dir="experiments/codebert-dedup",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    report_to="none"
)

# -----------------------------------------
# TRAINER
# -----------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# -----------------------------------------
# TRAIN
# -----------------------------------------

print("Starting training...")

trainer.train()

print("\nEvaluating on TEST set...")

test_results = trainer.evaluate(
    eval_dataset=test_dataset
)

print(test_results)

# -----------------------------------------
# EVALUATE
# -----------------------------------------

print("\nEvaluating model...")

results = trainer.evaluate()

print(results)

# -----------------------------------------
# Save results to file
# -----------------------------------------

RESULTS_PATH = Path("results/codebert_evaluation.AFTER_label_fix.json")

results_out = {
    "model": "codebert_dedup",
    "data": "train_dedup + val_dedup + test_dedup (debug subset: 4000 train / 1000 val / 1000 test)",
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "test_samples": len(test_df),
    "val_metrics": {k: (v.item() if hasattr(v, "item") else v) for k, v in results.items()},
    "test_metrics": {k: (v.item() if hasattr(v, "item") else v) for k, v in test_results.items()},
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(RESULTS_PATH, "w") as f:
    json.dump(results_out, f, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")

