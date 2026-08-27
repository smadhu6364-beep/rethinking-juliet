
"""
Hybrid CodeBERT + GraphCodeBERT
vulnerability detection model.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

import pandas as pd

from datasets import Dataset

from transformers import (
    Trainer,
    TrainingArguments
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)



from transformers import (
    AutoModel,
    AutoTokenizer
)

# -----------------------------------------
# MODEL NAMES
# -----------------------------------------

CODEBERT_MODEL = "microsoft/codebert-base"

GRAPHCODEBERT_MODEL = "microsoft/graphcodebert-base"

RESULTS_PATH = Path("results/hybrid_evaluation.AFTER_label_fix.json")


print("Loading datasets...")

train_df = pd.read_parquet(
    "data/splits/train_dedup.parquet"
)

val_df = pd.read_parquet(
    "data/splits/val_dedup.parquet"
)

test_df = pd.read_parquet(
    "data/splits/test_dedup.parquet"
)

# small debug subset

train_df = train_df.sample(
    2000,
    random_state=42
)

val_df = val_df.sample(
    500,
    random_state=42
)

test_df = test_df.sample(
    500,
    random_state=42
)


# -----------------------------------------
# CONVERT TO HF DATASETS
# -----------------------------------------

train_dataset = Dataset.from_pandas(
    train_df
)

val_dataset = Dataset.from_pandas(
    val_df
)

# -----------------------------------------
# TOKENIZER
# -----------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    CODEBERT_MODEL
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

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

test_dataset = Dataset.from_pandas(
    test_df
)

test_dataset = test_dataset.map(
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

test_dataset = test_dataset.rename_column(
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

test_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)




# -----------------------------------------
# HYBRID MODEL
# -----------------------------------------

class HybridModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ---------------------------------
        # BACKBONES
        # ---------------------------------

        self.codebert = AutoModel.from_pretrained(
            CODEBERT_MODEL
        )

        self.graphcodebert = AutoModel.from_pretrained(
            GRAPHCODEBERT_MODEL
        )

        # ---------------------------------
        # FREEZE BACKBONES
        # ---------------------------------

        for param in self.codebert.parameters():
            param.requires_grad = False

        for param in self.graphcodebert.parameters():
            param.requires_grad = False

        # ---------------------------------
        # CLASSIFIER HEAD
        # ---------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    # -------------------------------------
    # FORWARD PASS
    # -------------------------------------

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):

        # CodeBERT embedding

        codebert_outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # GraphCodeBERT embedding

        graph_outputs = self.graphcodebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS embeddings

        codebert_cls = codebert_outputs.last_hidden_state[:, 0, :]

        graph_cls = graph_outputs.last_hidden_state[:, 0, :]

        # Concatenate embeddings

        combined = torch.cat(
            [codebert_cls, graph_cls],
            dim=1
        )

        # Classification

        logits = self.classifier(combined)

        loss = None

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


# -----------------------------------------
# TEST MODEL
# -----------------------------------------

print("Loading hybrid model...")

model = HybridModel()

print("Hybrid model loaded successfully.")


# -----------------------------------------
# DUMMY FORWARD PASS
# -----------------------------------------

print("Running forward pass test...")

dummy_input_ids = torch.randint(
    0,
    1000,
    (2, 256)
)

dummy_attention_mask = torch.ones(
    (2, 256)
)

dummy_labels = torch.tensor([0, 1])

outputs = model(
    input_ids=dummy_input_ids,
    attention_mask=dummy_attention_mask,
    labels=dummy_labels
)

print("Logits shape:")

print(outputs["logits"].shape)

print("Loss (should be a scalar):")

print(outputs["loss"])

outputs["loss"].backward()

print("Backward pass succeeded.")


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
    output_dir="experiments/hybrid-test",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=25,
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

print("Starting hybrid training...")

trainer.train()

# -----------------------------------------
# EVALUATE
# -----------------------------------------

print("\nEvaluating hybrid model...")

results = trainer.evaluate()

print(results)

print("\nEvaluating hybrid model on TEST set...")

test_results = trainer.evaluate(eval_dataset=test_dataset)

print(test_results)

# -----------------------------------------
# Save results to file
# -----------------------------------------

results_out = {
    "model": "hybrid_codebert_graphcodebert",
    "data": "train_dedup + val_dedup + test_dedup (debug subset: 2000 train / 500 val / 500 test)",
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



