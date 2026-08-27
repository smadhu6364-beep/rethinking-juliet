"""
Full-scale CodeBERT + GraphCodeBERT hybrid training.

This is the production pipeline (extracted from
notebooks/06_hybrid_model_training.ipynb) that produced the paper's
original hybrid model result. Unlike train_hybrid.py (a small
2000/500-sample debug smoke test), this trains on the full
train_dedup.parquet / test_dedup.parquet splits for 3 epochs.

Requires a GPU to finish in reasonable time -- do not run on a
CPU-only laptop. Intended for Google Colab or the DCU cluster.

Usage:
    python src/models/train_hybrid_full.py

Before running on Colab:
    1. Upload this file plus data/splits/train_dedup.parquet and
       data/splits/test_dedup.parquet (the corrected, post-label-fix
       versions) into the Colab working directory, preserving the
       "data/splits/" relative path (or edit TRAIN_PATH/TEST_PATH below).
    2. Runtime > Change runtime type > GPU.
    3. !pip install torch transformers scikit-learn pandas pyarrow tqdm -q
    4. !python train_hybrid_full.py
    5. Download results/hybrid_evaluation.AFTER_label_fix.json when done.
"""

import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from tqdm import tqdm

CODEBERT_MODEL = "microsoft/codebert-base"
GRAPHCODEBERT_MODEL = "microsoft/graphcodebert-base"

TRAIN_PATH = "data/splits/train_dedup.parquet"
TEST_PATH = "data/splits/test_dedup.parquet"

RESULTS_PATH = Path("results/hybrid_evaluation.AFTER_label_fix.json")
CHECKPOINT_DIR = Path("experiments/hybrid-full")

EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
MAX_LENGTH = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CodeDataset(Dataset):

    def __init__(self, df, tok_cb, tok_gcb):
        self.codes = df["code"].tolist()
        self.labels = df["label"].tolist()
        self.tok_cb = tok_cb
        self.tok_gcb = tok_gcb

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]

        enc_cb = self.tok_cb(
            code, max_length=MAX_LENGTH, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        enc_gcb = self.tok_gcb(
            code, max_length=MAX_LENGTH, padding="max_length",
            truncation=True, return_tensors="pt"
        )

        return {
            "cb_ids": enc_cb["input_ids"].squeeze(),
            "cb_mask": enc_cb["attention_mask"].squeeze(),
            "gcb_ids": enc_gcb["input_ids"].squeeze(),
            "gcb_mask": enc_gcb["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class HybridModel(nn.Module):

    def __init__(self, cb, gcb):
        super().__init__()

        self.cb = cb
        self.gcb = gcb

        for p in self.cb.parameters():
            p.requires_grad = False

        for p in self.gcb.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, cb_ids, cb_mask, gcb_ids, gcb_mask):
        cb_out = self.cb(
            input_ids=cb_ids, attention_mask=cb_mask
        ).last_hidden_state[:, 0, :]

        gcb_out = self.gcb(
            input_ids=gcb_ids, attention_mask=gcb_mask
        ).last_hidden_state[:, 0, :]

        return self.head(torch.cat([cb_out, gcb_out], dim=1))


def main():
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cpu":
        print(
            "WARNING: no GPU detected. This script trains on the full "
            "dataset for 3 epochs and will take a very long time on CPU. "
            "Strongly recommended to run this on Colab/cluster with a GPU."
        )

    print("Loading data...")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    print("Loading tokenizers and backbones...")

    tok_cb = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    tok_gcb = AutoTokenizer.from_pretrained(GRAPHCODEBERT_MODEL)
    cb = AutoModel.from_pretrained(CODEBERT_MODEL)
    gcb = AutoModel.from_pretrained(GRAPHCODEBERT_MODEL)

    train_loader = DataLoader(
        CodeDataset(train_df, tok_cb, tok_gcb),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        CodeDataset(test_df, tok_cb, tok_gcb),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = HybridModel(cb, gcb).to(DEVICE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()

    print("Training...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            cb_ids = batch["cb_ids"].to(DEVICE)
            cb_mask = batch["cb_mask"].to(DEVICE)
            gcb_ids = batch["gcb_ids"].to(DEVICE)
            gcb_mask = batch["gcb_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(cb_ids, cb_mask, gcb_ids, gcb_mask), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} loss: {total_loss / len(train_loader):.4f}")

    print("Evaluating...")

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            cb_ids = batch["cb_ids"].to(DEVICE)
            cb_mask = batch["cb_mask"].to(DEVICE)
            gcb_ids = batch["gcb_ids"].to(DEVICE)
            gcb_mask = batch["gcb_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            preds = torch.argmax(model(cb_ids, cb_mask, gcb_ids, gcb_mask), dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")

    results = {
        "model": "hybrid_codebert_graphcodebert_full",
        "data": "train_dedup (full) + test_dedup (full)",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion_matrix": cm.tolist(),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {RESULTS_PATH}")

    # The CodeBERT/GraphCodeBERT backbones are frozen, pretrained, and
    # reloadable identically from the Hub -- only the trained classifier
    # head is unique to this run, so that's what needs to be persisted.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.head.state_dict(), CHECKPOINT_DIR / "hybrid_head.pt")
    print(f"Checkpoint saved to {CHECKPOINT_DIR}/hybrid_head.pt")


if __name__ == "__main__":
    main()
