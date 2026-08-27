"""
Independent verification: load the existing experiments/hybrid-{java,csharp}
checkpoints (frozen CodeBERT+GraphCodeBERT backbones + trained MLP head) and
re-run evaluation on the real test splits, to check whether the checkpoint
actually reproduces the paper's reported F1 (Java 0.8619, C# 0.9527) --
inference only, no training, so it's feasible on CPU.

Run: python verify_hybrid_multilang.py --language java
     python verify_hybrid_multilang.py --language csharp
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,
)

parser = argparse.ArgumentParser()
parser.add_argument("--language", required=True, choices=["java", "csharp"])
args = parser.parse_args()

CODEBERT_MODEL = "microsoft/codebert-base"
GRAPHCODEBERT_MODEL = "microsoft/graphcodebert-base"
SPLITS_DIR = Path("data/processed/multilang") / args.language
CHECKPOINT = Path(f"experiments/hybrid-{args.language}/hybrid_head.pt")
MAX_LENGTH = 256
BATCH_SIZE = 16


class CodeDataset(Dataset):
    def __init__(self, df, tok_cb, tok_gcb):
        self.codes = df["source_code"].tolist()
        self.labels = df["label"].tolist()
        self.tok_cb = tok_cb
        self.tok_gcb = tok_gcb

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        enc_cb = self.tok_cb(code, max_length=MAX_LENGTH, padding="max_length",
                              truncation=True, return_tensors="pt")
        enc_gcb = self.tok_gcb(code, max_length=MAX_LENGTH, padding="max_length",
                                truncation=True, return_tensors="pt")
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
        self.head = nn.Sequential(
            nn.Linear(768 * 2, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 2),
        )

    def forward(self, cb_ids, cb_mask, gcb_ids, gcb_mask):
        cb_out = self.cb(input_ids=cb_ids, attention_mask=cb_mask).last_hidden_state[:, 0, :]
        gcb_out = self.gcb(input_ids=gcb_ids, attention_mask=gcb_mask).last_hidden_state[:, 0, :]
        return self.head(torch.cat([cb_out, gcb_out], dim=1))


def main():
    t0 = time.time()
    print(f"Language: {args.language}")
    print(f"Loading checkpoint: {CHECKPOINT} (exists={CHECKPOINT.exists()})")

    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")
    print(f"Test samples: {len(test_df):,}")

    tok_cb = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    tok_gcb = AutoTokenizer.from_pretrained(GRAPHCODEBERT_MODEL)
    cb = AutoModel.from_pretrained(CODEBERT_MODEL)
    gcb = AutoModel.from_pretrained(GRAPHCODEBERT_MODEL)

    model = HybridModel(cb, gcb)
    state_dict = torch.load(CHECKPOINT, map_location="cpu")
    model.head.load_state_dict(state_dict)
    model.eval()

    test_loader = DataLoader(CodeDataset(test_df, tok_cb, tok_gcb), batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    n_batches = len(test_loader)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            preds = torch.argmax(
                model(batch["cb_ids"], batch["cb_mask"], batch["gcb_ids"], batch["gcb_mask"]),
                dim=1,
            )
            all_preds.extend(preds.numpy())
            all_labels.extend(batch["label"].numpy())
            if (i + 1) % 20 == 0 or (i + 1) == n_batches:
                elapsed = time.time() - t0
                print(f"  batch {i+1}/{n_batches}  ({elapsed:.0f}s elapsed)")

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 60)
    print(f"VERIFICATION RESULTS — hybrid_{args.language}")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print(f"Total time: {time.time() - t0:.0f}s")

    out = {
        "language": args.language,
        "checkpoint": str(CHECKPOINT),
        "test_samples": len(test_df),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion_matrix": cm.tolist(),
    }
    out_path = Path(f"results/hybrid_{args.language}_verification.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
