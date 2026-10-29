"""
Frozen-encoder hybrid: [CLS] embeddings of the pretrained CodeBERT and
GraphCodeBERT (no fine-tuning) are concatenated (1,536-d) and fed to an MLP
head (1536 -> 512 -> ReLU -> dropout 0.3 -> 2), trained with AdamW 2e-4,
batch 16, three epochs. Same 5,000-function stratified training sample as the
fine-tuned models (seed-matched), three seeds.

usage: hybrid.py LANG SCHEME
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from experiments_cpu import length_match, metrics
from finetune import DEV, HF, MAXLEN
from ood_suite import load_devign, load_ood

OUT = Path(__file__).resolve().parent / "out"
EMB = OUT / "emb"
SEEDS = [0, 1, 2]
N_TRAIN = int(os.environ.get("HYBRID_N_TRAIN", 5000))


@torch.no_grad()
def embed(name, texts, bs=64):
    tok = AutoTokenizer.from_pretrained(HF[name])
    enc_model = AutoModel.from_pretrained(HF[name]).to(DEV).eval()
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 768), dtype=np.float32)
    for i in range(0, len(texts), bs):
        sel = order[i:i + bs]
        enc = tok([texts[j] for j in sel], truncation=True, max_length=MAXLEN,
                  padding="longest", return_tensors="pt").to(DEV)
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            h = enc_model(**enc).last_hidden_state[:, 0]
        out[sel] = h.float().cpu().numpy()
    return out


def cached(key, name, texts):
    EMB.mkdir(parents=True, exist_ok=True)
    f = EMB / f"{key}_{name}.npy"
    if not f.exists():
        np.save(f, embed(name, texts))
    return np.load(f)


def both(key, texts):
    return np.hstack([cached(key, "codebert", texts), cached(key, "graphcodebert", texts)])


def train_head(X, y, seed):
    torch.manual_seed(seed)
    head = torch.nn.Sequential(torch.nn.Linear(1536, 512), torch.nn.ReLU(),
                               torch.nn.Dropout(0.3), torch.nn.Linear(512, 2))
    opt = torch.optim.AdamW(head.parameters(), lr=2e-4)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    g = torch.Generator().manual_seed(seed)
    for _ in range(3):
        perm = torch.randperm(len(Xt), generator=g)
        head.train()
        for i in range(0, len(Xt), 16):
            b = perm[i:i + 16]
            loss = torch.nn.functional.cross_entropy(head(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            opt.zero_grad()
    head.eval()
    return head


def run(lang, scheme):
    tr_full = pd.read_parquet(OUT / f"{lang}_{scheme}_train.parquet")
    te = pd.read_parquet(OUT / f"{lang}_{scheme}_test.parquet")
    Xte = both(f"{lang}_{scheme}_test", te.code)
    tem = length_match(te, np.random.default_rng(0))
    mi = te.index.get_indexer(tem.index)
    extra = {}
    if lang == "cpp":
        ood, dv = load_ood(), load_devign()
        extra = {"ood35": (both("ood35", ood.code), ood.label.to_numpy()),
                 "devign": (both("devign", dv.code), dv.label.to_numpy())}
    runs = []
    for seed in SEEDS:
        frac = min(1.0, N_TRAIN / len(tr_full))
        tr = pd.concat([g.sample(int(round(len(g) * frac)), random_state=seed)
                        for _, g in tr_full.groupby("label")])
        Xtr = both(f"{lang}_{scheme}_train_s{seed}", tr.code)
        head = train_head(Xtr, tr.label.to_numpy(), seed)
        with torch.no_grad():
            p = torch.softmax(head(torch.tensor(Xte)), -1)[:, 1].numpy()
            r = dict(seed=seed, test=metrics(te.label.to_numpy(), (p >= .5).astype(int)),
                     test_len_matched=metrics(tem.label.to_numpy(), (p[mi] >= .5).astype(int)))
            for k, (X, y) in extra.items():
                pk = torch.softmax(head(torch.tensor(X)), -1)[:, 1].numpy()
                r[k] = metrics(y, (pk >= .5).astype(int))
                r[k + "_pred_vuln"] = int((pk >= .5).sum())
        runs.append(r)
        print(lang, scheme, seed, "f1 %.4f" % r["test"]["f1"], flush=True)
    (OUT / f"hybrid_{lang}_{scheme}.json").write_text(json.dumps(runs, indent=1, default=float))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
