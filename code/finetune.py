"""
Fine-tune CodeBERT / GraphCodeBERT on the rebuilt splits (CPU), or train the
frozen-encoder hybrid head, and evaluate on: the Juliet test split, its
length-matched subset, the 35-sample hand-written suite and the Devign test set.

usage: finetune.py MODEL LANG SCHEME SEED [--epochs E] [--max-train N]
  MODEL in {codebert, graphcodebert}
Writes out/ft_<model>_<lang>_<scheme>_s<seed>.json and keeps the classifier
weights in out/ckpt/ for the attention/perturbation analysis.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiments_cpu import length_match, metrics
from ood_suite import load_ood, load_devign

OUT = Path(__file__).resolve().parent / "out"
HF = {"codebert": "microsoft/codebert-base", "graphcodebert": "microsoft/graphcodebert-base"}
MAXLEN = 256
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(tok, texts):
    return tok(list(texts), truncation=True, max_length=MAXLEN, padding="longest",
               return_tensors="pt")


@torch.no_grad()
def predict(model, tok, texts, bs=64):
    """Batches are formed over length-sorted inputs (dynamic padding);
    probabilities are returned in the original order."""
    model.eval()
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])
    probs = np.zeros(len(texts))
    for i in range(0, len(texts), bs):
        sel = order[i:i + bs]
        enc = encode(tok, [texts[j] for j in sel]).to(DEV)
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            logits = model(**enc).logits
        probs[sel] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("lang")
    ap.add_argument("scheme")
    ap.add_argument("seed", type=int)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-test", type=int, default=0, help="smoke tests only")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(8)

    tr = pd.read_parquet(OUT / f"{args.lang}_{args.scheme}_train.parquet")
    te = pd.read_parquet(OUT / f"{args.lang}_{args.scheme}_test.parquet")
    if args.max_test:
        te = te.sample(args.max_test, random_state=0).reset_index(drop=True)
    if args.max_train and len(tr) > args.max_train:
        frac = min(1.0, args.max_train / len(tr))
        tr = pd.concat([g.sample(int(round(len(g) * frac)), random_state=args.seed)
                        for _, g in tr.groupby("label")])

    tok = AutoTokenizer.from_pretrained(HF[args.model])
    model = AutoModelForSequenceClassification.from_pretrained(HF[args.model], num_labels=2).to(DEV)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # fraction of inputs truncated at MAXLEN, per class (reviewer 2, point 2c)
    ntok = tr.code.map(lambda c: len(tok.tokenize(c)))
    trunc = {str(l): float((ntok[tr.label == l] > MAXLEN - 2).mean()) for l in (0, 1)}

    idx = np.arange(len(tr))
    steps_total = int(np.ceil(len(tr) / args.bs * args.epochs))
    step, t0 = 0, time.time()
    model.train()
    while step < steps_total:
        np.random.shuffle(idx)
        for i in range(0, len(idx), args.bs):
            b = tr.iloc[idx[i:i + args.bs]]
            enc = encode(tok, b.code).to(DEV)
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                out = model(**enc, labels=torch.tensor(b.label.to_numpy()).to(DEV))
            scaler.scale(out.loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{steps_total} loss {out.loss.item():.4f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step >= steps_total:
                break

    res = dict(model=args.model, lang=args.lang, scheme=args.scheme, seed=args.seed,
               epochs=args.epochs, train_n=len(tr), steps=steps_total,
               train_seconds=time.time() - t0, truncated_frac_by_label=trunc)
    p = predict(model, tok, te.code)
    res["test"] = metrics(te.label.to_numpy(), (p >= 0.5).astype(int))
    tem = length_match(te, np.random.default_rng(0))
    pm = p[te.index.get_indexer(tem.index)]
    res["test_len_matched"] = metrics(tem.label.to_numpy(), (pm >= 0.5).astype(int))
    # accuracy by length quintile of the test set
    q = pd.qcut(te["len"], 5, labels=False, duplicates="drop")
    res["test_by_len_quintile"] = {int(k): metrics(te.label[q == k].to_numpy(),
                                                   (p[(q == k).to_numpy()] >= 0.5).astype(int))
                                   for k in sorted(q.unique())}
    if args.lang == "cpp":
        ood = load_ood()
        po = predict(model, tok, ood.code)
        ood["pred"], ood["p_vuln"] = (po >= 0.5).astype(int), po
        res["ood35"] = metrics(ood.label.to_numpy(), ood.pred.to_numpy())
        res["ood35_by_category"] = {c: dict(n=len(g), correct=int((g.pred == g.label).sum()))
                                    for c, g in ood.groupby("category")}
        res["ood35_pred_vuln"] = int(ood.pred.sum())
        res["ood35_rows"] = ood[["id", "category", "label", "pred", "p_vuln"]].to_dict("records")
        dv = load_devign()
        pd_ = predict(model, tok, dv.code)
        res["devign"] = metrics(dv.label.to_numpy(), (pd_ >= 0.5).astype(int))
        res["devign_pred_vuln_frac"] = float((pd_ >= 0.5).mean())
    tag = f"{args.model}_{args.lang}_{args.scheme}_s{args.seed}"
    (OUT / f"ft_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    if args.seed == 0:
        (OUT / "ckpt").mkdir(exist_ok=True)
        model.save_pretrained(OUT / "ckpt" / tag)
        tok.save_pretrained(OUT / "ckpt" / tag)
    print(json.dumps({k: res[k] for k in ("test", "test_len_matched")}, default=float))


if __name__ == "__main__":
    main()
