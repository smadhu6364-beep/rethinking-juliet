"""
RQ3 redone on a non-degenerate fine-tuned model (reviewer 2, point 2e).

For each test function: rank input tokens by final-layer [CLS] attention
(averaged over heads), then replace the top-k tokens with the tokenizer's
<mask> token. One token is swapped for one token, so input length is
unchanged. Control: k tokens chosen uniformly at random (same count, same
mask token). Report the mean drop in confidence of the originally predicted
class and the share of predictions that flip, for k in {1, 3, 5, 10}, on
all misclassified test items plus a random sample of correct ones.

usage: rq3.py TAG   (TAG = e.g. codebert_cpp_family_s0, a saved checkpoint)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune import DEV, MAXLEN

OUT = Path(__file__).resolve().parent / "out"
KS = [1, 3, 5, 10]


@torch.no_grad()
def probs(model, ids, att):
    with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
        out = model(input_ids=ids, attention_mask=att, output_attentions=True)
    return torch.softmax(out.logits.float(), -1), out.attentions[-1].float()


def main(tag):
    ck = OUT / "ckpt" / tag
    tok = AutoTokenizer.from_pretrained(ck)
    model = AutoModelForSequenceClassification.from_pretrained(
        ck, attn_implementation="eager").to(DEV).eval()
    _, lang, scheme, _ = tag.split("_")
    te = pd.read_parquet(OUT / f"{lang}_{scheme}_test.parquet")
    rng = np.random.default_rng(0)

    # first pass: predictions on the full test set
    enc_all = tok(list(te.code), truncation=True, max_length=MAXLEN, padding="max_length",
                  return_tensors="pt")
    preds = []
    for i in range(0, len(te), 64):
        p, _ = probs(model, enc_all.input_ids[i:i + 64].to(DEV), enc_all.attention_mask[i:i + 64].to(DEV))
        preds.append(p.argmax(-1).cpu().numpy())
    pred = np.concatenate(preds)
    wrong = np.where(pred != te.label.to_numpy())[0]
    right = np.where(pred == te.label.to_numpy())[0]
    sel = np.concatenate([wrong, rng.choice(right, min(1000, len(right)), replace=False)])

    special = set(tok.all_special_ids)
    rows = []
    for i in sel:
        ids = enc_all.input_ids[i:i + 1].clone().to(DEV)
        att = enc_all.attention_mask[i:i + 1].to(DEV)
        p0, a = probs(model, ids, att)
        c = int(p0.argmax())
        score = a[0, :, 0, :].mean(0).cpu().numpy()        # [CLS] row, head average
        valid = [j for j in range(int(att.sum())) if int(ids[0, j]) not in special]
        if len(valid) < 12:
            continue
        ranked = sorted(valid, key=lambda j: -score[j])
        r = dict(idx=int(i), correct=bool(pred[i] == te.label.iloc[i]), p0=float(p0[0, c]))
        for k in KS:
            for kind, pos in (("attn", ranked[:k]), ("rand", list(rng.choice(valid, k, replace=False)))):
                ids2 = ids.clone()
                ids2[0, pos] = tok.mask_token_id
                p1, _ = probs(model, ids2, att)
                r[f"{kind}{k}_delta"] = float(p0[0, c] - p1[0, c])
                r[f"{kind}{k}_flip"] = bool(int(p1.argmax()) != c)
        rows.append(r)
    df = pd.DataFrame(rows)
    res = dict(tag=tag, n=len(df), n_misclassified_in_test=int(len(wrong)), test_n=len(te))
    for k in KS:
        for kind in ("attn", "rand"):
            res[f"{kind}{k}"] = dict(mean_delta=float(df[f"{kind}{k}_delta"].mean()),
                                     flip_rate=float(df[f"{kind}{k}_flip"].mean()))
        # paired comparison attention vs random
        d = df[f"attn{k}_delta"] - df[f"rand{k}_delta"]
        res[f"attn_minus_rand{k}"] = dict(mean=float(d.mean()),
                                          se=float(d.std(ddof=1) / np.sqrt(len(d))))
    (OUT / f"rq3_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
